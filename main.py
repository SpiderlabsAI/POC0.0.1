# morpheus_vm.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from encryption_utils import encrypt, decrypt
from data_loader import load_encrypted_data
from web3 import Web3, HTTPProvider
import json
import os
import requests

# Define model segment for Subnet A
class ModelSegmentA(nn.Module):
    def __init__(self):
        super(ModelSegmentA, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Input layer to cut layer

    def forward(self, x):
        return self.layer1(x**2)  # Polynomial activation function

# Define model segment for Subnet B
class ModelSegmentB(nn.Module):
    def __init__(self):
        super(ModelSegmentB, self).__init__()
        self.layer2 = nn.Linear(128, 10)  # From cut layer to output

    def forward(self, x):
        return self.layer2(x**2)  # Polynomial activation function

# Initialize Subnet A and B Models
model_a = ModelSegmentA()
model_b = ModelSegmentB()
optimizer_a = optim.SGD(model_a.parameters(), lr=0.01)
optimizer_b = optim.SGD(model_b.parameters(), lr=0.01)

# Load Avalanche RPC endpoint and AWM smart contract
AVAX_RPC_URL = os.getenv("AVAX_RPC_URL")
if not AVAX_RPC_URL:
    raise EnvironmentError("AVAX_RPC_URL is not set in environment variables.")

web3 = Web3(HTTPProvider(AVAX_RPC_URL))
with open("AWM_ABI.json") as f:
    awm_abi = json.load(f)
awm_contract_address_a = os.getenv("AWM_CONTRACT_ADDRESS_A")
awm_contract_address_b = os.getenv("AWM_CONTRACT_ADDRESS_B")
if not awm_contract_address_a or not awm_contract_address_b:
    raise EnvironmentError("AWM_CONTRACT_ADDRESS_A or AWM_CONTRACT_ADDRESS_B is not set in environment variables.")

awm_contract_a = web3.eth.contract(address=awm_contract_address_a, abi=awm_abi)
awm_contract_b = web3.eth.contract(address=awm_contract_address_b, abi=awm_abi)

# Function for forward propagation on Subnet A
def forward_propagate_on_subnet_a(input_data):
    encrypted_input = encrypt(input_data)
    input_tensor = torch.tensor(encrypted_input, dtype=torch.float32)
    output = model_a(input_tensor)
    return output

# Function to send activations to Subnet B
def send_activations_to_subnet_b(activations):
    wallet_address_a = os.getenv("WALLET_ADDRESS_A")
    private_key_a = os.getenv("PRIVATE_KEY_A")
    subnet_b_address = os.getenv("SUBNET_B_ADDRESS")

    if not wallet_address_a or not private_key_a or not subnet_b_address:
        raise EnvironmentError("WALLET_ADDRESS_A, PRIVATE_KEY_A, or SUBNET_B_ADDRESS is not set in environment variables.")

    transaction = awm_contract_a.functions.sendMessage(
        subnet_b_address,
        json.dumps(activations.tolist())
    ).buildTransaction({
        'chainId': 43114,  # Avalanche Mainnet Chain ID
        'gas': 70000,
        'gasPrice': web3.toWei('2', 'gwei'),
        'nonce': web3.eth.getTransactionCount(wallet_address_a)
    })
    # Sign and send the transaction
    signed_txn = web3.eth.account.sign_transaction(transaction, private_key=private_key_a)
    tx_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    print(f"Transaction hash for sending activations: {web3.toHex(tx_hash)}")

# Function to receive activations in Subnet B
def receive_activations_from_subnet_a():
    subnet_b_address = os.getenv("SUBNET_B_ADDRESS")
    if not subnet_b_address:
        raise EnvironmentError("SUBNET_B_ADDRESS is not set in environment variables.")

    # Fetch received messages from the contract
    events = awm_contract_b.events.MessageReceived.createFilter(fromBlock='latest', argument_filters={"receiver": subnet_b_address}).get_all_entries()
    if events:
        message_data = events[-1]['args']['message']
        return json.loads(message_data)
    else:
        raise ValueError("No new messages received from Subnet A.")

# Function for forward propagation on Subnet B
def forward_propagate_on_subnet_b(encrypted_activations):
    decrypted_activations = decrypt(encrypted_activations)
    activations_tensor = torch.tensor(decrypted_activations, dtype=torch.float32)
    output = model_b(activations_tensor)
    return output

# Function to send gradients back to Subnet A
def send_gradients_to_subnet_a(gradients):
    wallet_address_b = os.getenv("WALLET_ADDRESS_B")
    private_key_b = os.getenv("PRIVATE_KEY_B")
    subnet_a_address = os.getenv("SUBNET_A_ADDRESS")

    if not wallet_address_b or not private_key_b or not subnet_a_address:
        raise EnvironmentError("WALLET_ADDRESS_B, PRIVATE_KEY_B, or SUBNET_A_ADDRESS is not set in environment variables.")

    transaction = awm_contract_b.functions.sendMessage(
        subnet_a_address,
        json.dumps([grad.tolist() for grad in gradients])
    ).buildTransaction({
        'chainId': 43114,  # Avalanche Mainnet Chain ID
        'gas': 70000,
        'gasPrice': web3.toWei('2', 'gwei'),
        'nonce': web3.eth.getTransactionCount(wallet_address_b)
    })
    # Sign and send the transaction
    signed_txn = web3.eth.account.sign_transaction(transaction, private_key=private_key_b)
    tx_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    print(f"Transaction hash for sending gradients: {web3.toHex(tx_hash)}")

if __name__ == "__main__":
    try:
        # Step 1: Load and encrypt the dataset
        encrypted_data = load_encrypted_data()

        # Step 2: Perform forward pass on Subnet A
        activations = forward_propagate_on_subnet_a(encrypted_data)

        # Step 3: Send activations to Subnet B
        send_activations_to_subnet_b(activations)

        # Step 4: Receive activations in Subnet B
        received_activations = receive_activations_from_subnet_a()

        # Step 5: Perform forward propagation and calculate loss on Subnet B
        output = forward_propagate_on_subnet_b(received_activations)
        target = torch.tensor([1], dtype=torch.float32)  # Placeholder target for loss calculation
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        # Step 6: Backward pass to calculate gradients on Subnet B
        loss.backward()
        gradients = [param.grad for param in model_b.parameters()]

        # Step 7: Send encrypted gradients back to Subnet A
        encrypted_gradients = [encrypt(grad.numpy()) for grad in gradients]
        send_gradients_to_subnet_a(encrypted_gradients)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
