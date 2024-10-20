# Morpheus VM: Decentralized Machine Learning with Avalanche Warp Messaging (AWM)

## Overview
Morpheus VM is a Proof of Concept (PoC) implementation of a decentralized machine learning system using split learning across two Avalanche L1 blockchains (Subnets). The code leverages Avalanche Warp Messaging (AWM) to facilitate communication between subnets while ensuring data privacy using Fully Homomorphic Encryption (FHE). The project aims to showcase how decentralized learning can be performed using Avalanche subnets, maintaining data security and interoperability.

## Features
- **Split Learning**: The neural network model is split between two subnets, with each subnet responsible for a different segment of the model.
- **Avalanche Warp Messaging (AWM)**: Used for secure inter-subnet communication to transfer encrypted data, activations, and gradients.
- **Fully Homomorphic Encryption (FHE)**: Ensures that all computations are performed on encrypted data, preserving data privacy throughout the learning process.

## System Requirements
- Python 3.x
- Avalanche Network RPC endpoint
- Environment variables for Avalanche addresses and private keys

## Dependencies
The following dependencies are required for the Morpheus VM project:

- `torch`: For implementing neural network models and handling machine learning operations.
- `web3`: For interacting with Avalanche smart contracts via an RPC endpoint.
- `numpy`: For data handling and numeric operations.
- `Pyfhel`: For Fully Homomorphic Encryption (FHE) operations.
- `requests`: For making HTTP requests (used in auxiliary functions).

## Installation
To install the dependencies, use the `requirements.txt` file provided:

```sh
pip install -r requirements.txt
```

## Environment Setup
To run Morpheus VM, ensure that you have set the following environment variables:

- `AVAX_RPC_URL`: RPC endpoint for Avalanche network.
- `AWM_CONTRACT_ADDRESS_A`: Address of the Avalanche Warp Messaging contract for Subnet A.
- `AWM_CONTRACT_ADDRESS_B`: Address of the Avalanche Warp Messaging contract for Subnet B.
- `WALLET_ADDRESS_A`: Wallet address for Subnet A.
- `PRIVATE_KEY_A`: Private key for signing transactions from Subnet A.
- `WALLET_ADDRESS_B`: Wallet address for Subnet B.
- `PRIVATE_KEY_B`: Private key for signing transactions from Subnet B.
- `SUBNET_A_ADDRESS` and `SUBNET_B_ADDRESS`: Addresses for Subnet A and Subnet B.

Ensure that you have the Avalanche smart contract ABI file (`AWM_ABI.json`) available in the root directory.

## Running the Code
To run the Morpheus VM PoC:

1. Set up your environment variables.
2. Deploy the Avalanche Warp Messaging contracts on both subnets.
3. Run the code using Python:

```sh
python morpheus_vm.py
```

## Code Flow
The main code (`morpheus_vm.py`) executes the following steps:

1. **Load and Encrypt Data**: The dataset is loaded and encrypted using FHE.
2. **Subnet A Forward Propagation**: The first segment of the model performs forward propagation on encrypted input data.
3. **Send Activations to Subnet B**: The encrypted activations are sent to Subnet B using Avalanche Warp Messaging.
4. **Subnet B Forward Propagation**: The second segment of the model receives the encrypted activations and performs forward propagation.
5. **Loss Calculation and Backpropagation**: The loss is computed, and backpropagation is performed on Subnet B.
6. **Send Gradients Back to Subnet A**: The encrypted gradients are sent back to Subnet A for weight updates.

## Notes
- This project is a Proof of Concept and should be treated as such. Real-world implementation will require optimizations and security enhancements.
- FHE operations are computationally intensive and can significantly slow down training.
- The code relies on Avalanche Warp Messaging, which is blockchain-based and may introduce latency.

## Limitations
- **Latency**: AWM communication may introduce delays that impact training efficiency.
- **Cost**: Every inter-subnet communication incurs transaction fees.
- **Scalability**: FHE and blockchain-based messaging can be resource-intensive, limiting scalability for large datasets.

## License
This project is open source. You can modify and use it as per the project requirements.

## Contact
For questions or further discussions, please reach out to [Naman Bajpai](mailto:naman.bajpai@drexel.edu).

# POC0.0.1
# POC0.0.1
