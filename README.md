# FedCKR: Class-wise Knowledge Routing for Bridging Knowledge Deficiency in Personalized Federated Learning

<p align="center">
    <img src="FedCKR.png">
</p>

> <h2>Additional implementation details and reproduction resources will be released progressively</h2>

## Installation

An example environment (works for me): ```Ubuntu 24.04```, ```Python 3.12```, ```CUDA 13.0```, ```PyTorch 2.8.0``` and ```Torchvision 0.23.0```.

Install the required dependencies:

```bash
pip install -r requirements.txt
Official PyTorch implementation of FedCKR for personalized federated learning under label-skewed non-IID data.
Datasets Preparation

We evaluate FedCKR on four benchmark datasets:

CIFAR-10
CIFAR-100
FashionMNIST
SVHN

The client data are partitioned using a Dirichlet distribution with:

beta = 0.1, 0.2, 0.3

The datasets can be downloaded automatically through torchvision or placed under the local data directory.

Example directory:

data/
├── cifar10/
├── cifar100/
├── fashionmnist/
└── svhn/

The current repository provides dataset loading and preprocessing utilities.

Training

FedCKR is evaluated under label-skewed non-IID federated learning settings.

The main experimental settings are:

Number of clients: 10
Client participation ratio: 0.5
Local epochs: 1
Batch size: 64
Optimizer: SGD
Learning rate: 0.01
Dirichlet beta: {0.1, 0.2, 0.3}

The complete FedCKR training implementation and reproduction scripts will be released progressively.

Evaluation

The main evaluation metrics include:

Personalized Accuracy
Weak-Class Accuracy
Worst-Client Accuracy
Global Accuracy

Detailed experimental results are reported in the paper.

Acknowledgment

We thank the authors of the related federated learning methods and open-source libraries used in this work.

LICENSE

License information will be updated with the complete code release.
