# FedCKR: Class-wise Knowledge Routing for Bridging Knowledge Deficiency in Personalized Federated Learning

<p align="center">
    <img src="FedCKR.png">
</p>

This repository provides preliminary code and reproducibility resources for **FedCKR**, a class-wise knowledge routing framework for personalized federated learning under label-skewed non-IID data.

> <h2>Additional implementation details and reproduction resources will be released progressively</h2>

## Installation

An example environment (works for me): ```Ubuntu 24.04```, ```Python 3.12```, ```CUDA 13.0```, ```PyTorch 2.8.0``` and ```Torchvision 0.23.0```.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Datasets Preparation

We evaluate FedCKR on four benchmark datasets:

- CIFAR-10
- CIFAR-100
- FashionMNIST
- SVHN

The client data are partitioned using a Dirichlet distribution to construct label-skewed non-IID settings.

The datasets can be downloaded automatically through ```torchvision``` or prepared in a local data directory.

## Training

FedCKR is designed for personalized federated learning under label-skewed non-IID data.

The complete training implementation and additional reproduction scripts will be released progressively.

## Evaluation

The main evaluation focuses on personalized performance, weak-class performance, and worst-client performance.

Detailed experimental results are reported in the paper.

## Acknowledgment

We thank the authors of the related federated learning methods and open-source libraries used in this work.

## LICENSE

License information will be updated with the complete code release.
