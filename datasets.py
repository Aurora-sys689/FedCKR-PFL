from dataclasses import dataclass
from typing import Tuple

import numpy as np
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    in_channels: int
    image_size: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


DATASET_SPECS = {
    "cifar10": DatasetSpec(
        name="cifar10",
        num_classes=10,
        in_channels=3,
        image_size=32,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        num_classes=100,
        in_channels=3,
        image_size=32,
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
    "svhn": DatasetSpec(
        name="svhn",
        num_classes=10,
        in_channels=3,
        image_size=32,
        mean=(0.4377, 0.4438, 0.4728),
        std=(0.1980, 0.2010, 0.1970),
    ),
    "fashionmnist": DatasetSpec(
        name="fashionmnist",
        num_classes=10,
        in_channels=1,
        image_size=28,
        mean=(0.2860,),
        std=(0.3530,),
    ),
}


def dataset_choices():
    return sorted(DATASET_SPECS.keys())


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    key = str(dataset_name).lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from {dataset_choices()}.")
    return DATASET_SPECS[key]


def _classification_transforms(spec: DatasetSpec, train: bool):
    ops = []
    if train:
        if spec.image_size == 32:
            ops.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            ops.append(transforms.RandomRotation(10))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])
    return transforms.Compose(ops)


def build_datasets(dataset_name: str, data_root: str, download: bool = False):
    """
    Return train dataset with augmentation, train dataset with eval transform, test dataset, spec.
    The train/eval pair share the same indices but differ in transforms.
    """
    spec = get_dataset_spec(dataset_name)
    train_transform = _classification_transforms(spec, train=True)
    eval_transform = _classification_transforms(spec, train=False)

    if spec.name == "cifar10":
        train_set = datasets.CIFAR10(data_root, train=True, download=download, transform=train_transform)
        train_eval_set = datasets.CIFAR10(data_root, train=True, download=False, transform=eval_transform)
        test_set = datasets.CIFAR10(data_root, train=False, download=download, transform=eval_transform)
    elif spec.name == "cifar100":
        train_set = datasets.CIFAR100(data_root, train=True, download=download, transform=train_transform)
        train_eval_set = datasets.CIFAR100(data_root, train=True, download=False, transform=eval_transform)
        test_set = datasets.CIFAR100(data_root, train=False, download=download, transform=eval_transform)
    elif spec.name == "svhn":
        train_set = datasets.SVHN(data_root, split="train", download=download, transform=train_transform)
        train_eval_set = datasets.SVHN(data_root, split="train", download=False, transform=eval_transform)
        test_set = datasets.SVHN(data_root, split="test", download=download, transform=eval_transform)
    elif spec.name == "fashionmnist":
        train_set = datasets.FashionMNIST(data_root, train=True, download=download, transform=train_transform)
        train_eval_set = datasets.FashionMNIST(data_root, train=True, download=False, transform=eval_transform)
        test_set = datasets.FashionMNIST(data_root, train=False, download=download, transform=eval_transform)
    else:
        raise ValueError(f"Unsupported dataset '{spec.name}'.")

    return train_set, train_eval_set, test_set, spec


def get_labels(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets, dtype=np.int64)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels, dtype=np.int64)
    raise AttributeError("Dataset has neither targets nor labels.")


def default_split_file(split_root: str, dataset_name: str, beta: float, seed: int) -> str:
    beta_str = str(beta).replace(".", "p")
    return f"{split_root}/{dataset_name}_noniid_beta{beta_str}_seed{seed}.npz"
