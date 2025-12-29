from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data"
    batch_size: int = 128
    val_size: int = 5000
    num_workers: int = 0
    seed: int = 42
    dataset: str = "mnist"


DATASET_REGISTRY = {
    "mnist": {
        "cls": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "channels": 1,
        "size": 28,
    },
    "fashion-mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "channels": 1,
        "size": 28,
    },
    "cifar10": {
        "cls": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "channels": 3,
        "size": 32,
    },
}


def normalize_dataset_name(name: str) -> str:
    lowered = name.lower().replace("_", "-")
    if lowered == "fashionmnist":
        lowered = "fashion-mnist"
    return lowered


def _load_dataset(dataset_cls, root: str, train: bool, transform, name: str):
    try:
        return dataset_cls(root, train=train, download=False, transform=transform)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{name} dataset not found in {root}. Download is disabled. "
            "Place the dataset files manually or set --data-dir to an existing dataset."
        ) from exc


def get_dataset_loaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    dataset_key = normalize_dataset_name(config.dataset)
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"unsupported dataset: {config.dataset}")

    meta = DATASET_REGISTRY[dataset_key]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(meta["mean"], meta["std"]),
        ]
    )

    train_dataset = _load_dataset(meta["cls"], config.data_dir, True, transform, dataset_key)
    test_dataset = _load_dataset(meta["cls"], config.data_dir, False, transform, dataset_key)

    val_size = min(config.val_size, len(train_dataset) - 1)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    info = {
        "dataset": dataset_key,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_dataset),
        "mean": meta["mean"],
        "std": meta["std"],
        "channels": meta["channels"],
        "height": meta["size"],
        "width": meta["size"],
    }
    return train_loader, val_loader, test_loader, info
