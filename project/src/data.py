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


def get_mnist_loaders(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(config.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(config.data_dir, train=False, download=True, transform=transform)

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
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_dataset),
        "mean": 0.1307,
        "std": 0.3081,
    }
    return train_loader, val_loader, test_loader, info
