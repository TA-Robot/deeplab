from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class DataConfig:
    data_dir: str
    batch_size: int
    val_size: int
    num_workers: int
    seed: int
    download: bool
    pin_memory: bool = True
    drop_last: bool = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, test_transform


def get_cifar10_datasets(config: DataConfig) -> Tuple[Dataset, Dataset, Dataset]:
    train_transform, test_transform = build_cifar10_transforms()

    full_train = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=config.download,
        transform=train_transform,
    )
    test = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=config.download,
        transform=test_transform,
    )

    if config.val_size <= 0:
        return full_train, full_train, test

    val_size = min(config.val_size, len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.seed)
    train, val = random_split(full_train, [train_size, val_size], generator=generator)
    return train, val, test


def get_cifar10_loaders(
    config: DataConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train, val, test = get_cifar10_datasets(config)

    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    return train_loader, val_loader, test_loader
