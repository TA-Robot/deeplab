from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .operator_basis import OperatorBasisConfig, OperatorBasisLayer


def _normalize_dims(dims: Sequence[int]) -> list[int]:
    if not dims:
        raise ValueError("hidden dims must be non-empty")
    return [int(d) for d in dims]


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Sequence[int] = (256, 256),
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = _normalize_dims(hidden_dims)
        self.hidden = nn.ModuleList()
        prev = input_dim
        for dim in dims:
            self.hidden.append(nn.Linear(prev, dim))
            prev = dim
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.out(x)


class MLPWithOBL(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Sequence[int] = (256, 256),
        num_classes: int = 10,
        dropout: float = 0.1,
        obl_config: OperatorBasisConfig | None = None,
    ) -> None:
        super().__init__()
        dims = _normalize_dims(hidden_dims)
        self.hidden = nn.ModuleList()
        self.obl_layers = nn.ModuleList()
        prev = input_dim
        for dim in dims:
            self.hidden.append(nn.Linear(prev, dim))
            if obl_config is None:
                config = OperatorBasisConfig(input_dim=dim)
            else:
                config = OperatorBasisConfig(**{**obl_config.__dict__, "input_dim": dim})
            self.obl_layers.append(OperatorBasisLayer(config))
            prev = dim
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer, obl in zip(self.hidden, self.obl_layers):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = obl(x)
        return self.out(x)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        fc_dims: Sequence[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        dims = _normalize_dims(fc_dims)
        prev = 64 * 7 * 7
        self.fcs = nn.ModuleList()
        for dim in dims:
            self.fcs.append(nn.Linear(prev, dim))
            prev = dim
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.fcs:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.out(x)


class CNNWithOBL(nn.Module):
    def __init__(
        self,
        fc_dims: Sequence[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.1,
        obl_config: OperatorBasisConfig | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        dims = _normalize_dims(fc_dims)
        prev = 64 * 7 * 7
        self.fcs = nn.ModuleList()
        self.obl_layers = nn.ModuleList()
        for dim in dims:
            self.fcs.append(nn.Linear(prev, dim))
            if obl_config is None:
                config = OperatorBasisConfig(input_dim=dim)
            else:
                config = OperatorBasisConfig(**{**obl_config.__dict__, "input_dim": dim})
            self.obl_layers.append(OperatorBasisLayer(config))
            prev = dim
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        for layer, obl in zip(self.fcs, self.obl_layers):
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = obl(x)
        return self.out(x)
