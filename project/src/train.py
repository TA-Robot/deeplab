from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrainMetrics:
    loss: float
    accuracy: float
    step_time_ms: float
    throughput: float
    steps: int
    samples: int


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    samples: int


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module, include_buffers: bool = False) -> int:
    total = sum(p.numel() for p in model.parameters())
    if include_buffers:
        total += sum(b.numel() for b in model.buffers())
    return total


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_l1_weight: float = 0.0,
    beta_l1_fn: Callable[[nn.Module], torch.Tensor] | None = None,
    warmup_steps: int = 5,
) -> TrainMetrics:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    step_times = []
    measured_samples = 0

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        if beta_l1_weight > 0 and beta_l1_fn is not None:
            loss = loss + beta_l1_weight * beta_l1_fn(model)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        if batch_idx >= warmup_steps:
            step_times.append(end - start)
            measured_samples += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    if step_times:
        avg_step = sum(step_times) / len(step_times)
        throughput = measured_samples / max(sum(step_times), 1e-12)
        step_time_ms = avg_step * 1000.0
    else:
        step_time_ms = 0.0
        throughput = 0.0
    return TrainMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        step_time_ms=step_time_ms,
        throughput=throughput,
        steps=len(loader),
        samples=total,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return EvalMetrics(loss=avg_loss, accuracy=accuracy, samples=total)
