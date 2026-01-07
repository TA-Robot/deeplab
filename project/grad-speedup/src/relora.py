from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn


def _kaiming_init(tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        return
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def _zero_init(tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        return
    nn.init.zeros_(tensor)


class ReLoRALayer(nn.Module):
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @torch.no_grad()
    def merge_into_base(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def relora_reset(self) -> None:
        raise NotImplementedError


class ReLoRALinear(ReLoRALayer):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for ReLoRA")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for ReLoRA")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for ReLoRA")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        self.relora_reset()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a.weight
        yield self.lora_b.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = self.dropout(x) if self.dropout is not None else x
        out = out + self.scaling * self.lora_b(self.lora_a(inp))
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        delta = self.lora_b.weight @ self.lora_a.weight
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        _kaiming_init(self.lora_a.weight)
        _zero_init(self.lora_b.weight)


class ReLoRAConv2d(ReLoRALayer):
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        train_bias: bool = False,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for ReLoRA")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for ReLoRA")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for ReLoRA")
        if base.groups != 1:
            raise NotImplementedError("ReLoRAConv2d currently supports groups=1 only")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

        self.lora_a = nn.Conv2d(
            base.in_channels,
            rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.lora_b = nn.Conv2d(
            rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        self.relora_reset()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a.weight
        yield self.lora_b.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = self.dropout(x) if self.dropout is not None else x
        out = out + self.scaling * self.lora_b(self.lora_a(inp))
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        b = self.lora_b.weight.squeeze(-1).squeeze(-1)  # [out, r]
        a = self.lora_a.weight.flatten(1)  # [r, in*k*k]
        delta = (b @ a).view_as(self.base.weight)
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        _kaiming_init(self.lora_a.weight)
        _zero_init(self.lora_b.weight)


def _matches_relora_scope(scope: str, full_name: str, module: nn.Module) -> bool:
    if scope == "linear":
        return isinstance(module, nn.Linear)
    if scope == "resnet-layer4":
        if full_name.startswith("layer4.") or full_name == "fc":
            return isinstance(module, (nn.Conv2d, nn.Linear))
        return False
    if scope == "resnet-layer3-4":
        if full_name.startswith(("layer3.", "layer4.")) or full_name == "fc":
            return isinstance(module, (nn.Conv2d, nn.Linear))
        return False
    if scope == "all":
        return isinstance(module, (nn.Conv2d, nn.Linear))
    raise ValueError(f"unsupported relora_scope: {scope}")


@dataclass
class ReLoRAController:
    merge_interval: int
    reset_optimizer_state: bool
    prune_optimizer_state_fraction: float
    layers: list[ReLoRALayer]

    @classmethod
    def apply(
        cls,
        model: nn.Module,
        *,
        merge_interval: int,
        rank: int,
        alpha: float,
        scope: str = "linear",
        dropout: float = 0.0,
        train_bias: bool = False,
        reset_optimizer_state: bool = True,
        prune_optimizer_state_fraction: float = 0.0,
    ) -> "ReLoRAController":
        if merge_interval <= 0:
            raise ValueError("merge_interval must be > 0 when enabling ReLoRA")
        if prune_optimizer_state_fraction < 0.0 or prune_optimizer_state_fraction >= 1.0:
            raise ValueError("prune_optimizer_state_fraction must be in [0, 1)")

        layers: list[ReLoRALayer] = []

        def _patch(module: nn.Module, prefix: str) -> None:
            for child_name, child in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if _matches_relora_scope(scope, full_name, child):
                    if isinstance(child, nn.Linear):
                        wrapped: ReLoRALayer = ReLoRALinear(
                            child, rank=rank, alpha=alpha, dropout=dropout, train_bias=train_bias
                        )
                    elif isinstance(child, nn.Conv2d):
                        wrapped = ReLoRAConv2d(
                            child, rank=rank, alpha=alpha, dropout=dropout, train_bias=train_bias
                        )
                    else:
                        continue
                    setattr(module, child_name, wrapped)
                    layers.append(wrapped)
                else:
                    _patch(child, full_name)

        _patch(model, "")
        if not layers:
            raise ValueError(f"no layers matched ReLoRA scope '{scope}'")

        return cls(
            merge_interval=int(merge_interval),
            reset_optimizer_state=bool(reset_optimizer_state),
            prune_optimizer_state_fraction=float(prune_optimizer_state_fraction),
            layers=layers,
        )

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            for param in layer.lora_parameters():
                params.append(param)
        return params

    def maybe_merge(self, *, step: int, optimizer: torch.optim.Optimizer, device: torch.device) -> Optional[float]:
        if self.merge_interval <= 0:
            return None
        if step <= 0 or step % self.merge_interval != 0:
            return None

        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.perf_counter()

        with torch.no_grad():
            for layer in self.layers:
                layer.merge_into_base()
            for layer in self.layers:
                layer.relora_reset()

        if device.type == "cuda":
            assert end_event is not None
            end_event.record()
            torch.cuda.synchronize()
            elapsed_s = float(start_event.elapsed_time(end_event) / 1000.0) if start_event is not None else 0.0
        else:
            elapsed_s = float(time.perf_counter() - start_time)

        if self.reset_optimizer_state:
            for param in self.lora_parameters():
                optimizer.state.pop(param, None)
        elif self.prune_optimizer_state_fraction > 0.0:
            self._prune_optimizer_state(optimizer)

        return elapsed_s

    def _prune_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        fraction = self.prune_optimizer_state_fraction
        if fraction <= 0.0:
            return
        keep_q = 1.0 - fraction
        for param in self.lora_parameters():
            state = optimizer.state.get(param)
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if not isinstance(value, torch.Tensor) or value.shape != param.shape:
                    continue
                flat = value.detach().abs().flatten()
                if flat.numel() == 0:
                    continue
                threshold = float(torch.quantile(flat, keep_q).item())
                mask = value.detach().abs() >= threshold
                value.mul_(mask)
