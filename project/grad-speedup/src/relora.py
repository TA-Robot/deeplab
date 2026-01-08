from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _kaiming_init(tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        return
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def _zero_init(tensor: torch.Tensor) -> None:
    if tensor.numel() == 0:
        return
    nn.init.zeros_(tensor)


def _make_inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


def _init_projection(tensor: torch.Tensor, *, identity: bool) -> None:
    if tensor.numel() == 0:
        return
    if tensor.ndim == 2:
        if identity:
            nn.init.eye_(tensor)
        else:
            nn.init.orthogonal_(tensor)
        return
    if tensor.ndim == 3:
        for idx in range(tensor.shape[0]):
            if identity:
                nn.init.eye_(tensor[idx])
            else:
                nn.init.orthogonal_(tensor[idx])
        return
    raise ValueError("projection tensor must be 2D or 3D")


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    # Fast Walsh-Hadamard transform along the last dimension (power-of-two length).
    n = x.shape[-1]
    h = x.reshape(-1, n)
    size = 1
    while size < n:
        h = h.reshape(-1, n // (size * 2), size, 2)
        a = h[..., 0]
        b = h[..., 1]
        h = torch.cat((a + b, a - b), dim=-1)
        size *= 2
    return h.reshape(*x.shape)


class FastfoodProjector(nn.Module):
    def __init__(self, dim: int, *, scale: float = 1.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("FastfoodProjector dim must be > 0")
        self.dim = int(dim)
        self.pad_dim = _next_power_of_two(self.dim)
        b = torch.randint(0, 2, (self.pad_dim,), dtype=torch.float32) * 2.0 - 1.0
        g = torch.randn(self.pad_dim, dtype=torch.float32)
        perm = torch.randperm(self.pad_dim)
        s = torch.full((self.pad_dim,), float(scale) / math.sqrt(self.pad_dim), dtype=torch.float32)
        self.register_buffer("b", b)
        self.register_buffer("g", g)
        self.register_buffer("perm", perm)
        self.register_buffer("s", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)
        if self.pad_dim > self.dim:
            pad = x.new_zeros(x.shape[0], self.pad_dim - self.dim)
            x = torch.cat((x, pad), dim=-1)
        x = x * self.b
        x = _hadamard_transform(x)
        x = x * self.g
        x = _hadamard_transform(x)
        x = x.index_select(-1, self.perm)
        x = x * self.s
        x = x[..., : self.dim]
        return x.reshape(*orig_shape)

    def apply_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Apply projection along channel dimension.
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, c)
        x_proj = self.forward(x_flat)
        return x_proj.reshape(b, h, w, c).permute(0, 3, 1, 2)


class ReLoRALayer(nn.Module):
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError

    @torch.no_grad()
    def merge_into_base(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def relora_reset(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def shuffle(self) -> None:
        return


class ReLoRALinear(ReLoRALayer):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
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
        self.init_method = init_method

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
        if self.init_method == "qr":
            if _qr_init_linear(self.base.weight, self.lora_b.weight, self.lora_a.weight, self.rank):
                return
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
        init_method: str = "kaiming",
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
        self.init_method = init_method

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
        if self.init_method == "qr":
            if _qr_init_conv(self.base.weight, self.lora_b.weight, self.lora_a.weight, self.rank):
                return
        _kaiming_init(self.lora_a.weight)
        _zero_init(self.lora_b.weight)


class SuperLoRALinear(ReLoRALayer):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        group_count: int = 1,
        projection: str = "none",
        shuffle: bool = False,
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for SuperLoRA")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for SuperLoRA")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for SuperLoRA")
        if group_count <= 0:
            raise ValueError("group_count must be > 0 for SuperLoRA")
        if base.in_features % group_count != 0:
            raise ValueError("SuperLoRA group_count must divide in_features")
        if base.out_features % group_count != 0:
            raise ValueError("SuperLoRA group_count must divide out_features")
        if projection not in ("none", "fixed", "learned", "fastfood"):
            raise ValueError(f"unsupported SuperLoRA projection: {projection}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.group_count = int(group_count)
        self.in_group = base.in_features // self.group_count
        self.out_group = base.out_features // self.group_count
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.init_method = init_method
        self.projection = projection
        self.shuffle_enabled = bool(shuffle)
        self.fastfood: Optional[FastfoodProjector] = None
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(self.group_count, self.rank, self.in_group, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(
            torch.empty(self.group_count, self.out_group, self.rank, device=device, dtype=dtype)
        )
        if projection == "learned":
            self.lora_p = nn.Parameter(torch.empty(self.group_count, self.rank, self.rank, device=device, dtype=dtype))
        elif projection == "fixed":
            self.register_buffer(
                "lora_p",
                torch.empty(self.group_count, self.rank, self.rank, device=device, dtype=dtype),
            )
        else:
            self.lora_p = None
        if projection == "fastfood":
            self.fastfood = FastfoodProjector(base.in_features).to(device=device)

        self.register_buffer("in_perm", torch.arange(base.in_features, device=device, dtype=torch.long))
        self.register_buffer("in_inv_perm", torch.arange(base.in_features, device=device, dtype=torch.long))
        self.register_buffer("out_perm", torch.arange(base.out_features, device=device, dtype=torch.long))
        self.register_buffer("out_inv_perm", torch.arange(base.out_features, device=device, dtype=torch.long))

        if projection == "fixed":
            _init_projection(self.lora_p, identity=False)

        if self.shuffle_enabled:
            self.shuffle()

        self.relora_reset()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a
        yield self.lora_b
        if isinstance(self.lora_p, nn.Parameter):
            yield self.lora_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = x
        if self.shuffle_enabled:
            inp = inp.index_select(-1, self.in_perm)
        if self.fastfood is not None:
            inp = self.fastfood(inp)
        if self.dropout is not None:
            inp = self.dropout(inp)
        inp = inp.reshape(*inp.shape[:-1], self.group_count, self.in_group)
        hidden = torch.einsum("...gi,gri->...gr", inp, self.lora_a)
        if self.lora_p is not None:
            hidden = torch.einsum("...gr,grs->...gs", hidden, self.lora_p)
        delta = torch.einsum("...gr,gor->...go", hidden, self.lora_b)
        delta = delta.reshape(*delta.shape[:-2], self.base.out_features)
        if self.shuffle_enabled:
            delta = delta.index_select(-1, self.out_inv_perm)
        out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        delta = torch.zeros_like(self.base.weight)
        for group in range(self.group_count):
            out_slice = slice(group * self.out_group, (group + 1) * self.out_group)
            in_slice = slice(group * self.in_group, (group + 1) * self.in_group)
            b = self.lora_b[group]
            a = self.lora_a[group]
            if self.lora_p is not None:
                b = b @ self.lora_p[group]
            delta[out_slice, in_slice] = b @ a
        if self.shuffle_enabled:
            delta = delta.index_select(0, self.out_inv_perm).index_select(1, self.in_inv_perm)
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        if self.projection == "learned":
            _init_projection(self.lora_p, identity=True)
        if self.init_method == "qr":
            self._qr_init_groups()
            return
        for group in range(self.group_count):
            _kaiming_init(self.lora_a[group])
            _zero_init(self.lora_b[group])

    @torch.no_grad()
    def shuffle(self) -> None:
        if not self.shuffle_enabled:
            return
        device = self.base.weight.device
        in_perm = torch.randperm(self.base.in_features, device=device)
        out_perm = torch.randperm(self.base.out_features, device=device)
        self.in_perm.copy_(in_perm)
        self.out_perm.copy_(out_perm)
        self.in_inv_perm.copy_(_make_inverse_perm(in_perm))
        self.out_inv_perm.copy_(_make_inverse_perm(out_perm))

    def _qr_init_groups(self) -> None:
        weight = self.base.weight.detach()
        if self.shuffle_enabled:
            weight = weight.index_select(0, self.out_perm).index_select(1, self.in_perm)
        for group in range(self.group_count):
            out_slice = slice(group * self.out_group, (group + 1) * self.out_group)
            in_slice = slice(group * self.in_group, (group + 1) * self.in_group)
            if _qr_init_from_weight(weight[out_slice, in_slice], self.lora_b[group], self.lora_a[group], self.rank):
                continue
            _kaiming_init(self.lora_a[group])
            _zero_init(self.lora_b[group])


class SuperLoRAConv2d(ReLoRALayer):
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
        group_count: int = 1,
        projection: str = "none",
        shuffle: bool = False,
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for SuperLoRA")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for SuperLoRA")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for SuperLoRA")
        if group_count <= 0:
            raise ValueError("group_count must be > 0 for SuperLoRA")
        if base.groups != 1:
            raise NotImplementedError("SuperLoRAConv2d currently supports groups=1 only")
        if base.in_channels % group_count != 0:
            raise ValueError("SuperLoRA group_count must divide in_channels")
        if base.out_channels % group_count != 0:
            raise ValueError("SuperLoRA group_count must divide out_channels")
        if projection not in ("none", "fixed", "learned", "fastfood"):
            raise ValueError(f"unsupported SuperLoRA projection: {projection}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.group_count = int(group_count)
        self.in_group = base.in_channels // self.group_count
        self.out_group = base.out_channels // self.group_count
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.init_method = init_method
        self.projection = projection
        self.shuffle_enabled = bool(shuffle)
        self.fastfood: Optional[FastfoodProjector] = None
        device = base.weight.device
        dtype = base.weight.dtype

        self.lora_a = nn.Conv2d(
            base.in_channels,
            self.group_count * self.rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=self.group_count,
            bias=False,
        )
        self.lora_b = nn.Conv2d(
            self.group_count * self.rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.group_count,
            bias=False,
        )
        if projection not in ("none", "fastfood"):
            self.lora_p = nn.Conv2d(
                self.group_count * self.rank,
                self.group_count * self.rank,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.group_count,
                bias=False,
            )
            if projection == "fixed":
                self.lora_p.weight.requires_grad_(False)
        else:
            self.lora_p = None
        if projection == "fastfood":
            self.fastfood = FastfoodProjector(base.in_channels).to(device=device)
        self.lora_a.to(device=device, dtype=dtype)
        self.lora_b.to(device=device, dtype=dtype)
        if self.lora_p is not None:
            self.lora_p.to(device=device, dtype=dtype)

        self.register_buffer("in_perm", torch.arange(base.in_channels, device=device, dtype=torch.long))
        self.register_buffer("in_inv_perm", torch.arange(base.in_channels, device=device, dtype=torch.long))
        self.register_buffer("out_perm", torch.arange(base.out_channels, device=device, dtype=torch.long))
        self.register_buffer("out_inv_perm", torch.arange(base.out_channels, device=device, dtype=torch.long))

        if projection == "fixed":
            self._init_projection_weights(identity=False)

        if self.shuffle_enabled:
            self.shuffle()

        self.relora_reset()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a.weight
        yield self.lora_b.weight
        if self.lora_p is not None and self.lora_p.weight.requires_grad:
            yield self.lora_p.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = x
        if self.shuffle_enabled:
            inp = inp.index_select(1, self.in_perm)
        if self.fastfood is not None:
            inp = self.fastfood.apply_conv(inp)
        if self.dropout is not None:
            inp = self.dropout(inp)
        hidden = self.lora_a(inp)
        if self.lora_p is not None:
            hidden = self.lora_p(hidden)
        delta = self.lora_b(hidden)
        if self.shuffle_enabled:
            delta = delta.index_select(1, self.out_inv_perm)
        out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        delta = torch.zeros_like(self.base.weight)
        b_weight = self.lora_b.weight
        a_weight = self.lora_a.weight
        p_weight = self.lora_p.weight if self.lora_p is not None else None
        for group in range(self.group_count):
            out_slice = slice(group * self.out_group, (group + 1) * self.out_group)
            in_slice = slice(group * self.in_group, (group + 1) * self.in_group)
            rank_slice = slice(group * self.rank, (group + 1) * self.rank)
            b = b_weight[out_slice].squeeze(-1).squeeze(-1)
            a = a_weight[rank_slice].flatten(1)
            if p_weight is not None:
                p = p_weight[rank_slice].squeeze(-1).squeeze(-1)
                b = b @ p
            delta[out_slice, in_slice] = (b @ a).view(self.out_group, self.in_group, *self.base.kernel_size)
        if self.shuffle_enabled:
            delta = delta.index_select(0, self.out_inv_perm).index_select(1, self.in_inv_perm)
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        if self.projection == "learned":
            self._init_projection_weights(identity=True)
        if self.init_method == "qr":
            self._qr_init_groups()
            return
        _kaiming_init(self.lora_a.weight)
        _zero_init(self.lora_b.weight)

    @torch.no_grad()
    def shuffle(self) -> None:
        if not self.shuffle_enabled:
            return
        device = self.base.weight.device
        in_perm = torch.randperm(self.base.in_channels, device=device)
        out_perm = torch.randperm(self.base.out_channels, device=device)
        self.in_perm.copy_(in_perm)
        self.out_perm.copy_(out_perm)
        self.in_inv_perm.copy_(_make_inverse_perm(in_perm))
        self.out_inv_perm.copy_(_make_inverse_perm(out_perm))

    def _init_projection_weights(self, *, identity: bool) -> None:
        if self.lora_p is None:
            return
        weight = self.lora_p.weight.view(self.group_count, self.rank, self.rank)
        _init_projection(weight, identity=identity)

    def _qr_init_groups(self) -> None:
        weight = self.base.weight.detach()
        if self.shuffle_enabled:
            weight = weight.index_select(0, self.out_perm).index_select(1, self.in_perm)
        for group in range(self.group_count):
            out_slice = slice(group * self.out_group, (group + 1) * self.out_group)
            in_slice = slice(group * self.in_group, (group + 1) * self.in_group)
            rank_slice = slice(group * self.rank, (group + 1) * self.rank)
            block = weight[out_slice, in_slice].flatten(1)
            b = self.lora_b.weight[out_slice].squeeze(-1).squeeze(-1)
            a = self.lora_a.weight[rank_slice].flatten(1)
            if _qr_init_from_weight(block, b, a, self.rank):
                continue
            _kaiming_init(self.lora_a.weight[rank_slice])
            _zero_init(self.lora_b.weight[out_slice])

class TracSharedCore(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        inner_rank: int,
        init_method: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for TRAC shared core")
        if inner_rank <= 0:
            raise ValueError("inner_rank must be > 0 for TRAC shared core")
        if init_method not in ("identity", "kaiming"):
            raise ValueError(f"unsupported TRAC core init_method: {init_method}")
        self.rank = int(rank)
        self.inner_rank = int(inner_rank)
        self.init_method = init_method
        self.g3a = nn.Parameter(torch.empty(self.rank, self.inner_rank, device=device, dtype=dtype))
        self.g3b = nn.Parameter(torch.empty(self.inner_rank, self.rank, device=device, dtype=dtype))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.init_method == "identity":
            self.g3a.zero_()
            diag = min(self.rank, self.inner_rank)
            self.g3a[:diag, :diag].fill_(1.0)
        else:
            _kaiming_init(self.g3a)
        _zero_init(self.g3b)


def _init_trac_middle(weight: torch.Tensor) -> None:
    if weight.numel() == 0:
        return
    if weight.shape[0] == weight.shape[1]:
        nn.init.eye_(weight)
    else:
        _kaiming_init(weight)


class TracLinear(ReLoRALayer):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        inner_rank: int,
        alpha: float,
        shared_core: TracSharedCore,
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
        freeze_middle: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for TRAC")
        if inner_rank <= 0:
            raise ValueError("inner_rank must be > 0 for TRAC")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for TRAC")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for TRAC")
        if init_method not in ("kaiming", "qr", "tt-norm"):
            raise ValueError(f"unsupported TRAC init_method: {init_method}")

        self.base = base
        self.rank = int(rank)
        self.inner_rank = int(inner_rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.init_method = init_method
        self.freeze_middle = bool(freeze_middle)
        self.shared_core = shared_core

        self.lora_a1 = nn.Linear(base.in_features, self.inner_rank, bias=False)
        self.lora_a2 = nn.Linear(self.inner_rank, self.inner_rank, bias=False)
        self.lora_b2 = nn.Linear(self.inner_rank, self.inner_rank, bias=False)
        self.lora_b1 = nn.Linear(self.inner_rank, base.out_features, bias=False)
        self.lora_a1.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a2.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b2.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b1.to(device=base.weight.device, dtype=base.weight.dtype)

        self.a_in_scale = nn.Parameter(torch.zeros(self.inner_rank, device=base.weight.device, dtype=base.weight.dtype))
        self.a_out_scale = nn.Parameter(torch.zeros(self.rank, device=base.weight.device, dtype=base.weight.dtype))
        self.b_in_scale = nn.Parameter(torch.zeros(self.rank, device=base.weight.device, dtype=base.weight.dtype))
        self.b_out_scale = nn.Parameter(torch.zeros(self.inner_rank, device=base.weight.device, dtype=base.weight.dtype))

        self.relora_reset()

        if self.freeze_middle:
            self.lora_a2.weight.requires_grad_(False)
            self.lora_b2.weight.requires_grad_(False)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a1.weight
        yield self.lora_a2.weight
        yield self.lora_b2.weight
        yield self.lora_b1.weight
        yield self.a_in_scale
        yield self.a_out_scale
        yield self.b_in_scale
        yield self.b_out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = self.dropout(x) if self.dropout is not None else x
        a1 = self.lora_a1(inp)
        a2 = self.lora_a2(a1)
        a2 = a2 * (1.0 + self.a_in_scale)
        a3 = F.linear(a2, self.shared_core.g3a)
        a3 = a3 * (1.0 + self.a_out_scale)
        b1 = F.linear(a3 * (1.0 + self.b_in_scale), self.shared_core.g3b)
        b1 = b1 * (1.0 + self.b_out_scale)
        b2 = self.lora_b2(b1)
        delta = self.lora_b1(b2)
        out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        a_out = 1.0 + self.a_out_scale
        a_in = 1.0 + self.a_in_scale
        b_out = 1.0 + self.b_out_scale
        b_in = 1.0 + self.b_in_scale
        g3a = self.shared_core.g3a * a_out[:, None] * a_in[None, :]
        g3b = self.shared_core.g3b * b_out[:, None] * b_in[None, :]
        a_total = g3a @ self.lora_a2.weight @ self.lora_a1.weight
        b_total = self.lora_b1.weight @ self.lora_b2.weight @ g3b
        delta = b_total @ a_total
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        if self.init_method == "qr":
            if _qr_init_from_weight(self.base.weight, self.lora_b1.weight, self.lora_a1.weight, self.inner_rank):
                _init_trac_middle(self.lora_a2.weight)
                _init_trac_middle(self.lora_b2.weight)
                self.a_in_scale.zero_()
                self.a_out_scale.zero_()
                self.b_in_scale.zero_()
                self.b_out_scale.zero_()
                return
        _kaiming_init(self.lora_a1.weight)
        _kaiming_init(self.lora_b1.weight)
        _init_trac_middle(self.lora_a2.weight)
        _init_trac_middle(self.lora_b2.weight)
        if self.init_method == "tt-norm":
            scale = 1.0 / math.sqrt(max(1, self.inner_rank))
            self.lora_a1.weight.mul_(scale)
            self.lora_b1.weight.mul_(scale)
        self.a_in_scale.zero_()
        self.a_out_scale.zero_()
        self.b_in_scale.zero_()
        self.b_out_scale.zero_()


class TracConv2d(ReLoRALayer):
    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int,
        inner_rank: int,
        alpha: float,
        shared_core: TracSharedCore,
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
        freeze_middle: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0 for TRAC")
        if inner_rank <= 0:
            raise ValueError("inner_rank must be > 0 for TRAC")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0 for TRAC")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0 for TRAC")
        if base.groups != 1:
            raise NotImplementedError("TRAC Conv2d currently supports groups=1 only")
        if init_method not in ("kaiming", "qr", "tt-norm"):
            raise ValueError(f"unsupported TRAC init_method: {init_method}")

        self.base = base
        self.rank = int(rank)
        self.inner_rank = int(inner_rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None
        self.init_method = init_method
        self.freeze_middle = bool(freeze_middle)
        self.shared_core = shared_core

        self.lora_a1 = nn.Conv2d(
            base.in_channels,
            self.inner_rank,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=False,
        )
        self.lora_a2 = nn.Conv2d(
            self.inner_rank,
            self.inner_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lora_b2 = nn.Conv2d(
            self.inner_rank,
            self.inner_rank,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lora_b1 = nn.Conv2d(
            self.inner_rank,
            base.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.lora_a1.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_a2.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b2.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b1.to(device=base.weight.device, dtype=base.weight.dtype)

        self.a_in_scale = nn.Parameter(torch.zeros(self.inner_rank, device=base.weight.device, dtype=base.weight.dtype))
        self.a_out_scale = nn.Parameter(torch.zeros(self.rank, device=base.weight.device, dtype=base.weight.dtype))
        self.b_in_scale = nn.Parameter(torch.zeros(self.rank, device=base.weight.device, dtype=base.weight.dtype))
        self.b_out_scale = nn.Parameter(torch.zeros(self.inner_rank, device=base.weight.device, dtype=base.weight.dtype))

        self.relora_reset()

        if self.freeze_middle:
            self.lora_a2.weight.requires_grad_(False)
            self.lora_b2.weight.requires_grad_(False)

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(bool(train_bias))

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.lora_a1.weight
        yield self.lora_a2.weight
        yield self.lora_b2.weight
        yield self.lora_b1.weight
        yield self.a_in_scale
        yield self.a_out_scale
        yield self.b_in_scale
        yield self.b_out_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        inp = self.dropout(x) if self.dropout is not None else x
        a1 = self.lora_a1(inp)
        a2 = self.lora_a2(a1)
        a2 = a2 * (1.0 + self.a_in_scale)[None, :, None, None]
        g3a_weight = self.shared_core.g3a.view(self.rank, self.inner_rank, 1, 1)
        a3 = F.conv2d(a2, g3a_weight, bias=None, stride=1, padding=0)
        a3 = a3 * (1.0 + self.a_out_scale)[None, :, None, None]
        b1 = a3 * (1.0 + self.b_in_scale)[None, :, None, None]
        g3b_weight = self.shared_core.g3b.view(self.inner_rank, self.rank, 1, 1)
        b1 = F.conv2d(b1, g3b_weight, bias=None, stride=1, padding=0)
        b1 = b1 * (1.0 + self.b_out_scale)[None, :, None, None]
        b2 = self.lora_b2(b1)
        delta = self.lora_b1(b2)
        out = out + self.scaling * delta
        return out

    @torch.no_grad()
    def merge_into_base(self) -> None:
        a_out = 1.0 + self.a_out_scale
        a_in = 1.0 + self.a_in_scale
        b_out = 1.0 + self.b_out_scale
        b_in = 1.0 + self.b_in_scale
        g3a = self.shared_core.g3a * a_out[:, None] * a_in[None, :]
        g3b = self.shared_core.g3b * b_out[:, None] * b_in[None, :]
        a1_flat = self.lora_a1.weight.flatten(1)
        a2 = self.lora_a2.weight.squeeze(-1).squeeze(-1)
        b2 = self.lora_b2.weight.squeeze(-1).squeeze(-1)
        b1 = self.lora_b1.weight.squeeze(-1).squeeze(-1)
        a_total = g3a @ a2 @ a1_flat
        b_total = b1 @ b2 @ g3b
        delta = (b_total @ a_total).view_as(self.base.weight)
        self.base.weight.add_(delta, alpha=self.scaling)

    @torch.no_grad()
    def relora_reset(self) -> None:
        if self.init_method == "qr":
            if _qr_init_conv(self.base.weight, self.lora_b1.weight, self.lora_a1.weight, self.inner_rank):
                _init_trac_middle(self.lora_a2.weight.squeeze(-1).squeeze(-1))
                _init_trac_middle(self.lora_b2.weight.squeeze(-1).squeeze(-1))
                self.a_in_scale.zero_()
                self.a_out_scale.zero_()
                self.b_in_scale.zero_()
                self.b_out_scale.zero_()
                return
        _kaiming_init(self.lora_a1.weight)
        _kaiming_init(self.lora_b1.weight)
        _init_trac_middle(self.lora_a2.weight.squeeze(-1).squeeze(-1))
        _init_trac_middle(self.lora_b2.weight.squeeze(-1).squeeze(-1))
        if self.init_method == "tt-norm":
            scale = 1.0 / math.sqrt(max(1, self.inner_rank))
            self.lora_a1.weight.mul_(scale)
            self.lora_b1.weight.mul_(scale)
        self.a_in_scale.zero_()
        self.a_out_scale.zero_()
        self.b_in_scale.zero_()
        self.b_out_scale.zero_()
def _qr_init_linear(
    base_weight: torch.Tensor,
    lora_b_weight: torch.Tensor,
    lora_a_weight: torch.Tensor,
    rank: int,
) -> bool:
    return _qr_init_from_weight(base_weight, lora_b_weight, lora_a_weight, rank)


def _qr_init_conv(
    base_weight: torch.Tensor,
    lora_b_weight: torch.Tensor,
    lora_a_weight: torch.Tensor,
    rank: int,
) -> bool:
    if base_weight.ndim != 4:
        return False
    if rank <= 0:
        return False
    with torch.no_grad():
        w = base_weight.detach()
        dtype = w.dtype
        w_qr = w.float() if w.dtype in (torch.float16, torch.bfloat16) else w
        weight_2d = w_qr.flatten(1)
        try:
            q, r = torch.linalg.qr(weight_2d, mode="reduced")
        except RuntimeError:
            return False
        k = min(rank, q.shape[1], r.shape[0])
        if k <= 0:
            return False
        lora_b_weight.zero_()
        lora_a_weight.zero_()
        lora_b_weight[:, :k, 0, 0].copy_(q[:, :k].to(dtype))
        lora_a_weight[:k].copy_(
            r[:k, :].to(dtype).view(k, base_weight.shape[1], base_weight.shape[2], base_weight.shape[3])
        )
    return True


def _qr_init_from_weight(
    weight_2d: torch.Tensor,
    lora_b_weight: torch.Tensor,
    lora_a_weight: torch.Tensor,
    rank: int,
) -> bool:
    """Initialize LoRA weights with a QR decomposition of the base weight.

    Returns True if the QR init succeeded and weights were set.
    """
    if weight_2d.ndim != 2:
        return False
    if rank <= 0:
        return False
    with torch.no_grad():
        w = weight_2d.detach()
        dtype = w.dtype
        w_qr = w.float() if w.dtype in (torch.float16, torch.bfloat16) else w
        try:
            q, r = torch.linalg.qr(w_qr, mode="reduced")
        except RuntimeError:
            return False
        k = min(rank, q.shape[1], r.shape[0])
        if k <= 0:
            return False
        lora_b_weight.zero_()
        lora_a_weight.zero_()
        lora_b_weight[:, :k].copy_(q[:, :k].to(dtype))
        lora_a_weight[:k, :].copy_(r[:k, :].to(dtype))
    return True


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
    shuffle_interval: int = 0

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
        init_method: str = "kaiming",
        reset_optimizer_state: bool = True,
        prune_optimizer_state_fraction: float = 0.0,
    ) -> "ReLoRAController":
        if merge_interval <= 0:
            raise ValueError("merge_interval must be > 0 when enabling ReLoRA")
        if prune_optimizer_state_fraction < 0.0 or prune_optimizer_state_fraction >= 1.0:
            raise ValueError("prune_optimizer_state_fraction must be in [0, 1)")
        if init_method not in ("kaiming", "qr"):
            raise ValueError(f"unsupported ReLoRA init_method: {init_method}")

        layers: list[ReLoRALayer] = []

        def _patch(module: nn.Module, prefix: str) -> None:
            for child_name, child in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if _matches_relora_scope(scope, full_name, child):
                    if isinstance(child, nn.Linear):
                        wrapped: ReLoRALayer = ReLoRALinear(
                            child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
                        )
                    elif isinstance(child, nn.Conv2d):
                        wrapped = ReLoRAConv2d(
                            child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
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
            shuffle_interval=0,
        )

    @classmethod
    def apply_superlora(
        cls,
        model: nn.Module,
        *,
        merge_interval: int,
        rank: int,
        alpha: float,
        group_count: int = 1,
        projection: str = "none",
        shuffle: bool = False,
        shuffle_interval: int = 0,
        scope: str = "linear",
        dropout: float = 0.0,
        train_bias: bool = False,
        init_method: str = "kaiming",
        reset_optimizer_state: bool = True,
        prune_optimizer_state_fraction: float = 0.0,
    ) -> "ReLoRAController":
        if merge_interval <= 0:
            raise ValueError("merge_interval must be > 0 when enabling SuperLoRA")
        if prune_optimizer_state_fraction < 0.0 or prune_optimizer_state_fraction >= 1.0:
            raise ValueError("prune_optimizer_state_fraction must be in [0, 1)")
        if init_method not in ("kaiming", "qr"):
            raise ValueError(f"unsupported SuperLoRA init_method: {init_method}")
        if group_count <= 0:
            raise ValueError("group_count must be > 0 for SuperLoRA")
        if projection not in ("none", "fixed", "learned", "fastfood"):
            raise ValueError(f"unsupported SuperLoRA projection: {projection}")
        if shuffle and shuffle_interval <= 0:
            shuffle_interval = merge_interval

        layers: list[ReLoRALayer] = []

        def _patch(module: nn.Module, prefix: str) -> None:
            for child_name, child in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if _matches_relora_scope(scope, full_name, child):
                    if isinstance(child, nn.Linear):
                        wrapped: ReLoRALayer = SuperLoRALinear(
                            child,
                            rank=rank,
                            alpha=alpha,
                            group_count=group_count,
                            projection=projection,
                            shuffle=shuffle,
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
                        )
                    elif isinstance(child, nn.Conv2d):
                        wrapped = SuperLoRAConv2d(
                            child,
                            rank=rank,
                            alpha=alpha,
                            group_count=group_count,
                            projection=projection,
                            shuffle=shuffle,
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
                        )
                    else:
                        continue
                    setattr(module, child_name, wrapped)
                    layers.append(wrapped)
                else:
                    _patch(child, full_name)

        _patch(model, "")
        if not layers:
            raise ValueError(f"no layers matched SuperLoRA scope '{scope}'")

        return cls(
            merge_interval=int(merge_interval),
            reset_optimizer_state=bool(reset_optimizer_state),
            prune_optimizer_state_fraction=float(prune_optimizer_state_fraction),
            layers=layers,
            shuffle_interval=int(shuffle_interval) if shuffle else 0,
        )

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            for param in layer.lora_parameters():
                params.append(param)
        return params

    def maybe_merge(self, *, step: int, optimizer: torch.optim.Optimizer, device: torch.device) -> Optional[float]:
        do_merge = self.merge_interval > 0 and step > 0 and step % self.merge_interval == 0
        do_shuffle = self.shuffle_interval > 0 and step > 0 and step % self.shuffle_interval == 0
        if not do_merge and not do_shuffle:
            return None

        if do_merge:
            start_event = end_event = None
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

        with torch.no_grad():
            if do_merge:
                for layer in self.layers:
                    layer.merge_into_base()
                for layer in self.layers:
                    layer.relora_reset()
            if do_shuffle:
                for layer in self.layers:
                    layer.shuffle()

        if not do_merge:
            return None

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
@dataclass
class TracController:
    merge_interval: int
    reset_optimizer_state: bool
    prune_optimizer_state_fraction: float
    layers: list[ReLoRALayer]
    shared_core: TracSharedCore

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
        init_method: str = "kaiming",
        core_init: str = "identity",
        inner_rank: int = 0,
        freeze_middle: bool = True,
        reset_optimizer_state: bool = True,
        prune_optimizer_state_fraction: float = 0.0,
    ) -> "TracController":
        if merge_interval <= 0:
            raise ValueError("merge_interval must be > 0 when enabling TRAC")
        if prune_optimizer_state_fraction < 0.0 or prune_optimizer_state_fraction >= 1.0:
            raise ValueError("prune_optimizer_state_fraction must be in [0, 1)")
        if init_method not in ("kaiming", "qr", "tt-norm"):
            raise ValueError(f"unsupported TRAC init_method: {init_method}")
        if core_init not in ("identity", "kaiming"):
            raise ValueError(f"unsupported TRAC core_init: {core_init}")

        if inner_rank <= 0:
            inner_rank = max(1, rank // 2)

        layers: list[ReLoRALayer] = []
        shared_core: Optional[TracSharedCore] = None

        def _get_core(weight: torch.Tensor) -> TracSharedCore:
            nonlocal shared_core
            if shared_core is None:
                shared_core = TracSharedCore(
                    rank=rank,
                    inner_rank=inner_rank,
                    init_method=core_init,
                    device=weight.device,
                    dtype=weight.dtype,
                )
            return shared_core

        def _patch(module: nn.Module, prefix: str) -> None:
            for child_name, child in module.named_children():
                full_name = f"{prefix}.{child_name}" if prefix else child_name
                if _matches_relora_scope(scope, full_name, child):
                    if isinstance(child, nn.Linear):
                        wrapped: ReLoRALayer = TracLinear(
                            child,
                            rank=rank,
                            inner_rank=inner_rank,
                            alpha=alpha,
                            shared_core=_get_core(child.weight),
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
                            freeze_middle=freeze_middle,
                        )
                    elif isinstance(child, nn.Conv2d):
                        wrapped = TracConv2d(
                            child,
                            rank=rank,
                            inner_rank=inner_rank,
                            alpha=alpha,
                            shared_core=_get_core(child.weight),
                            dropout=dropout,
                            train_bias=train_bias,
                            init_method=init_method,
                            freeze_middle=freeze_middle,
                        )
                    else:
                        continue
                    setattr(module, child_name, wrapped)
                    layers.append(wrapped)
                else:
                    _patch(child, full_name)

        _patch(model, "")
        if not layers:
            raise ValueError(f"no layers matched TRAC scope '{scope}'")
        if shared_core is None:
            raise ValueError("TRAC shared core was not initialized")

        return cls(
            merge_interval=int(merge_interval),
            reset_optimizer_state=bool(reset_optimizer_state),
            prune_optimizer_state_fraction=float(prune_optimizer_state_fraction),
            layers=layers,
            shared_core=shared_core,
        )

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for layer in self.layers:
            for param in layer.lora_parameters():
                params.append(param)
        params.append(self.shared_core.g3a)
        params.append(self.shared_core.g3b)
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
            self.shared_core.reset_parameters()

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
