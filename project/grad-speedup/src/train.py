from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .modules import SUPPORTED_CLIP_MODES, SUPPORTED_DIRECTIONS, SUPPORTED_SPARSITY, SUPPORTED_STEP_RULES

MUON_NS_COEFFS = (3.4445, -4.7750, 2.0315)
MUON_SCALE_MODES = ("none", "baseline", "update-norm", "adjusted-lr")

@dataclass
class TrainMetrics:
    loss: float
    accuracy: float
    step_time_ms: float
    throughput: float
    steps: int
    samples: int
    step_time_count: int
    step_time_total_s: float
    step_size_mean: Optional[float] = None
    step_size_p50: Optional[float] = None
    step_size_p90: Optional[float] = None
    grad_norm_mean: Optional[float] = None
    grad_norm_p50: Optional[float] = None
    grad_norm_p90: Optional[float] = None
    curvature_mean: Optional[float] = None
    curvature_p50: Optional[float] = None
    curvature_p90: Optional[float] = None
    direction_scale_mean: Optional[float] = None
    direction_scale_p50: Optional[float] = None
    direction_scale_p90: Optional[float] = None
    clip_coef_mean: Optional[float] = None
    clip_coef_p50: Optional[float] = None
    clip_coef_p90: Optional[float] = None
    sophia_hessian_mean: Optional[float] = None
    sophia_hessian_p50: Optional[float] = None
    sophia_hessian_p90: Optional[float] = None
    sophia_clip_frac_mean: Optional[float] = None
    sophia_clip_frac_p50: Optional[float] = None
    sophia_clip_frac_p90: Optional[float] = None
    muon_ortho_iters_mean: Optional[float] = None
    muon_ortho_iters_p50: Optional[float] = None
    muon_ortho_iters_p90: Optional[float] = None
    line_search_attempts: int = 0
    line_search_accepted: int = 0
    line_search_rejected: int = 0
    line_search_iters_mean: Optional[float] = None
    line_search_iters_p50: Optional[float] = None
    line_search_iters_p90: Optional[float] = None
    anderson_applied: int = 0
    anderson_failed: int = 0
    precond_update_count: int = 0
    precond_apply_count: int = 0
    precond_update_time_s: Optional[float] = None
    precond_apply_time_s: Optional[float] = None
    precond_layer_stats: Optional[list[Dict[str, Any]]] = None
    data_wait_time_s: Optional[float] = None
    max_memory_bytes: Optional[int] = None
    sparsity_fraction: Optional[float] = None
    dense_flops: Optional[float] = None
    effective_flops: Optional[float] = None
    sparsity_updates: int = 0
    sparsity_update_interval: Optional[int] = None
    sparsity_update_rate: Optional[float] = None


@dataclass
class EvalMetrics:
    loss: float
    accuracy: float
    samples: int


@dataclass
class StepLog:
    epoch: int
    step_in_epoch: int
    global_step: int
    loss: float
    accuracy: float
    lr: float
    step_size: Optional[float]
    grad_norm: Optional[float]
    curvature: Optional[float]
    step_time_ms: Optional[float] = None
    line_search_iters: Optional[int] = None
    line_search_accepted: Optional[bool] = None


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module, include_buffers: bool = False) -> int:
    total = sum(p.numel() for p in model.parameters())
    if include_buffers:
        total += sum(b.numel() for b in model.buffers())
    return total


def _sum_grad_sq(params: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for param in params:
        if param.grad is None:
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def _sum_tensor_sq(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in tensors:
        value = tensor.detach().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total


def compute_grad_norm(params: Iterable[torch.Tensor]) -> float:
    total = _sum_grad_sq(params)
    return float(torch.sqrt(total).item())


def _flatten_params(params: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in params])


def _assign_params(params: Iterable[torch.Tensor], flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.data.copy_(flat[offset : offset + numel].view_as(param))
        offset += numel


@torch.no_grad()
def _apply_manual_update(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        lr = group["lr"]
        weight_decay = group.get("weight_decay", 0.0)
        for param in group["params"]:
            if param.grad is None:
                continue
            if weight_decay:
                param.data.mul_(1.0 - lr * weight_decay)
            param.data.add_(param.grad, alpha=-lr)


def _zero_sgd_momentum(optimizer: torch.optim.Optimizer) -> None:
    if not isinstance(optimizer, torch.optim.SGD):
        return
    for group in optimizer.param_groups:
        group["momentum"] = 0.0
        group["dampening"] = 0.0
        group["nesterov"] = False


def _ggnc_apply(
    params: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    rho: float,
    alpha: float,
    eps: float,
    mode: str,
) -> float:
    if mode not in ("ggnc-global", "ggnc-layerwise"):
        raise ValueError(f"unsupported GGNC mode: {mode}")

    entries: list[tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]] = []
    for param in params:
        grad = param.grad
        if grad is None:
            entries.append((param, None, None))
            continue
        state = optimizer.state[param]
        estimate = state.get("ggnc_d")
        if estimate is None:
            estimate = torch.zeros_like(grad)
        estimate.mul_(1.0 - alpha).add_(grad, alpha=alpha)
        state["ggnc_d"] = estimate
        if mode == "ggnc-layerwise":
            norm = float(torch.sqrt(estimate.detach().pow(2).sum()).item())
            entries.append((param, estimate, norm))
        else:
            entries.append((param, estimate, None))

    if mode == "ggnc-global":
        total_sq = None
        for _, estimate, _ in entries:
            if estimate is None:
                continue
            value = estimate.detach().pow(2).sum()
            total_sq = value if total_sq is None else total_sq + value
        total_norm = float(torch.sqrt(total_sq).item()) if total_sq is not None else 0.0
        if not math.isfinite(total_norm) or total_norm <= 0.0:
            tau = 1.0
        elif total_norm > rho:
            tau = rho / max(total_norm, eps)
        else:
            tau = 1.0
        for param, estimate, _ in entries:
            if estimate is None or param.grad is None:
                continue
            param.grad.copy_(estimate)
            if tau < 1.0:
                param.grad.mul_(tau)
        return float(tau)

    dual_norm = 0.0
    for _, _, norm in entries:
        if norm is None or not math.isfinite(norm):
            continue
        dual_norm += norm
    if not math.isfinite(dual_norm) or dual_norm <= 0.0:
        tau = 1.0
        eta = 0.0
    else:
        eta = min(rho, dual_norm)
        tau = 1.0 if dual_norm <= rho else rho / max(dual_norm, eps)
    for param, estimate, norm in entries:
        if estimate is None or param.grad is None:
            continue
        if norm is None or norm <= 0.0 or not math.isfinite(norm):
            param.grad.zero_()
            continue
        param.grad.copy_(estimate)
        param.grad.mul_(eta / max(norm, eps))
    return float(tau)


def _normalize_clip_mode(mode: str) -> str:
    if mode in ("global", "ggnc"):
        return "ggnc-global"
    if mode == "layerwise":
        return "ggnc-layerwise"
    return mode


def _linbreg_shrink(value: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0:
        return value
    return value.sign() * torch.clamp(value.abs() - threshold, min=0.0)


def _should_sparsify(param: torch.Tensor) -> bool:
    return param.requires_grad and param.ndim > 1


def _sparsity_counts(params: Iterable[torch.Tensor]) -> tuple[int, int]:
    nonzero = 0
    total = 0
    for param in params:
        total += param.numel()
        if param.numel() == 0:
            continue
        nonzero += int(torch.count_nonzero(param).item())
    return nonzero, total


def _estimate_dense_flops_per_sample(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
) -> int:
    flops: list[int] = []

    def _hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            out = output
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            if isinstance(module.kernel_size, tuple):
                k_h, k_w = module.kernel_size
            else:
                k_h = k_w = int(module.kernel_size)
            out_channels = int(out.shape[1])
            out_h = int(out.shape[2])
            out_w = int(out.shape[3])
            kernel_ops = int(k_h * k_w * (module.in_channels // module.groups))
            flops.append(int(out_channels * out_h * out_w * kernel_ops * 2))
        elif isinstance(module, nn.Linear):
            flops.append(int(module.in_features * module.out_features * 2))

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(input_shape, device=device)
        model(dummy)
    for handle in handles:
        handle.remove()
    if was_training:
        model.train()
    return int(sum(flops))


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))


def _anderson_coefficients(residuals: torch.Tensor, reg: float, eps: float) -> Optional[torch.Tensor]:
    m_dim = residuals.shape[1]
    if m_dim == 1:
        return torch.ones(1, device=residuals.device, dtype=residuals.dtype)
    gram = residuals.t() @ residuals
    if reg > 0.0:
        gram = gram + reg * torch.eye(m_dim, device=gram.device, dtype=gram.dtype)
    ones = torch.ones(m_dim, device=gram.device, dtype=gram.dtype)
    try:
        sol = torch.linalg.solve(gram, ones)
    except RuntimeError:
        return None
    denom = ones.dot(sol)
    if not torch.isfinite(denom) or float(abs(denom).item()) <= eps:
        return None
    alpha = sol / denom
    if not torch.isfinite(alpha).all():
        return None
    return alpha


@torch.no_grad()
def _muon_orthogonalize(update: torch.Tensor, ns_iters: int, eps: float) -> tuple[torch.Tensor, int]:
    if update.ndim < 2:
        return update, 0
    original_dtype = update.dtype
    mat = update.detach().float()
    original_shape = mat.shape
    if mat.ndim > 2:
        mat = mat.reshape(mat.shape[0], -1)
    frob = torch.linalg.norm(mat)
    if not torch.isfinite(frob) or frob <= eps:
        return torch.zeros_like(update), 0
    x = mat / frob
    a, b, c = MUON_NS_COEFFS
    for _ in range(ns_iters):
        gram = x @ x.t()
        gram_x = gram @ x
        x = a * x + b * gram_x + c * (gram @ gram_x)
    x = x.reshape(original_shape).to(original_dtype)
    return x, ns_iters


def _muon_scale_factor(
    update: torch.Tensor,
    param: torch.Tensor,
    mode: str,
    rms_scale: float,
    hidden_size: int,
    eps: float,
) -> float:
    if mode == "none":
        return 1.0
    if mode == "baseline":
        if hidden_size <= 0:
            raise ValueError("muon_hidden_size must be > 0 when muon_scale_mode='baseline'")
        return rms_scale * math.sqrt(hidden_size)
    if mode == "update-norm":
        rms = float(update.detach().float().pow(2).mean().sqrt().item())
        if not math.isfinite(rms) or rms <= eps:
            return 0.0
        return rms_scale / rms
    if mode == "adjusted-lr":
        if param.ndim == 0:
            size = 1
        else:
            rows = int(param.shape[0])
            cols = max(int(param.numel() / max(rows, 1)), 1)
            size = max(rows, cols)
        return rms_scale * math.sqrt(size)
    raise ValueError(f"unsupported muon_scale_mode: {mode}")


class StepTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.start_event: Optional[torch.cuda.Event] = None
        self.end_event: Optional[torch.cuda.Event] = None
        self.start_time: Optional[float] = None
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        if self.use_cuda:
            assert self.start_event is not None
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        if self.use_cuda:
            assert self.start_event is not None
            assert self.end_event is not None
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            return float(elapsed_ms / 1000.0)
        end_time = time.perf_counter()
        if self.start_time is None:
            return 0.0
        return float(end_time - self.start_time)


def _maybe_sync(device: torch.device, sync: bool) -> None:
    if sync and device.type == "cuda":
        torch.cuda.synchronize()


def _matrix_inverse_power(matrix: torch.Tensor, power: float, damping: float) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    safe_damping = max(damping, 1e-12)
    if safe_damping > 0.0:
        eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        matrix = matrix + safe_damping * eye
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = eigvals.clamp_min(safe_damping)
    eigvals_pow = eigvals.pow(power)
    return (eigvecs * eigvals_pow) @ eigvecs.t()


def _soap_eigenvectors(matrix: torch.Tensor, estimate: Optional[torch.Tensor]) -> torch.Tensor:
    """Update eigenvectors via Algorithm 4 from arXiv:2409.11321.

    Uses one power iteration step followed by QR re-orthonormalization. If an
    eigenvector estimate is missing or incompatible, initializes via a full
    eigen-decomposition.
    """
    if matrix.numel() == 0:
        if estimate is None:
            return matrix
        return estimate
    if estimate is None or estimate.shape != matrix.shape:
        _, eigvecs = torch.linalg.eigh(matrix)
        return eigvecs
    product = matrix @ estimate
    q, _ = torch.linalg.qr(product, mode="reduced")
    return q


def _apply_preconditioners(tensor: torch.Tensor, matrices: list[torch.Tensor]) -> torch.Tensor:
    out = tensor
    order = out.ndim
    for dim, mat in enumerate(matrices):
        perm = [dim] + [idx for idx in range(order) if idx != dim]
        inv_perm = [perm.index(idx) for idx in range(order)]
        out = out.permute(perm)
        shape = out.shape
        out = out.reshape(shape[0], -1)
        out = mat @ out
        out = out.reshape(shape)
        out = out.permute(inv_perm)
    return out


def _update_shampoo_stats(
    grad: torch.Tensor,
    stats: list[torch.Tensor],
    beta: float,
) -> None:
    order = grad.ndim
    use_ema = beta < 1.0
    for dim, stat in enumerate(stats):
        perm = [dim] + [idx for idx in range(order) if idx != dim]
        grad_mat = grad.permute(perm).reshape(grad.shape[dim], -1)
        outer = grad_mat @ grad_mat.t()
        if use_ema:
            stat.mul_(beta).add_(outer, alpha=1.0 - beta)
        else:
            stat.add_(outer)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int,
    log_fn: Optional[Callable[[StepLog], None]] = None,
    on_step_end: Optional[Callable[[int, int, int], bool]] = None,
    warmup_steps: int = 50,
    measure_steps: int = 200,
    grad_norm_every: int = 0,
    step_rule: str = "none",
    step_eoss_beta: Optional[float] = None,
    step_eoss_ema: float = 0.9,
    step_eoss_interval: int = 10,
    step_eoss_eps: float = 1e-8,
    step_eoss_clip_min: float = 1e-5,
    step_eoss_clip_max: float = 1.0,
    step_l0: float = 1.0,
    step_l1: float = 0.0,
    step_fstar: float = 0.0,
    step_sps_beta: float = 0.9,
    step_sps_c: float = 1.0,
    step_sps_max: Optional[float] = None,
    step_backtrack_c: float = 0.1,
    step_backtrack_max: int = 10,
    step_backtrack_rho: float = 0.5,
    step_silver_rho: float = 2.414213562373095,
    step_sagd_delta: float = 1e-2,
    step_eps: float = 1e-12,
    direction: str = "none",
    direction_beta: float = 0.9,
    direction_eps: float = 1e-8,
    direction_beta1: float = 0.9,
    direction_damping: float = 1e-5,
    direction_update_every: int = 1,
    sophia_beta1: float = 0.96,
    sophia_beta2: float = 0.99,
    sophia_gamma: float = 0.01,
    sophia_eps: float = 1e-12,
    sophia_hessian_every: int = 10,
    sophia_hutchinson_samples: int = 1,
    muon_beta: float = 0.95,
    muon_eps: float = 1e-8,
    muon_ns_iters: int = 5,
    muon_scale_mode: str = "adjusted-lr",
    muon_rms_scale: float = 0.2,
    muon_hidden_size: int = 0,
    clip_mode: str = "none",
    clip_rho: float = 1.0,
    clip_alpha: float = 1.0,
    sparsity: str = "none",
    sparsity_lambda: float = 0.0,
    sparsity_update_interval: int = 1,
    anderson_memory: int = 0,
    anderson_interval: int = 0,
    anderson_damping: float = 0.5,
    anderson_lambda: float = 1e-4,
    anderson_state: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    diagnostics: bool = False,
) -> tuple[TrainMetrics, int]:
    model.train()
    if step_rule not in SUPPORTED_STEP_RULES:
        raise ValueError(f"unsupported step_rule: {step_rule}")
    if step_rule == "l0l1" and (step_l0 < 0.0 or step_l1 < 0.0):
        raise ValueError("step_l0 and step_l1 must be non-negative")
    if step_rule == "eoss":
        if not 0.0 <= step_eoss_ema < 1.0:
            raise ValueError("step_eoss_ema must be in [0, 1) for eoss")
        if step_eoss_interval <= 0:
            raise ValueError("step_eoss_interval must be > 0 for eoss")
        if step_eoss_eps <= 0.0:
            raise ValueError("step_eoss_eps must be > 0 for eoss")
        if step_eoss_clip_min <= 0.0 or step_eoss_clip_max <= 0.0:
            raise ValueError("step_eoss_clip_min/max must be > 0 for eoss")
        if step_eoss_clip_min > step_eoss_clip_max:
            raise ValueError("step_eoss_clip_min must be <= step_eoss_clip_max for eoss")
    if step_rule == "sps-momentum" and not 0.0 <= step_sps_beta < 1.0:
        raise ValueError("step_sps_beta must be in [0, 1) for sps-momentum")
    if step_rule == "sps-momentum" and step_sps_c <= 0.0:
        raise ValueError("step_sps_c must be > 0 for sps-momentum")
    if step_rule == "sps-momentum" and step_sps_max is not None and step_sps_max <= 0.0:
        raise ValueError("step_sps_max must be > 0 for sps-momentum when provided")
    if step_rule == "adaptive-backtracking":
        if not 0.0 < step_backtrack_c < 1.0:
            raise ValueError("step_backtrack_c must be in (0, 1) for adaptive-backtracking")
        if step_backtrack_max <= 0:
            raise ValueError("step_backtrack_max must be > 0 for adaptive-backtracking")
        if not 0.0 < step_backtrack_rho < 1.0:
            raise ValueError("step_backtrack_rho must be in (0, 1) for adaptive-backtracking")
    if step_rule == "silver" and step_silver_rho <= 1.0:
        raise ValueError("step_silver_rho must be > 1 for silver")
    if step_rule == "sagd" and not 0.0 < step_sagd_delta < 0.5:
        raise ValueError("step_sagd_delta must be in (0, 0.5) for sagd")
    if step_eps <= 0.0:
        raise ValueError("step_eps must be > 0")
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"unsupported direction: {direction}")
    if clip_mode not in SUPPORTED_CLIP_MODES:
        raise ValueError(f"unsupported clip_mode: {clip_mode}")
    clip_mode = _normalize_clip_mode(clip_mode)
    if clip_mode != "none":
        if clip_rho <= 0.0:
            raise ValueError("clip_rho must be > 0 when GGNC is enabled")
        if not 0.0 < clip_alpha <= 1.0:
            raise ValueError("clip_alpha must be in (0, 1] for GGNC")
    if sparsity not in SUPPORTED_SPARSITY:
        raise ValueError(f"unsupported sparsity: {sparsity}")
    if sparsity != "none":
        if sparsity_lambda < 0.0:
            raise ValueError("sparsity_lambda must be non-negative")
        if sparsity_update_interval <= 0:
            raise ValueError("sparsity_update_interval must be > 0 when sparsity is enabled")
    if anderson_memory < 0 or anderson_interval < 0:
        raise ValueError("anderson_memory and anderson_interval must be >= 0")
    if direction in ("shampoo", "soap"):
        if direction_damping < 0.0:
            raise ValueError("direction_damping must be >= 0")
    if direction == "shampoo":
        if not 0.0 <= direction_beta <= 1.0:
            raise ValueError("direction_beta must be in [0, 1] for shampoo")
    if direction == "soap":
        if not 0.0 <= direction_beta1 < 1.0:
            raise ValueError("direction_beta1 must be in [0, 1)")
        if not 0.0 <= direction_beta < 1.0:
            raise ValueError("direction_beta must be in [0, 1)")
        if direction_eps <= 0.0:
            raise ValueError("direction_eps must be > 0")
    if direction == "sophia":
        if not 0.0 <= sophia_beta1 < 1.0:
            raise ValueError("sophia_beta1 must be in [0, 1)")
        if not 0.0 <= sophia_beta2 < 1.0:
            raise ValueError("sophia_beta2 must be in [0, 1)")
        if sophia_gamma <= 0.0:
            raise ValueError("sophia_gamma must be > 0")
        if sophia_eps <= 0.0:
            raise ValueError("sophia_eps must be > 0")
        if sophia_hessian_every <= 0:
            raise ValueError("sophia_hessian_every must be > 0")
        if sophia_hutchinson_samples <= 0:
            raise ValueError("sophia_hutchinson_samples must be > 0")
    if direction == "muon":
        if not 0.0 <= muon_beta < 1.0:
            raise ValueError("muon_beta must be in [0, 1)")
        if muon_eps <= 0.0:
            raise ValueError("muon_eps must be > 0")
        if muon_ns_iters <= 0:
            raise ValueError("muon_ns_iters must be > 0")
        if muon_scale_mode not in MUON_SCALE_MODES:
            raise ValueError(f"muon_scale_mode must be one of {MUON_SCALE_MODES}")
        if muon_rms_scale <= 0.0 and muon_scale_mode != "none":
            raise ValueError("muon_rms_scale must be > 0 when muon scaling is enabled")
        if muon_scale_mode == "baseline" and muon_hidden_size <= 0:
            raise ValueError("muon_hidden_size must be > 0 when muon_scale_mode='baseline'")
    if step_rule in ("adaptive-backtracking", "sps-momentum", "sagd") and direction != "none":
        raise ValueError(f"{step_rule} requires direction='none' for paper-accurate updates")
    if step_rule in ("adaptive-backtracking", "sps-momentum", "sagd") and sparsity != "none":
        raise ValueError(f"{step_rule} requires sparsity='none' for paper-accurate updates")
    if step_rule in ("l0l1", "sps", "silver") and not isinstance(optimizer, torch.optim.SGD):
        raise ValueError(f"{step_rule} requires an SGD optimizer for paper-accurate updates")
    if step_rule == "sps-momentum" and not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("sps-momentum requires an SGD optimizer")
    if step_rule == "adaptive-backtracking" and not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("adaptive-backtracking requires an SGD optimizer")
    if step_rule == "sagd" and not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("sagd requires an SGD optimizer")
    if anderson_state is None:
        anderson_state = []

    params: list[torch.Tensor] = []
    param_names: Dict[int, str] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        param_names[id(param)] = name
    sparsity_enabled = sparsity != "none"
    sparse_params: list[torch.Tensor] = []
    if sparsity_enabled:
        sparse_params = [param for param in params if _should_sparsify(param)]
    base_lrs: list[float] = []
    if step_rule != "none":
        for group in optimizer.param_groups:
            base_lr = group.get("base_lr")
            if base_lr is None:
                base_lr = group["lr"]
                group["base_lr"] = base_lr
            base_lrs.append(base_lr)
    if step_rule == "eoss" and step_eoss_beta is None:
        if base_lrs:
            step_eoss_beta = float(base_lrs[0])
        elif optimizer.param_groups:
            step_eoss_beta = float(optimizer.param_groups[0]["lr"])
    step_state: Dict[str, Any] = {}
    if optimizer.param_groups:
        step_state = optimizer.param_groups[0].setdefault("step_control_state", {})

    total_loss = 0.0
    correct = 0
    total = 0

    step_times: list[float] = []
    measured_samples = 0
    measured_steps = 0
    step_sizes: list[float] = []
    grad_norms: list[float] = []
    curvatures: list[float] = []
    direction_scales: list[float] = []
    clip_coefs: list[float] = []
    sophia_hessian_vals: list[float] = []
    sophia_clip_fracs: list[float] = []
    muon_ortho_iters: list[float] = []
    anderson_applied = 0
    anderson_failed = 0
    precond_update_count = 0
    precond_apply_count = 0
    precond_update_time_s = 0.0
    precond_apply_time_s = 0.0
    precond_layer_stats: Dict[str, Dict[str, Any]] = {}
    sparsity_fracs: list[float] = []
    dense_flops_vals: list[float] = []
    effective_flops_vals: list[float] = []
    sparsity_update_count = 0
    line_search_iters_list: list[int] = []
    line_search_attempts = 0
    line_search_accepted = 0
    line_search_rejected = 0
    dense_flops_per_sample = getattr(model, "_gs_dense_flops_per_sample", None) if sparsity_enabled else None
    last_batch_size: Optional[int] = None
    last_input_shape: Optional[tuple[int, ...]] = None

    timer = StepTimer(device)
    precond_sync = diagnostics
    if diagnostics and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data_wait_times: list[float] = []
    prev_iter_end = time.perf_counter()
    anderson_enabled = anderson_memory > 0 and anderson_interval > 0
    steps_seen = 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        iter_start = time.perf_counter()
        if diagnostics:
            data_wait_times.append(iter_start - prev_iter_end)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        last_batch_size = data.size(0)
        last_input_shape = tuple(data.shape[1:])

        measure_this = batch_idx > warmup_steps and measured_steps < measure_steps
        if measure_this:
            timer.start()

        optimizer.zero_grad(set_to_none=True)
        params_flat_before = None
        if anderson_enabled:
            params_flat_before = _flatten_params(params)
        output = model(data)
        loss = F.cross_entropy(output, target)
        curvature = None
        grad_norm = None
        grad_norm_sq_val = None
        needs_grad_norm = step_rule in ("l0l1", "sps", "sps-momentum", "adaptive-backtracking") or (
            grad_norm_every > 0 and global_step % grad_norm_every == 0
        )
        needs_grad_norm_sq = step_rule in ("sps", "sps-momentum", "adaptive-backtracking") or needs_grad_norm
        use_sophia_hessian = direction == "sophia" and (global_step % sophia_hessian_every == 0)
        use_eoss_hvp = step_rule == "eoss" and step_eoss_interval > 0 and (global_step % step_eoss_interval == 0)
        eoss_curvature = None
        sophia_diag_estimates: dict[torch.Tensor, torch.Tensor] = {}
        if use_sophia_hessian or step_rule == "eoss":
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=use_sophia_hessian or use_eoss_hvp,
                allow_unused=True,
            )
            grads_used = []
            params_used = []
            for param, grad in zip(params, grads):
                if grad is None:
                    continue
                grads_used.append(grad)
                params_used.append(param)

            if needs_grad_norm_sq or step_rule == "eoss":
                grad_norm_sq = _sum_tensor_sq(grads_used)
                grad_norm_sq_val = float(grad_norm_sq.item())
                grad_norm = float(torch.sqrt(grad_norm_sq).item())

            if use_eoss_hvp and grads_used:
                hvp = torch.autograd.grad(
                    grads_used,
                    params_used,
                    grad_outputs=grads_used,
                    retain_graph=use_sophia_hessian,
                    allow_unused=True,
                )
                g_norm_sq_val = float(grad_norm_sq_val or 0.0)
                if g_norm_sq_val > 0.0 and math.isfinite(g_norm_sq_val):
                    g_dot_h = 0.0
                    for grad, hv in zip(grads_used, hvp):
                        if hv is None:
                            continue
                        g_dot_h += float((grad.detach() * hv.detach()).sum().item())
                    s_val = g_dot_h / (g_norm_sq_val + step_eoss_eps)
                    if math.isfinite(s_val):
                        s_hat = step_state.get("eoss_curvature_ema")
                        if s_hat is None:
                            s_hat = s_val
                        else:
                            s_hat = step_eoss_ema * float(s_hat) + (1.0 - step_eoss_ema) * s_val
                        step_state["eoss_curvature_ema"] = s_hat
                        step_state["eoss_last_curvature"] = s_val
                        eoss_curvature = float(s_val)

            if use_sophia_hessian and grads_used:
                diag_estimates: list[Optional[torch.Tensor]] = [None for _ in grads_used]
                for sample_idx in range(sophia_hutchinson_samples):
                    rand_vecs = [
                        torch.empty_like(grad).bernoulli_(0.5).mul_(2.0).sub_(1.0) for grad in grads_used
                    ]
                    retain = sample_idx < sophia_hutchinson_samples - 1
                    hvp = torch.autograd.grad(
                        grads_used,
                        params_used,
                        grad_outputs=rand_vecs,
                        retain_graph=retain,
                        allow_unused=True,
                    )
                    for idx, (vec, hv) in enumerate(zip(rand_vecs, hvp)):
                        if hv is None:
                            continue
                        value = vec * hv
                        if diag_estimates[idx] is None:
                            diag_estimates[idx] = value.detach()
                        else:
                            diag_estimates[idx] = diag_estimates[idx] + value.detach()
                for param, estimate in zip(params_used, diag_estimates):
                    if estimate is None:
                        continue
                    sophia_diag_estimates[param] = estimate / float(sophia_hutchinson_samples)

            for param, grad in zip(params, grads):
                param.grad = None if grad is None else grad.detach()
        else:
            loss.backward()
            if needs_grad_norm_sq:
                grad_norm_sq = _sum_grad_sq(params)
                grad_norm_sq_val = float(grad_norm_sq.item())
                grad_norm = float(torch.sqrt(grad_norm_sq).item())

        current_grads = None
        if step_rule == "sagd":
            current_grads = [
                (param.grad.detach().clone() if param.grad is not None else None) for param in params
            ]

        if eoss_curvature is not None:
            curvature = eoss_curvature

        if grad_norm is not None:
            grad_norms.append(grad_norm)

        if curvature is not None:
            curvatures.append(curvature)

        if clip_mode != "none":
            coef = _ggnc_apply(params, optimizer, clip_rho, clip_alpha, step_eps, clip_mode)
            clip_coefs.append(float(coef))

        if direction == "diag-precond":
            if direction_update_every <= 0:
                direction_update_every = 1
            if global_step % direction_update_every == 0:
                for param in params:
                    if param.grad is None:
                        continue
                    state = optimizer.state[param]
                    ema = state.get("gs_dir_ema")
                    if ema is None:
                        ema = torch.zeros_like(param.grad)
                        state["gs_dir_ema"] = ema
                    ema.mul_(direction_beta).addcmul_(param.grad, param.grad, value=1.0 - direction_beta)
            scale_vals: list[float] = []
            for param in params:
                if param.grad is None:
                    continue
                state = optimizer.state[param]
                ema = state.get("gs_dir_ema")
                if ema is None:
                    continue
                scale = (ema + direction_eps).rsqrt()
                param.grad.mul_(scale)
                scale_vals.append(float(scale.mean().item()))
            if scale_vals:
                direction_scales.append(float(sum(scale_vals) / len(scale_vals)))
        elif direction in ("shampoo", "soap"):
            if direction_update_every <= 0:
                direction_update_every = 1
            beta2 = direction_beta
            beta1 = direction_beta1
            step_idx = global_step + 1
            update_precond = step_idx % direction_update_every == 0
            bias_correction1 = 1.0 - beta1**step_idx
            bias_correction2 = 1.0 - beta2**step_idx
            if bias_correction1 <= 0.0:
                bias_correction1 = 1.0
            if bias_correction2 <= 0.0:
                bias_correction2 = 1.0
            with torch.no_grad():
                for idx, param in enumerate(params):
                    if param.grad is None:
                        continue
                    name = param_names.get(id(param), f"param_{idx}")
                    layer_stats = precond_layer_stats.get(name)
                    if layer_stats is None:
                        layer_stats = {
                            "name": name,
                            "shape": list(param.shape),
                            "order": param.ndim,
                            "stat_updates": 0,
                            "precond_updates": 0,
                            "apply_count": 0,
                            "stat_update_time_s": 0.0,
                            "precond_update_time_s": 0.0,
                            "update_time_s": 0.0,
                            "apply_time_s": 0.0,
                        }
                        precond_layer_stats[name] = layer_stats

                    state = optimizer.state[param]
                    stats = state.get("shampoo_stats")
                    if stats is None:
                        stats = [
                            torch.zeros((dim, dim), device=param.device, dtype=param.dtype)
                            for dim in param.shape
                        ]
                        state["shampoo_stats"] = stats

                    _maybe_sync(device, precond_sync)
                    update_start = time.perf_counter()
                    _update_shampoo_stats(param.grad, stats, beta2)
                    _maybe_sync(device, precond_sync)
                    update_elapsed = time.perf_counter() - update_start
                    precond_update_time_s += update_elapsed
                    layer_stats["stat_updates"] += 1
                    layer_stats["stat_update_time_s"] += update_elapsed
                    layer_stats["update_time_s"] += update_elapsed

                    if direction == "shampoo":
                        if update_precond:
                            _maybe_sync(device, precond_sync)
                            precond_start = time.perf_counter()
                            order = max(1, param.ndim)
                            power = -1.0 / (2.0 * order)
                            inv_roots = [
                                _matrix_inverse_power(stat, power, direction_damping) for stat in stats
                            ]
                            state["shampoo_inv_root"] = inv_roots
                            _maybe_sync(device, precond_sync)
                            precond_elapsed = time.perf_counter() - precond_start
                            precond_update_time_s += precond_elapsed
                            precond_update_count += 1
                            layer_stats["precond_updates"] += 1
                            layer_stats["precond_update_time_s"] += precond_elapsed
                            layer_stats["update_time_s"] += precond_elapsed

                        _maybe_sync(device, precond_sync)
                        apply_start = time.perf_counter()
                        inv_roots = state.get("shampoo_inv_root")
                        if inv_roots is None:
                            inv_roots = [
                                torch.eye(dim, device=param.device, dtype=param.dtype)
                                for dim in param.shape
                            ]
                            state["shampoo_inv_root"] = inv_roots
                        precond_grad = _apply_preconditioners(param.grad, inv_roots)
                        param.grad.copy_(precond_grad)
                        _maybe_sync(device, precond_sync)
                        apply_elapsed = time.perf_counter() - apply_start
                        precond_apply_time_s += apply_elapsed
                        precond_apply_count += 1
                        layer_stats["apply_count"] += 1
                        layer_stats["apply_time_s"] += apply_elapsed
                    else:
                        bases = state.get("soap_basis")
                        if bases is None or len(bases) != len(param.shape):
                            bases = [
                                torch.eye(dim, device=param.device, dtype=param.dtype)
                                for dim in param.shape
                            ]
                            state["soap_basis"] = bases
                        else:
                            for dim, basis in zip(param.shape, bases):
                                if basis.shape != (dim, dim):
                                    bases = [
                                        torch.eye(dim, device=param.device, dtype=param.dtype)
                                        for dim in param.shape
                                    ]
                                    state["soap_basis"] = bases
                                    break

                        _maybe_sync(device, precond_sync)
                        apply_start = time.perf_counter()
                        proj = _apply_preconditioners(param.grad, [basis.t() for basis in bases])
                        m = state.get("soap_m")
                        if m is None:
                            m = torch.zeros_like(param.grad)
                        m.mul_(beta1).add_(param.grad, alpha=1.0 - beta1)
                        state["soap_m"] = m
                        v = state.get("soap_v")
                        if v is None:
                            v = torch.zeros_like(proj)
                        v.mul_(beta2).addcmul_(proj, proj, value=1.0 - beta2)
                        state["soap_v"] = v
                        m_proj = _apply_preconditioners(m, [basis.t() for basis in bases])
                        if bias_correction1 > 0.0:
                            m_hat = m_proj / bias_correction1
                        else:
                            m_hat = m_proj
                        if bias_correction2 > 0.0:
                            v_hat = v / bias_correction2
                        else:
                            v_hat = v
                        update = m_hat / (torch.sqrt(v_hat) + direction_eps)
                        precond_grad = _apply_preconditioners(update, bases)
                        param.grad.copy_(precond_grad)
                        _maybe_sync(device, precond_sync)
                        apply_elapsed = time.perf_counter() - apply_start
                        precond_apply_time_s += apply_elapsed
                        precond_apply_count += 1
                        layer_stats["apply_count"] += 1
                        layer_stats["apply_time_s"] += apply_elapsed

                        if update_precond:
                            _maybe_sync(device, precond_sync)
                            precond_start = time.perf_counter()
                            new_bases = []
                            for stat, prev_basis in zip(stats, bases):
                                mat = stat
                                if direction_damping > 0.0:
                                    eye = torch.eye(stat.shape[0], device=stat.device, dtype=stat.dtype)
                                    mat = mat + direction_damping * eye
                                new_bases.append(_soap_eigenvectors(mat, prev_basis))
                            state["soap_basis"] = new_bases
                            _maybe_sync(device, precond_sync)
                            precond_elapsed = time.perf_counter() - precond_start
                            precond_update_time_s += precond_elapsed
                            precond_update_count += 1
                            layer_stats["precond_updates"] += 1
                            layer_stats["precond_update_time_s"] += precond_elapsed
                            layer_stats["update_time_s"] += precond_elapsed
        elif direction == "sophia":
            step_hessian_vals: list[float] = []
            step_clip_fracs: list[float] = []
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad
                state = optimizer.state[param]
                m = state.get("sophia_m")
                if m is None:
                    m = torch.zeros_like(grad)
                m.mul_(sophia_beta1).add_(grad, alpha=1.0 - sophia_beta1)
                state["sophia_m"] = m
                h = state.get("sophia_h")
                if h is None:
                    h = torch.zeros_like(grad)
                if use_sophia_hessian:
                    diag_est = sophia_diag_estimates.get(param)
                    if diag_est is not None:
                        h.mul_(sophia_beta2).add_(diag_est, alpha=1.0 - sophia_beta2)
                state["sophia_h"] = h
                h_pos = torch.clamp(h, min=0.0)
                denom = torch.clamp(h_pos * sophia_gamma, min=sophia_eps)
                update = m / denom
                clip_mask = update.abs() > 1.0
                if clip_mask.any():
                    step_clip_fracs.append(float(clip_mask.float().mean().item()))
                else:
                    step_clip_fracs.append(0.0)
                update = torch.clamp(update, min=-1.0, max=1.0)
                param.grad = update
                step_hessian_vals.append(float(h_pos.mean().item()))
            if step_hessian_vals:
                sophia_hessian_vals.append(float(sum(step_hessian_vals) / len(step_hessian_vals)))
            if step_clip_fracs:
                sophia_clip_fracs.append(float(sum(step_clip_fracs) / len(step_clip_fracs)))
        elif direction == "muon":
            step_iters: list[float] = []
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad
                state = optimizer.state[param]
                m = state.get("muon_m")
                if m is None:
                    m = torch.zeros_like(grad)
                m.mul_(muon_beta).add_(grad, alpha=1.0 - muon_beta)
                state["muon_m"] = m
                update = m
                if update.ndim >= 2:
                    update, iters = _muon_orthogonalize(update, muon_ns_iters, muon_eps)
                    if iters:
                        step_iters.append(float(iters))
                    scale = _muon_scale_factor(
                        update,
                        param,
                        muon_scale_mode,
                        muon_rms_scale,
                        muon_hidden_size,
                        muon_eps,
                    )
                    update = update * scale
                param.grad = update
            if step_iters:
                muon_ortho_iters.append(float(sum(step_iters) / len(step_iters)))

        step_size = None
        line_search_iters = None
        line_search_accepted_step = None
        applied_update = False
        loss_value = float(loss.item())
        if step_rule != "none":
            if step_rule == "eoss":
                s_hat = step_state.get("eoss_curvature_ema")
                base_lr = step_state.get("eoss_base_lr")
                if base_lr is None:
                    if base_lrs:
                        base_lr = float(base_lrs[0])
                    else:
                        base_lr = float(optimizer.param_groups[0]["lr"])
                    step_state["eoss_base_lr"] = base_lr
                step_size = float(base_lr)
                if s_hat is not None and math.isfinite(float(s_hat)):
                    denom = float(s_hat) + step_eoss_eps
                    if denom <= step_eps:
                        denom = step_eps
                    step_size = float(step_eoss_beta) * 2.0 / denom
                    step_size = min(max(step_size, step_eoss_clip_min), step_eoss_clip_max)
                new_lrs = [step_size for _ in base_lrs] if base_lrs else [step_size]
                step_state["eoss_last_step_size"] = step_size
            elif step_rule == "l0l1":
                # L0L1-GD update (arXiv:2409.14989 Algorithm 1).
                _zero_sgd_momentum(optimizer)
                grad_norm_val = grad_norm
                if grad_norm_val is None:
                    grad_norm_val = math.sqrt(max(grad_norm_sq_val or 0.0, 0.0))
                denom = step_l0 + step_l1 * grad_norm_val
                if not denom or denom <= step_eps:
                    denom = step_eps
                new_lrs = [base_lr / denom for base_lr in base_lrs]
            elif step_rule in ("sps", "sps-momentum"):
                # SPS / MomSPS update (arXiv:2409.14989 Algorithm 2; arXiv:2406.04142 Algorithm 1).
                if grad_norm_sq_val is None:
                    grad_norm_sq_val = float(_sum_grad_sq(params).item())
                numer = max(loss_value - step_fstar, 0.0)
                denom = grad_norm_sq_val + step_eps
                scale = numer / denom if denom > 0.0 else 0.0
                if step_rule == "sps-momentum":
                    _zero_sgd_momentum(optimizer)
                    scale = step_sps_c * scale
                    cap = scale if step_sps_max is None else min(scale, step_sps_max)
                    step_size = (1.0 - step_sps_beta) * max(cap, 0.0)
                    prev_params = step_state.get("sps_momentum_prev_params")
                    if prev_params is None or len(prev_params) != len(params):
                        prev_params = [param.detach().clone() for param in params]
                    current_params = [param.detach().clone() for param in params]
                    with torch.no_grad():
                        prev_param_map = {id(param): prev for param, prev in zip(params, prev_params)}
                        cur_param_map = {id(param): cur for param, cur in zip(params, current_params)}
                        for group in optimizer.param_groups:
                            weight_decay = float(group.get("weight_decay", 0.0))
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                cur_param = cur_param_map[id(param)]
                                prev_param = prev_param_map[id(param)]
                                grad = param.grad.detach()
                                if weight_decay:
                                    grad = grad.add(cur_param, alpha=weight_decay)
                                momentum_term = cur_param - prev_param
                                param.data.copy_(cur_param)
                                param.data.add_(grad, alpha=-step_size)
                                param.data.add_(momentum_term, alpha=step_sps_beta)
                    step_state["sps_momentum_prev_params"] = current_params
                    new_lrs = [step_size for _ in base_lrs]
                    applied_update = True
                else:
                    _zero_sgd_momentum(optimizer)
                    new_lrs = [scale for _ in base_lrs]
            elif step_rule == "silver":
                # Silver step-size schedule (COLT/JMLR 2025; see method-conformance.md).
                _zero_sgd_momentum(optimizer)
                t_step = global_step + 1
                v_idx = _silver_schedule_index(t_step)
                factor = 1.0 + (step_silver_rho ** max(v_idx - 1, 0))
                new_lrs = [base_lr * factor for base_lr in base_lrs]
            elif step_rule == "sagd":
                # SAGD without descent (arXiv:2509.14969 Algorithm 1, Variant III).
                _zero_sgd_momentum(optimizer)
                lambda0 = base_lrs[0] if base_lrs else float(optimizer.param_groups[0]["lr"])
                step_size = lambda0
                prev_batch = step_state.get("sagd_prev_batch")
                prev_grads = step_state.get("sagd_prev_grads")
                prev_params = step_state.get("sagd_prev_params")
                prev_lr = float(step_state.get("sagd_prev_lr", lambda0))
                prev_prev_lr = float(step_state.get("sagd_prev_prev_lr", prev_lr))

                if prev_batch is not None and prev_grads is not None and prev_params is not None:
                    prev_data, prev_target = prev_batch
                    prev_data = prev_data.to(device, non_blocking=True)
                    prev_target = prev_target.to(device, non_blocking=True)
                    buffer_snapshot = [buf.detach().clone() for buf in model.buffers()]
                    prev_output = model(prev_data)
                    prev_loss = F.cross_entropy(prev_output, prev_target)
                    prev_grads_cur = torch.autograd.grad(prev_loss, params, allow_unused=True)
                    prev_grads_cur = [g.detach() if g is not None else None for g in prev_grads_cur]
                    for buf, saved in zip(model.buffers(), buffer_snapshot):
                        buf.copy_(saved)

                    grad_diff_sq = 0.0
                    for grad_cur, grad_prev in zip(prev_grads_cur, prev_grads):
                        if grad_cur is None and grad_prev is None:
                            continue
                        if grad_cur is None:
                            diff = -grad_prev
                        elif grad_prev is None:
                            diff = grad_cur
                        else:
                            diff = grad_cur - grad_prev
                        grad_diff_sq += float((diff * diff).sum().item())
                    grad_diff_norm = math.sqrt(grad_diff_sq)

                    delta_sq = 0.0
                    for cur_param, prev_param in zip(params, prev_params):
                        diff = cur_param.detach() - prev_param
                        delta_sq += float((diff * diff).sum().item())
                    delta_norm = math.sqrt(delta_sq)

                    if grad_diff_norm > step_eps and delta_norm > step_eps:
                        if global_step == 1:
                            step_size = delta_norm / (2.0 * math.sqrt(2.0) * grad_diff_norm)
                        else:
                            k = max(global_step, 1)
                            k_pow = k ** (0.5 + step_sagd_delta)
                            l_hat = grad_diff_norm / max(delta_norm, step_eps)
                            term1 = 1.0 / (2.0 * math.sqrt(2.0) * max(l_hat, step_eps) * k_pow)
                            ratio = prev_lr / max(prev_prev_lr, step_eps)
                            term2 = prev_lr * math.sqrt(
                                max(1.0 + (1.0 - 1.0 / k_pow) * ratio, 0.0)
                            )
                            step_size = min(term1, term2)

                new_lrs = [step_size for _ in base_lrs]
                if current_grads is not None:
                    step_state["sagd_prev_prev_lr"] = prev_lr
                    step_state["sagd_prev_lr"] = step_size
                    step_state["sagd_prev_batch"] = (data.detach(), target.detach())
                    step_state["sagd_prev_params"] = [param.detach().clone() for param in params]
                    step_state["sagd_prev_grads"] = current_grads
            elif step_rule == "adaptive-backtracking":
                # Adaptive backtracking line search (arXiv:2408.13150 Eq. 4a/4b, Algorithm 2).
                _zero_sgd_momentum(optimizer)
                if grad_norm_sq_val is None:
                    grad_norm_sq_val = float(_sum_grad_sq(params).item())
                start_lr = base_lrs[0] if base_lrs else optimizer.param_groups[0]["lr"]
                if "backtrack_lr" in step_state:
                    start_lr = float(step_state["backtrack_lr"])
                start_lr = float(start_lr)
                if not math.isfinite(start_lr) or start_lr <= 0.0:
                    start_lr = base_lrs[0] if base_lrs else float(optimizer.param_groups[0]["lr"])

                param_snapshot = [param.detach().clone() for param in params]
                buffer_snapshot = [buf.detach().clone() for buf in model.buffers()]
                optimizer_state = copy.deepcopy(optimizer.state_dict())
                line_search_attempts += 1

                for attempt in range(1, step_backtrack_max + 1):
                    line_search_iters = attempt
                    for group in optimizer.param_groups:
                        group["lr"] = start_lr

                    if direction in ("sophia", "muon", "shampoo", "soap"):
                        _apply_manual_update(optimizer)
                    else:
                        optimizer.step()

                    with torch.no_grad():
                        trial_output = model(data)
                        trial_loss = F.cross_entropy(trial_output, target)
                        trial_value = float(trial_loss.item())
                        for buf, saved in zip(model.buffers(), buffer_snapshot):
                            buf.copy_(saved)

                    denom = max(step_backtrack_c * start_lr * grad_norm_sq_val, step_eps)
                    v_val = (loss_value - trial_value) / denom
                    if v_val >= 1.0:
                        line_search_accepted_step = True
                        line_search_accepted += 1
                        step_size = start_lr
                        step_state = optimizer.param_groups[0].setdefault("step_control_state", {})
                        step_state["backtrack_lr"] = start_lr
                        applied_update = True
                        break

                    line_search_accepted_step = False
                    for param, saved in zip(params, param_snapshot):
                        param.data.copy_(saved)
                    optimizer.load_state_dict(optimizer_state)
                    step_state = optimizer.param_groups[0].setdefault("step_control_state", {})
                    denom_factor = 1.0 - step_backtrack_c * v_val
                    if denom_factor <= step_eps:
                        shrink = step_backtrack_rho
                    else:
                        exponent = (1.0 - step_backtrack_c) / denom_factor
                        shrink = step_backtrack_rho ** exponent
                    shrink = min(max(shrink, step_eps), 1.0)
                    start_lr = max(start_lr * shrink, step_eps)

                if not applied_update:
                    for group in optimizer.param_groups:
                        group["lr"] = start_lr
                    if direction in ("sophia", "muon", "shampoo", "soap"):
                        _apply_manual_update(optimizer)
                    else:
                        optimizer.step()
                    line_search_rejected += 1
                    line_search_accepted_step = False
                    step_size = start_lr
                    step_state = optimizer.param_groups[0].setdefault("step_control_state", {})
                    step_state["backtrack_lr"] = start_lr
                    applied_update = True
            else:
                new_lrs = list(base_lrs)
            if step_rule != "adaptive-backtracking":
                for group, lr in zip(optimizer.param_groups, new_lrs):
                    group["lr"] = lr

        if not applied_update and sparsity_enabled and sparse_params:
            update_mask = global_step % sparsity_update_interval == 0
            sparse_updates: list[tuple[torch.Tensor, torch.Tensor, float]] = []
            for group in optimizer.param_groups:
                lr = group["lr"]
                weight_decay = float(group.get("weight_decay", 0.0))
                for param in group["params"]:
                    if param.grad is None or not param.requires_grad:
                        continue
                    if not _should_sparsify(param):
                        continue
                    grad = param.grad.detach()
                    if weight_decay:
                        grad = grad.add(param.data, alpha=weight_decay)
                    sparse_updates.append((param, grad, float(lr)))
                    param.grad = None
            if direction in ("sophia", "muon", "shampoo", "soap"):
                _apply_manual_update(optimizer)
            else:
                optimizer.step()
            if sparse_updates:
                with torch.no_grad():
                    for param, grad, lr in sparse_updates:
                        state = optimizer.state[param]
                        z = state.get("linbreg_z")
                        if z is None:
                            z = param.data.detach().clone()
                            state["linbreg_z"] = z
                        z.add_(grad, alpha=-lr)
                    if update_mask:
                        for param, _, _ in sparse_updates:
                            z = optimizer.state[param]["linbreg_z"]
                            param.data.copy_(_linbreg_shrink(z, sparsity_lambda))
                        sparsity_update_count += 1
                        if dense_flops_per_sample is None and last_input_shape is not None:
                            dense_flops_per_sample = _estimate_dense_flops_per_sample(
                                model,
                                (1,) + last_input_shape,
                                device,
                            )
                            model._gs_dense_flops_per_sample = dense_flops_per_sample
                        nonzero, total = _sparsity_counts(sparse_params)
                        if total > 0:
                            sparsity_fracs.append(1.0 - (nonzero / total))
                            if dense_flops_per_sample is not None and last_batch_size is not None:
                                dense_flops = float(dense_flops_per_sample * last_batch_size)
                                dense_flops_vals.append(dense_flops)
                                effective_flops_vals.append(float(dense_flops * (nonzero / total)))
        elif not applied_update:
            if direction in ("sophia", "muon", "shampoo", "soap"):
                _apply_manual_update(optimizer)
            else:
                optimizer.step()

        if step_size is None and optimizer.param_groups:
            step_size = float(optimizer.param_groups[0]["lr"])
        if step_size is not None:
            step_sizes.append(step_size)
        if line_search_iters is not None:
            line_search_iters_list.append(line_search_iters)

        if anderson_enabled:
            params_flat_after = _flatten_params(params)
            if params_flat_before is None:
                params_flat_before = params_flat_after.detach().clone()
            residual = params_flat_after - params_flat_before
            anderson_state.append((params_flat_before.detach(), residual.detach()))
            if len(anderson_state) > anderson_memory + 1:
                anderson_state.pop(0)
            if global_step % anderson_interval == 0 and len(anderson_state) >= 2:
                m_hist = min(anderson_memory, len(anderson_state) - 1)
                if m_hist >= 1 and anderson_damping > 0.0:
                    try:
                        use_states = anderson_state[-(m_hist + 1) :]
                        x_mat = torch.stack([state[0] for state in use_states], dim=1)
                        r_mat = torch.stack([state[1] for state in use_states], dim=1)
                        reg = max(anderson_lambda, 0.0)
                        alpha = _anderson_coefficients(r_mat, reg, step_eps)
                        if alpha is None:
                            anderson_failed += 1
                        else:
                            x_accel = x_mat @ alpha
                            r_accel = r_mat @ alpha
                            if torch.isfinite(x_accel).all() and torch.isfinite(r_accel).all():
                                damping = min(anderson_damping, 1.0)
                                x_new = x_accel + damping * r_accel
                                if torch.isfinite(x_new).all():
                                    _assign_params(params, x_new)
                                    anderson_applied += 1
                                else:
                                    anderson_failed += 1
                            else:
                                anderson_failed += 1
                    except Exception:
                        anderson_failed += 1

        if measure_this:
            elapsed = timer.stop()
            step_times.append(elapsed)
            measured_samples += data.size(0)
            measured_steps += 1

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        if log_fn is not None and log_interval > 0 and global_step % log_interval == 0:
            batch_acc = pred.eq(target).sum().item() / max(data.size(0), 1)
            lr = optimizer.param_groups[0]["lr"]
            step_time_ms = None
            if measure_this:
                step_time_ms = elapsed * 1000.0
            log_fn(
                StepLog(
                    epoch=epoch,
                    step_in_epoch=batch_idx,
                    global_step=global_step,
                    loss=float(loss.item()),
                    accuracy=float(batch_acc),
                    lr=float(lr),
                    step_size=step_size,
                    grad_norm=grad_norm,
                    curvature=curvature,
                    step_time_ms=step_time_ms,
                    line_search_iters=line_search_iters,
                    line_search_accepted=line_search_accepted_step,
                )
            )

        steps_seen += 1
        global_step += 1
        if on_step_end is not None and on_step_end(global_step, epoch, batch_idx):
            break
        prev_iter_end = time.perf_counter()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    if step_times:
        avg_step = sum(step_times) / len(step_times)
        throughput = measured_samples / max(sum(step_times), 1e-12)
        step_time_ms = avg_step * 1000.0
    else:
        step_time_ms = 0.0
        throughput = 0.0

    step_size_mean = step_size_p50 = step_size_p90 = None
    if step_sizes:
        step_size_mean, step_size_p50, step_size_p90 = _percentiles(step_sizes)

    grad_norm_mean = grad_norm_p50 = grad_norm_p90 = None
    if grad_norms:
        grad_norm_mean, grad_norm_p50, grad_norm_p90 = _percentiles(grad_norms)

    curvature_mean = curvature_p50 = curvature_p90 = None
    if curvatures:
        curvature_mean, curvature_p50, curvature_p90 = _percentiles(curvatures)

    direction_scale_mean = direction_scale_p50 = direction_scale_p90 = None
    if direction_scales:
        direction_scale_mean, direction_scale_p50, direction_scale_p90 = _percentiles(direction_scales)

    clip_coef_mean = clip_coef_p50 = clip_coef_p90 = None
    if clip_coefs:
        clip_coef_mean, clip_coef_p50, clip_coef_p90 = _percentiles(clip_coefs)

    sophia_hessian_mean = sophia_hessian_p50 = sophia_hessian_p90 = None
    if sophia_hessian_vals:
        sophia_hessian_mean, sophia_hessian_p50, sophia_hessian_p90 = _percentiles(sophia_hessian_vals)

    sophia_clip_frac_mean = sophia_clip_frac_p50 = sophia_clip_frac_p90 = None
    if sophia_clip_fracs:
        sophia_clip_frac_mean, sophia_clip_frac_p50, sophia_clip_frac_p90 = _percentiles(sophia_clip_fracs)

    muon_ortho_iters_mean = muon_ortho_iters_p50 = muon_ortho_iters_p90 = None
    if muon_ortho_iters:
        muon_ortho_iters_mean, muon_ortho_iters_p50, muon_ortho_iters_p90 = _percentiles(muon_ortho_iters)

    line_search_iters_mean = line_search_iters_p50 = line_search_iters_p90 = None
    if line_search_iters_list:
        line_search_iters_mean, line_search_iters_p50, line_search_iters_p90 = _percentiles(line_search_iters_list)

    precond_layer_list = None
    if precond_layer_stats:
        precond_layer_list = [precond_layer_stats[name] for name in sorted(precond_layer_stats)]

    sparsity_fraction = None
    dense_flops = None
    effective_flops = None
    sparsity_update_rate = None
    if sparsity_enabled and sparse_params:
        if dense_flops_per_sample is None and last_input_shape is not None:
            dense_flops_per_sample = _estimate_dense_flops_per_sample(
                model,
                (1,) + last_input_shape,
                device,
            )
            model._gs_dense_flops_per_sample = dense_flops_per_sample
        if sparsity_fracs:
            sparsity_fraction = float(sum(sparsity_fracs) / len(sparsity_fracs))
        else:
            nonzero, total = _sparsity_counts(sparse_params)
            if total > 0:
                sparsity_fraction = float(1.0 - (nonzero / total))
        if dense_flops_vals:
            dense_flops = float(sum(dense_flops_vals) / len(dense_flops_vals))
        elif dense_flops_per_sample is not None and last_batch_size is not None:
            dense_flops = float(dense_flops_per_sample * last_batch_size)
        if effective_flops_vals:
            effective_flops = float(sum(effective_flops_vals) / len(effective_flops_vals))
        elif dense_flops is not None and sparsity_fraction is not None:
            effective_flops = float(dense_flops * (1.0 - sparsity_fraction))
        if steps_seen > 0:
            sparsity_update_rate = float(sparsity_update_count / steps_seen)

    data_wait_time_s = None
    max_memory_bytes = None
    if diagnostics:
        data_wait_time_s = float(sum(data_wait_times))
        if device.type == "cuda":
            max_memory_bytes = int(torch.cuda.max_memory_allocated())

    metrics = TrainMetrics(
        loss=float(avg_loss),
        accuracy=float(accuracy),
        step_time_ms=float(step_time_ms),
        throughput=float(throughput),
        steps=steps_seen,
        samples=total,
        step_time_count=len(step_times),
        step_time_total_s=float(sum(step_times)),
        step_size_mean=step_size_mean,
        step_size_p50=step_size_p50,
        step_size_p90=step_size_p90,
        grad_norm_mean=grad_norm_mean,
        grad_norm_p50=grad_norm_p50,
        grad_norm_p90=grad_norm_p90,
        curvature_mean=curvature_mean,
        curvature_p50=curvature_p50,
        curvature_p90=curvature_p90,
        direction_scale_mean=direction_scale_mean,
        direction_scale_p50=direction_scale_p50,
        direction_scale_p90=direction_scale_p90,
        clip_coef_mean=clip_coef_mean,
        clip_coef_p50=clip_coef_p50,
        clip_coef_p90=clip_coef_p90,
        sophia_hessian_mean=sophia_hessian_mean,
        sophia_hessian_p50=sophia_hessian_p50,
        sophia_hessian_p90=sophia_hessian_p90,
        sophia_clip_frac_mean=sophia_clip_frac_mean,
        sophia_clip_frac_p50=sophia_clip_frac_p50,
        sophia_clip_frac_p90=sophia_clip_frac_p90,
        muon_ortho_iters_mean=muon_ortho_iters_mean,
        muon_ortho_iters_p50=muon_ortho_iters_p50,
        muon_ortho_iters_p90=muon_ortho_iters_p90,
        line_search_attempts=line_search_attempts,
        line_search_accepted=line_search_accepted,
        line_search_rejected=line_search_rejected,
        line_search_iters_mean=line_search_iters_mean,
        line_search_iters_p50=line_search_iters_p50,
        line_search_iters_p90=line_search_iters_p90,
        precond_update_count=precond_update_count,
        precond_apply_count=precond_apply_count,
        precond_update_time_s=float(precond_update_time_s),
        precond_apply_time_s=float(precond_apply_time_s),
        precond_layer_stats=precond_layer_list,
        anderson_applied=anderson_applied,
        anderson_failed=anderson_failed,
        data_wait_time_s=data_wait_time_s,
        max_memory_bytes=max_memory_bytes,
        sparsity_fraction=sparsity_fraction,
        dense_flops=dense_flops,
        effective_flops=effective_flops,
        sparsity_updates=sparsity_update_count,
        sparsity_update_interval=sparsity_update_interval if sparsity_enabled and sparse_params else None,
        sparsity_update_rate=sparsity_update_rate,
    )
    return metrics, global_step


def _silver_schedule_index(t_step: int) -> int:
    """Return v(t) where v(t) = max { v : t >= F_v }, F_0=0, F_1=1."""
    if t_step <= 0:
        return 0
    f_prev, f_curr = 0, 1
    v_idx = 1
    while t_step >= f_curr:
        f_prev, f_curr = f_curr, f_prev + f_curr
        v_idx += 1
    return v_idx - 1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        loss = F.cross_entropy(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return EvalMetrics(loss=float(avg_loss), accuracy=float(accuracy), samples=total)
