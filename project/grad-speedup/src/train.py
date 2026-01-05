from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .modules import SUPPORTED_CLIP_MODES, SUPPORTED_DIRECTIONS, SUPPORTED_STEP_RULES

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
    anderson_applied: int = 0
    anderson_failed: int = 0
    data_wait_time_s: Optional[float] = None
    max_memory_bytes: Optional[int] = None


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
    grad_norm: Optional[float]
    curvature: Optional[float]
    step_time_ms: Optional[float]


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


def _clip_grads_global(params: Iterable[torch.Tensor], max_norm: float, eps: float) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += param.grad.detach().pow(2).sum().item()
    total_norm = float(total) ** 0.5
    if total_norm <= max_norm or total_norm <= 0.0:
        return 1.0
    coef = max_norm / (total_norm + eps)
    for param in params:
        if param.grad is None:
            continue
        param.grad.mul_(coef)
    return float(coef)


def _clip_grads_layerwise(params: Iterable[torch.Tensor], max_norm: float, eps: float) -> float:
    coefs: list[float] = []
    for param in params:
        if param.grad is None:
            continue
        norm = float(param.grad.detach().pow(2).sum().item()) ** 0.5
        if norm <= max_norm or norm <= 0.0:
            coefs.append(1.0)
            continue
        coef = max_norm / (norm + eps)
        param.grad.mul_(coef)
        coefs.append(float(coef))
    if not coefs:
        return 1.0
    return float(sum(coefs) / len(coefs))


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 90))


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


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int,
    log_fn: Optional[Callable[[StepLog], None]] = None,
    warmup_steps: int = 50,
    measure_steps: int = 200,
    grad_norm_every: int = 0,
    step_rule: str = "none",
    step_l0: float = 1.0,
    step_l1: float = 0.0,
    step_curv_every: int = 50,
    step_curv_eps: float = 1e-8,
    step_eoss_beta: float = 1.0,
    step_silver_rho: float = 2.414213562373095,
    step_eps: float = 1e-12,
    direction: str = "none",
    direction_beta: float = 0.9,
    direction_eps: float = 1e-8,
    direction_update_every: int = 1,
    clip_mode: str = "none",
    clip_rho: float = 1.0,
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
    if step_rule == "eoss" and step_curv_every <= 0:
        raise ValueError("step_curv_every must be > 0 for eoss")
    if step_rule == "silver" and step_silver_rho <= 0.0:
        raise ValueError("step_silver_rho must be > 0 for silver")
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(f"unsupported direction: {direction}")
    if clip_mode not in SUPPORTED_CLIP_MODES:
        raise ValueError(f"unsupported clip_mode: {clip_mode}")
    if anderson_memory < 0 or anderson_interval < 0:
        raise ValueError("anderson_memory and anderson_interval must be >= 0")
    if anderson_state is None:
        anderson_state = []

    params = [param for param in model.parameters() if param.requires_grad]
    base_lrs: list[float] = []
    if step_rule != "none":
        for group in optimizer.param_groups:
            base_lr = group.get("base_lr")
            if base_lr is None:
                base_lr = group["lr"]
                group["base_lr"] = base_lr
            base_lrs.append(base_lr)

    total_loss = 0.0
    correct = 0
    total = 0

    step_times: list[float] = []
    measured_samples = 0
    measured_steps = 0
    grad_norms: list[float] = []
    curvatures: list[float] = []
    current_curvature: Optional[float] = None
    direction_scales: list[float] = []
    clip_coefs: list[float] = []
    anderson_applied = 0
    anderson_failed = 0

    timer = StepTimer(device)
    if diagnostics and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    data_wait_times: list[float] = []
    prev_iter_end = time.perf_counter()
    anderson_enabled = anderson_memory > 0 and anderson_interval > 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        iter_start = time.perf_counter()
        if diagnostics:
            data_wait_times.append(iter_start - prev_iter_end)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

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

        use_curvature = step_rule == "eoss" and (global_step % step_curv_every == 0)
        if use_curvature:
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            grads_used = []
            params_used = []
            for param, grad in zip(params, grads):
                if grad is None:
                    continue
                grads_used.append(grad)
                params_used.append(param)

            grad_norm_sq = _sum_grad_sq(grads_used)
            grad_norm = float(torch.sqrt(grad_norm_sq).item())
            if grads_used:
                hvp = torch.autograd.grad(
                    grads_used,
                    params_used,
                    grad_outputs=grads_used,
                    allow_unused=True,
                )
                curv_num = None
                for grad, hv in zip(grads_used, hvp):
                    if hv is None:
                        continue
                    value = (grad.detach() * hv).sum()
                    curv_num = value if curv_num is None else curv_num + value
                if curv_num is not None:
                    curvature_val = float((curv_num / (grad_norm_sq + step_curv_eps)).detach().item())
                    if curvature_val > 0.0 and curvature_val == curvature_val:
                        curvature = curvature_val
            for param, grad in zip(params, grads):
                param.grad = None if grad is None else grad.detach()
        else:
            loss.backward()
            if step_rule in ("l0l1", "eoss") or (grad_norm_every > 0 and global_step % grad_norm_every == 0):
                grad_norm = compute_grad_norm(params)

        if grad_norm is not None:
            grad_norms.append(grad_norm)

        if curvature is not None:
            current_curvature = curvature
            curvatures.append(curvature)

        if clip_mode != "none":
            if clip_mode == "global":
                coef = _clip_grads_global(params, clip_rho, step_eps)
            else:
                coef = _clip_grads_layerwise(params, clip_rho, step_eps)
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

        if step_rule != "none":
            if step_rule == "l0l1":
                denom = step_l0 + step_l1 * (grad_norm or 0.0)
                if not denom or denom <= step_eps:
                    denom = step_eps
                new_lrs = [base_lr / denom for base_lr in base_lrs]
            elif step_rule == "eoss":
                if current_curvature is None or current_curvature <= step_curv_eps:
                    new_lrs = list(base_lrs)
                else:
                    cap = step_eoss_beta * 2.0 / (current_curvature + step_curv_eps)
                    new_lrs = [min(base_lr, cap) for base_lr in base_lrs]
            elif step_rule == "silver":
                t_step = max(global_step, 0)
                v_idx = _silver_schedule_index(t_step)
                factor = 1.0 + (step_silver_rho ** max(v_idx - 1, 0))
                new_lrs = [base_lr * factor for base_lr in base_lrs]
            else:
                new_lrs = list(base_lrs)
            for group, lr in zip(optimizer.param_groups, new_lrs):
                group["lr"] = lr

        optimizer.step()

        if anderson_enabled:
            params_flat_after = _flatten_params(params)
            if params_flat_before is None:
                params_flat_before = params_flat_after.detach().clone()
            residual = params_flat_after - params_flat_before
            anderson_state.append((params_flat_after.detach(), residual.detach()))
            if len(anderson_state) > anderson_memory + 1:
                anderson_state.pop(0)
            if global_step % anderson_interval == 0 and len(anderson_state) >= 2:
                try:
                    m_hist = min(anderson_memory, len(anderson_state) - 1)
                    f_k, r_k = anderson_state[-1]
                    cols_r = []
                    cols_f = []
                    for i in range(1, m_hist + 1):
                        f_i, r_i = anderson_state[-1 - i]
                        cols_r.append(r_i - r_k)
                        cols_f.append(f_i - f_k)
                    if cols_r:
                        r_mat = torch.stack(cols_r, dim=1)
                        f_mat = torch.stack(cols_f, dim=1)
                        rt_r = r_mat.t() @ r_mat
                        rhs = r_mat.t() @ r_k
                        eye = torch.eye(rt_r.shape[0], device=rt_r.device, dtype=rt_r.dtype)
                        gamma = torch.linalg.solve(rt_r + anderson_lambda * eye, rhs)
                        f_accel = f_k - f_mat @ gamma
                        if torch.isfinite(f_accel).all():
                            f_new = (1.0 - anderson_damping) * f_k + anderson_damping * f_accel
                            _assign_params(params, f_new)
                            anderson_applied += 1
                            anderson_state[-1] = (f_new.detach(), (f_new - params_flat_before).detach())
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
                    grad_norm=grad_norm,
                    curvature=curvature,
                    step_time_ms=step_time_ms,
                )
            )

        global_step += 1
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
        steps=len(loader),
        samples=total,
        step_time_count=len(step_times),
        step_time_total_s=float(sum(step_times)),
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
        anderson_applied=anderson_applied,
        anderson_failed=anderson_failed,
        data_wait_time_s=data_wait_time_s,
        max_memory_bytes=max_memory_bytes,
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
