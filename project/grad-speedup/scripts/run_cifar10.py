from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchvision

SCRIPT_DIR = Path(__file__).resolve().parent
GS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = GS_ROOT.parent
sys.path.insert(0, str(GS_ROOT))

from src.data import DataConfig, get_cifar10_loaders  # noqa: E402
from src.models import ModelConfig, build_model  # noqa: E402
from src.train import EvalMetrics, StepLog, TrainMetrics, evaluate, set_seed, train_one_epoch  # noqa: E402


MODEL_CHOICES = ("resnet18", "small-cnn")
OPT_CHOICES = ("sgd", "adam")
MUON_SCALE_CHOICES = ("none", "baseline", "update-norm", "adjusted-lr")
GN_LAYER_CHOICES = ("all", "topk", "bottomk", "randomk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grad-speedup CIFAR-10 runner")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=OPT_CHOICES, default="sgd")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default="runs/grad-speedup")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--log-interval-steps", type=int, default=100)
    parser.add_argument("--eval-interval-epochs", type=int, default=0)
    parser.add_argument("--eval-interval-steps", type=int, default=200)
    parser.add_argument("--target-acc", type=str, default="0.80,0.85,0.90,0.92,0.94")
    parser.add_argument("--early-stop", choices=("max", "first"), default="max")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=200)
    parser.add_argument("--grad-norm-every", type=int, default=0)
    parser.add_argument(
        "--step-rule",
        choices=("none", "eoss", "l0l1", "sps", "sps-momentum", "adaptive-backtracking", "sagd", "silver"),
        default="none",
    )
    parser.add_argument("--step-eoss-beta", type=float, default=None)
    parser.add_argument("--step-eoss-ema", type=float, default=0.9)
    parser.add_argument("--step-eoss-interval", type=int, default=10)
    parser.add_argument("--step-eoss-eps", type=float, default=1e-8)
    parser.add_argument("--step-eoss-clip-min", type=float, default=1e-5)
    parser.add_argument("--step-eoss-clip-max", type=float, default=1.0)
    parser.add_argument("--step-l0", type=float, default=1.0)
    parser.add_argument("--step-l1", type=float, default=0.0)
    parser.add_argument("--step-fstar", type=float, default=0.0)
    parser.add_argument("--step-sps-beta", type=float, default=0.9)
    parser.add_argument("--step-sps-c", type=float, default=1.0)
    parser.add_argument("--step-sps-max", type=float, default=None)
    parser.add_argument("--step-backtrack-c", type=float, default=0.1)
    parser.add_argument("--step-backtrack-max", type=int, default=10)
    parser.add_argument("--step-backtrack-rho", type=float, default=0.5)
    parser.add_argument("--step-silver-rho", type=float, default=2.414213562373095)
    parser.add_argument("--step-sagd-delta", type=float, default=1e-2)
    parser.add_argument(
        "--direction",
        choices=("none", "diag-precond", "gn-layerwise", "gn-layerwise-exact", "shampoo", "soap", "sophia", "muon"),
        default="none",
    )
    parser.add_argument("--direction-beta", type=float, default=0.9)
    parser.add_argument("--direction-beta1", type=float, default=0.9)
    parser.add_argument("--direction-eps", type=float, default=1e-8)
    parser.add_argument("--direction-damping", type=float, default=1e-5)
    parser.add_argument("--direction-update-every", type=int, default=1)
    parser.add_argument("--direction-max-size", type=int, default=0)
    parser.add_argument("--gn-cg-iters", type=int, default=10)
    parser.add_argument("--gn-cg-tol", type=float, default=1e-4)
    parser.add_argument("--gn-layer-mode", choices=GN_LAYER_CHOICES, default="all")
    parser.add_argument("--gn-layer-k", type=int, default=0)
    parser.add_argument("--gn-update-interval", type=int, default=1)
    parser.add_argument(
        "--gn-layer-random-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reselect GN layers every step for randomk mode.",
    )
    parser.add_argument("--sophia-beta1", type=float, default=0.96)
    parser.add_argument("--sophia-beta2", type=float, default=0.99)
    parser.add_argument("--sophia-gamma", "--sophia-rho", dest="sophia_gamma", type=float, default=0.01)
    parser.add_argument("--sophia-eps", type=float, default=1e-12)
    parser.add_argument("--sophia-hessian-every", type=int, default=10)
    parser.add_argument("--sophia-hutchinson-samples", type=int, default=1)
    parser.add_argument("--muon-beta", type=float, default=0.95)
    parser.add_argument("--muon-eps", type=float, default=1e-8)
    parser.add_argument("--muon-ns-iters", type=int, default=5)
    parser.add_argument("--muon-scale-mode", choices=MUON_SCALE_CHOICES, default="adjusted-lr")
    parser.add_argument("--muon-rms-scale", type=float, default=0.2)
    parser.add_argument("--muon-hidden-size", type=int, default=0)
    parser.add_argument(
        "--clip-mode",
        choices=("none", "ggnc", "ggnc-global", "ggnc-layerwise", "global", "layerwise"),
        default="none",
    )
    parser.add_argument("--clip-rho", type=float, default=1.0)
    parser.add_argument("--clip-alpha", type=float, default=1.0)
    parser.add_argument("--sparsity", choices=("none", "linbreg"), default="none")
    parser.add_argument("--sparsity-lambda", type=float, default=0.0)
    parser.add_argument("--sparsity-update-interval", type=int, default=1)
    parser.add_argument("--anderson-memory", type=int, default=0)
    parser.add_argument("--anderson-interval", type=int, default=0)
    parser.add_argument("--anderson-damping", type=float, default=0.5)
    parser.add_argument("--anderson-lambda", type=float, default=1e-4)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--diagnostics", action="store_true")
    return parser


def parse_args() -> tuple[argparse.Namespace, Dict[str, Any]]:
    parser = build_parser()
    args = parser.parse_args()
    defaults: Dict[str, Any] = {}
    for action in parser._actions:
        if action.dest:
            defaults[action.dest] = action.default
    return args, defaults


def parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


def parse_targets(raw: str) -> List[float]:
    targets = [float(v.strip()) for v in raw.split(",") if v.strip()]
    targets = sorted(set(targets))
    return targets


def _set_if_default(args: argparse.Namespace, defaults: Dict[str, Any], key: str, value: Any) -> None:
    if key not in defaults:
        return
    if getattr(args, key) == defaults[key]:
        setattr(args, key, value)


def apply_config(args: argparse.Namespace, defaults: Dict[str, Any], config: Dict[str, Any]) -> None:
    run_cfg = config.get("run", {})
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    train = config.get("train", {})
    logging_cfg = config.get("logging", {})
    modules = config.get("modules", {})

    if "run_id" in run_cfg:
        _set_if_default(args, defaults, "run_id", run_cfg["run_id"])
    if "output_root" in run_cfg:
        _set_if_default(args, defaults, "output_root", run_cfg["output_root"])

    if dataset:
        if "data_dir" in dataset:
            _set_if_default(args, defaults, "data_dir", dataset["data_dir"])
        if "val_size" in dataset:
            _set_if_default(args, defaults, "val_size", dataset["val_size"])
        if "batch_size" in dataset:
            _set_if_default(args, defaults, "batch_size", dataset["batch_size"])
        if "num_workers" in dataset:
            _set_if_default(args, defaults, "num_workers", dataset["num_workers"])
        if "seed" in dataset:
            _set_if_default(args, defaults, "data_seed", dataset["seed"])
        if "download" in dataset:
            _set_if_default(args, defaults, "download", dataset["download"])

    if model and "name" in model:
        _set_if_default(args, defaults, "model", model["name"])

    if optimizer:
        if "type" in optimizer:
            _set_if_default(args, defaults, "optimizer", optimizer["type"])
        if "lr" in optimizer:
            _set_if_default(args, defaults, "lr", optimizer["lr"])
        if "momentum" in optimizer:
            _set_if_default(args, defaults, "momentum", optimizer["momentum"])
        if "weight_decay" in optimizer:
            _set_if_default(args, defaults, "weight_decay", optimizer["weight_decay"])

    if train:
        if "epochs" in train:
            _set_if_default(args, defaults, "epochs", train["epochs"])
        if "max_steps" in train:
            _set_if_default(args, defaults, "max_steps", train["max_steps"])
        if "deterministic" in train:
            _set_if_default(args, defaults, "deterministic", train["deterministic"])
        if "device" in train:
            _set_if_default(args, defaults, "device", train["device"])
        if "seeds" in train:
            _set_if_default(args, defaults, "seeds", ",".join(str(v) for v in train["seeds"]))
        if "diagnostics" in train:
            _set_if_default(args, defaults, "diagnostics", train["diagnostics"])

    if logging_cfg:
        if "log_interval_steps" in logging_cfg:
            _set_if_default(args, defaults, "log_interval_steps", logging_cfg["log_interval_steps"])
        if "eval_interval_epochs" in logging_cfg:
            _set_if_default(args, defaults, "eval_interval_epochs", logging_cfg["eval_interval_epochs"])
        if "eval_interval_steps" in logging_cfg:
            _set_if_default(args, defaults, "eval_interval_steps", logging_cfg["eval_interval_steps"])
        if "warmup_steps" in logging_cfg:
            _set_if_default(args, defaults, "warmup_steps", logging_cfg["warmup_steps"])
        if "measure_steps" in logging_cfg:
            _set_if_default(args, defaults, "measure_steps", logging_cfg["measure_steps"])
        if "grad_norm_every" in logging_cfg:
            _set_if_default(args, defaults, "grad_norm_every", logging_cfg["grad_norm_every"])

    if "targets" in config:
        _set_if_default(args, defaults, "target_acc", ",".join(str(v) for v in config["targets"]))
    if "early_stop" in config:
        _set_if_default(args, defaults, "early_stop", config["early_stop"])

    step_control = modules.get("step_control", {})
    if step_control:
        if "name" in step_control:
            _set_if_default(args, defaults, "step_rule", step_control["name"])
        if "beta" in step_control:
            _set_if_default(args, defaults, "step_eoss_beta", step_control["beta"])
        if "ema" in step_control:
            _set_if_default(args, defaults, "step_eoss_ema", step_control["ema"])
        if "interval" in step_control:
            _set_if_default(args, defaults, "step_eoss_interval", step_control["interval"])
        if "eps" in step_control:
            _set_if_default(args, defaults, "step_eoss_eps", step_control["eps"])
        if "clip_min" in step_control:
            _set_if_default(args, defaults, "step_eoss_clip_min", step_control["clip_min"])
        if "clip_max" in step_control:
            _set_if_default(args, defaults, "step_eoss_clip_max", step_control["clip_max"])
        if "l0" in step_control:
            _set_if_default(args, defaults, "step_l0", step_control["l0"])
        if "l1" in step_control:
            _set_if_default(args, defaults, "step_l1", step_control["l1"])
        if "fstar" in step_control:
            _set_if_default(args, defaults, "step_fstar", step_control["fstar"])
        if "sps_beta" in step_control:
            _set_if_default(args, defaults, "step_sps_beta", step_control["sps_beta"])
        if "sps_c" in step_control:
            _set_if_default(args, defaults, "step_sps_c", step_control["sps_c"])
        if "sps_max" in step_control:
            _set_if_default(args, defaults, "step_sps_max", step_control["sps_max"])
        if "backtrack_c" in step_control:
            _set_if_default(args, defaults, "step_backtrack_c", step_control["backtrack_c"])
        if "backtrack_max" in step_control:
            _set_if_default(args, defaults, "step_backtrack_max", step_control["backtrack_max"])
        if "backtrack_rho" in step_control:
            _set_if_default(args, defaults, "step_backtrack_rho", step_control["backtrack_rho"])
        if "silver_rho" in step_control:
            _set_if_default(args, defaults, "step_silver_rho", step_control["silver_rho"])
        if "sagd_delta" in step_control:
            _set_if_default(args, defaults, "step_sagd_delta", step_control["sagd_delta"])

    direction = modules.get("direction", {})
    if direction:
        if "name" in direction:
            _set_if_default(args, defaults, "direction", direction["name"])
        if "beta" in direction:
            _set_if_default(args, defaults, "direction_beta", direction["beta"])
        if "beta1" in direction:
            _set_if_default(args, defaults, "direction_beta1", direction["beta1"])
        if "eps" in direction:
            _set_if_default(args, defaults, "direction_eps", direction["eps"])
        if "damping" in direction:
            _set_if_default(args, defaults, "direction_damping", direction["damping"])
        if "update_every" in direction:
            _set_if_default(args, defaults, "direction_update_every", direction["update_every"])
        if "max_size" in direction:
            _set_if_default(args, defaults, "direction_max_size", direction["max_size"])
        if "gn_cg_iters" in direction:
            _set_if_default(args, defaults, "gn_cg_iters", direction["gn_cg_iters"])
        if "gn_cg_tol" in direction:
            _set_if_default(args, defaults, "gn_cg_tol", direction["gn_cg_tol"])
        if "gn_layer_mode" in direction:
            _set_if_default(args, defaults, "gn_layer_mode", direction["gn_layer_mode"])
        if "gn_layer_k" in direction:
            _set_if_default(args, defaults, "gn_layer_k", direction["gn_layer_k"])
        if "gn_update_interval" in direction:
            _set_if_default(args, defaults, "gn_update_interval", direction["gn_update_interval"])
        if "gn_layer_random_every_step" in direction:
            _set_if_default(args, defaults, "gn_layer_random_every_step", direction["gn_layer_random_every_step"])
        if "sophia_beta1" in direction:
            _set_if_default(args, defaults, "sophia_beta1", direction["sophia_beta1"])
        if "sophia_beta2" in direction:
            _set_if_default(args, defaults, "sophia_beta2", direction["sophia_beta2"])
        if "sophia_gamma" in direction:
            _set_if_default(args, defaults, "sophia_gamma", direction["sophia_gamma"])
        if "sophia_rho" in direction:
            _set_if_default(args, defaults, "sophia_gamma", direction["sophia_rho"])
        if "sophia_eps" in direction:
            _set_if_default(args, defaults, "sophia_eps", direction["sophia_eps"])
        if "sophia_hessian_every" in direction:
            _set_if_default(args, defaults, "sophia_hessian_every", direction["sophia_hessian_every"])
        if "sophia_hutchinson_samples" in direction:
            _set_if_default(args, defaults, "sophia_hutchinson_samples", direction["sophia_hutchinson_samples"])
        if "muon_beta" in direction:
            _set_if_default(args, defaults, "muon_beta", direction["muon_beta"])
        if "muon_eps" in direction:
            _set_if_default(args, defaults, "muon_eps", direction["muon_eps"])
        if "muon_ns_iters" in direction:
            _set_if_default(args, defaults, "muon_ns_iters", direction["muon_ns_iters"])
        if "muon_scale_mode" in direction:
            _set_if_default(args, defaults, "muon_scale_mode", direction["muon_scale_mode"])
        if "muon_rms_scale" in direction:
            _set_if_default(args, defaults, "muon_rms_scale", direction["muon_rms_scale"])
        if "muon_hidden_size" in direction:
            _set_if_default(args, defaults, "muon_hidden_size", direction["muon_hidden_size"])

    clip = modules.get("clip", {})
    if clip:
        if "mode" in clip:
            _set_if_default(args, defaults, "clip_mode", clip["mode"])
        if "rho" in clip:
            _set_if_default(args, defaults, "clip_rho", clip["rho"])
        if "alpha" in clip:
            _set_if_default(args, defaults, "clip_alpha", clip["alpha"])

    sparsity = modules.get("sparsity", {})
    if sparsity:
        if "name" in sparsity:
            _set_if_default(args, defaults, "sparsity", sparsity["name"])
        if "lambda" in sparsity:
            _set_if_default(args, defaults, "sparsity_lambda", sparsity["lambda"])
        if "update_interval" in sparsity:
            _set_if_default(args, defaults, "sparsity_update_interval", sparsity["update_interval"])

    outer = modules.get("outer", {})
    if outer:
        if "memory" in outer:
            _set_if_default(args, defaults, "anderson_memory", outer["memory"])
        if "interval" in outer:
            _set_if_default(args, defaults, "anderson_interval", outer["interval"])
        if "damping" in outer:
            _set_if_default(args, defaults, "anderson_damping", outer["damping"])
        if "lambda" in outer:
            _set_if_default(args, defaults, "anderson_lambda", outer["lambda"])


def setup_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(device_str)
    raise ValueError(f"unsupported device: {device_str}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_id_default(args: argparse.Namespace) -> str:
    date = datetime.now().strftime("%Y%m%d")
    time_tag = datetime.now().strftime("%H%M%S")
    return f"{date}-grad-speedup-cifar10-{args.model}-{args.optimizer}-{time_tag}"


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def log_step(path: Path, log: StepLog) -> None:
    payload = {
        "type": "step",
        "split": "train",
        "epoch": log.epoch,
        "step_in_epoch": log.step_in_epoch,
        "global_step": log.global_step,
        "loss": log.loss,
        "accuracy": log.accuracy,
        "lr": log.lr,
        "step_size": log.step_size,
        "grad_norm": log.grad_norm,
        "curvature": log.curvature,
        "step_time_ms": log.step_time_ms,
        "line_search_iters": log.line_search_iters,
        "line_search_accepted": log.line_search_accepted,
        "gn_selected_count": log.gn_selected_count,
        "gn_selected_layers": log.gn_selected_layers,
        "gn_update_time_ms": log.gn_update_time_ms,
        "gn_apply_time_ms": log.gn_apply_time_ms,
    }
    append_jsonl(path, payload)


def log_epoch(path: Path, split: str, epoch: int, global_step: int, metrics: TrainMetrics | EvalMetrics) -> None:
    payload = {
        "type": "epoch",
        "split": split,
        "epoch": epoch,
        "global_step": global_step,
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "samples": metrics.samples,
    }
    if isinstance(metrics, TrainMetrics):
        payload.update(
            {
                "step_time_ms": metrics.step_time_ms,
                "throughput": metrics.throughput,
                "steps": metrics.steps,
                "step_size_mean": metrics.step_size_mean,
                "step_size_p50": metrics.step_size_p50,
                "step_size_p90": metrics.step_size_p90,
                "grad_norm_mean": metrics.grad_norm_mean,
                "grad_norm_p50": metrics.grad_norm_p50,
                "grad_norm_p90": metrics.grad_norm_p90,
                "curvature_mean": metrics.curvature_mean,
                "curvature_p50": metrics.curvature_p50,
                "curvature_p90": metrics.curvature_p90,
                "direction_scale_mean": metrics.direction_scale_mean,
                "direction_scale_p50": metrics.direction_scale_p50,
                "direction_scale_p90": metrics.direction_scale_p90,
                "clip_coef_mean": metrics.clip_coef_mean,
                "clip_coef_p50": metrics.clip_coef_p50,
                "clip_coef_p90": metrics.clip_coef_p90,
                "sophia_hessian_mean": metrics.sophia_hessian_mean,
                "sophia_hessian_p50": metrics.sophia_hessian_p50,
                "sophia_hessian_p90": metrics.sophia_hessian_p90,
                "sophia_clip_frac_mean": metrics.sophia_clip_frac_mean,
                "sophia_clip_frac_p50": metrics.sophia_clip_frac_p50,
                "sophia_clip_frac_p90": metrics.sophia_clip_frac_p90,
                "muon_ortho_iters_mean": metrics.muon_ortho_iters_mean,
                "muon_ortho_iters_p50": metrics.muon_ortho_iters_p50,
                "muon_ortho_iters_p90": metrics.muon_ortho_iters_p90,
                "line_search_attempts": metrics.line_search_attempts,
                "line_search_accepted": metrics.line_search_accepted,
                "line_search_rejected": metrics.line_search_rejected,
                "line_search_iters_mean": metrics.line_search_iters_mean,
                "line_search_iters_p50": metrics.line_search_iters_p50,
                "line_search_iters_p90": metrics.line_search_iters_p90,
                "precond_update_count": metrics.precond_update_count,
                "precond_apply_count": metrics.precond_apply_count,
                "precond_update_time_s": metrics.precond_update_time_s,
                "precond_apply_time_s": metrics.precond_apply_time_s,
                "precond_layer_stats": metrics.precond_layer_stats,
                "gn_update_count": metrics.gn_update_count,
                "gn_apply_count": metrics.gn_apply_count,
                "gn_update_time_s": metrics.gn_update_time_s,
                "gn_apply_time_s": metrics.gn_apply_time_s,
                "gn_layer_stats": metrics.gn_layer_stats,
                "anderson_applied": metrics.anderson_applied,
                "anderson_failed": metrics.anderson_failed,
                "data_wait_time_s": metrics.data_wait_time_s,
                "max_memory_bytes": metrics.max_memory_bytes,
                "sparsity_fraction": metrics.sparsity_fraction,
                "dense_flops": metrics.dense_flops,
                "effective_flops": metrics.effective_flops,
                "sparsity_updates": metrics.sparsity_updates,
                "sparsity_update_interval": metrics.sparsity_update_interval,
                "sparsity_update_rate": metrics.sparsity_update_rate,
            }
        )
    append_jsonl(path, payload)


def summarize_targets(
    targets: List[float],
    hits: Dict[float, Optional[Dict[str, float]]],
    mean_step_time_sec: Optional[float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for target in targets:
        hit = hits.get(target)
        if hit is None:
            summary[str(target)] = None
            continue
        cost_to_target = None
        if mean_step_time_sec is not None:
            cost_to_target = hit["steps"] * mean_step_time_sec
        summary[str(target)] = {
            "steps_to_target": hit["steps"],
            "time_to_target_sec": hit["time_sec"],
            "cost_to_target_sec": cost_to_target,
            "accuracy": hit["accuracy"],
            "epoch": hit["epoch"],
        }
    return summary


def aggregate_values(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "count": 1}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
        "count": len(values),
    }


def main() -> int:
    args, defaults = parse_args()
    if args.config:
        config_path = Path(args.config)
        config = json.loads(config_path.read_text())
        apply_config(args, defaults, config)

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    if args.step_eoss_beta is None:
        args.step_eoss_beta = args.lr

    device = setup_device(args.device)
    if args.deterministic and device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seeds = parse_seeds(args)
    targets = parse_targets(args.target_acc)
    run_id = args.run_id or run_id_default(args)

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "deterministic": args.deterministic,
    }

    config = {
        "run_id": run_id,
        "config_path": args.config or None,
        "model": args.model,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seeds": seeds,
        "data_seed": args.data_seed,
        "val_size": args.val_size,
        "num_workers": args.num_workers,
        "num_threads": args.num_threads,
        "device": str(device),
        "deterministic": args.deterministic,
        "output_root": str(output_root),
        "data_dir": args.data_dir,
        "log_interval_steps": args.log_interval_steps,
        "eval_interval_epochs": args.eval_interval_epochs,
        "eval_interval_steps": args.eval_interval_steps,
        "targets": targets,
        "early_stop": args.early_stop,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "grad_norm_every": args.grad_norm_every,
        "step_rule": args.step_rule,
        "step_eoss_beta": args.step_eoss_beta,
        "step_eoss_ema": args.step_eoss_ema,
        "step_eoss_interval": args.step_eoss_interval,
        "step_eoss_eps": args.step_eoss_eps,
        "step_eoss_clip_min": args.step_eoss_clip_min,
        "step_eoss_clip_max": args.step_eoss_clip_max,
        "step_l0": args.step_l0,
        "step_l1": args.step_l1,
        "step_fstar": args.step_fstar,
        "step_sps_beta": args.step_sps_beta,
        "step_sps_c": args.step_sps_c,
        "step_sps_max": args.step_sps_max,
        "step_backtrack_c": args.step_backtrack_c,
        "step_backtrack_max": args.step_backtrack_max,
        "step_backtrack_rho": args.step_backtrack_rho,
        "step_silver_rho": args.step_silver_rho,
        "direction": args.direction,
        "direction_beta": args.direction_beta,
        "direction_beta1": args.direction_beta1,
        "direction_eps": args.direction_eps,
        "direction_damping": args.direction_damping,
        "direction_update_every": args.direction_update_every,
        "direction_max_size": args.direction_max_size,
        "gn_cg_iters": args.gn_cg_iters,
        "gn_cg_tol": args.gn_cg_tol,
        "gn_layer_mode": args.gn_layer_mode,
        "gn_layer_k": args.gn_layer_k,
        "gn_update_interval": args.gn_update_interval,
        "gn_layer_random_every_step": args.gn_layer_random_every_step,
        "sophia_beta1": args.sophia_beta1,
        "sophia_beta2": args.sophia_beta2,
        "sophia_gamma": args.sophia_gamma,
        "sophia_eps": args.sophia_eps,
        "sophia_hessian_every": args.sophia_hessian_every,
        "sophia_hutchinson_samples": args.sophia_hutchinson_samples,
        "muon_beta": args.muon_beta,
        "muon_eps": args.muon_eps,
        "muon_ns_iters": args.muon_ns_iters,
        "muon_scale_mode": args.muon_scale_mode,
        "muon_rms_scale": args.muon_rms_scale,
        "muon_hidden_size": args.muon_hidden_size,
        "clip_mode": args.clip_mode,
        "clip_rho": args.clip_rho,
        "clip_alpha": args.clip_alpha,
        "sparsity": args.sparsity,
        "sparsity_lambda": args.sparsity_lambda,
        "sparsity_update_interval": args.sparsity_update_interval,
        "anderson_memory": args.anderson_memory,
        "anderson_interval": args.anderson_interval,
        "anderson_damping": args.anderson_damping,
        "anderson_lambda": args.anderson_lambda,
        "diagnostics": args.diagnostics,
    }

    write_json(run_root / "config.json", config)
    write_json(run_root / "env.json", env)

    aggregated: Dict[str, Any] = {
        "run_id": run_id,
        "model": args.model,
        "optimizer": args.optimizer,
        "seeds": seeds,
        "targets": targets,
        "per_seed": [],
    }

    for seed in seeds:
        seed_dir = run_root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = seed_dir / "metrics.jsonl"

        set_seed(seed, deterministic=args.deterministic)

        data_config = DataConfig(
            data_dir=str((PROJECT_ROOT / args.data_dir).resolve()),
            batch_size=args.batch_size,
            val_size=args.val_size,
            num_workers=args.num_workers,
            seed=args.data_seed,
            download=args.download,
        )
        train_loader, val_loader, test_loader = get_cifar10_loaders(data_config)

        model = build_model(ModelConfig(name=args.model))
        model = model.to(device)
        optimizer = build_optimizer(args, model)

        global_step = 0
        run_start = time.perf_counter()
        anderson_state: list[tuple[torch.Tensor, torch.Tensor]] = []
        targets_hit: Dict[float, Optional[Dict[str, float]]] = {t: None for t in targets}
        max_target = max(targets) if targets else None
        last_eval_step: Optional[int] = None
        last_epoch = 0

        total_step_time_s = 0.0
        total_step_count = 0

        def _record_eval(step: int, eval_epoch: int) -> EvalMetrics:
            nonlocal last_eval_step
            was_training = model.training
            eval_metrics = evaluate(model, test_loader, device)
            if was_training:
                model.train()
            log_epoch(metrics_path, "test", eval_epoch, step, eval_metrics)
            last_eval_step = step

            elapsed = time.perf_counter() - run_start
            for target in targets:
                if targets_hit[target] is not None:
                    continue
                if eval_metrics.accuracy >= target:
                    targets_hit[target] = {
                        "steps": step,
                        "time_sec": float(elapsed),
                        "accuracy": float(eval_metrics.accuracy),
                        "epoch": eval_epoch,
                    }
            return eval_metrics

        def _on_step_end(step: int, step_epoch: int, _step_in_epoch: int) -> bool:
            if args.eval_interval_steps > 0 and step % args.eval_interval_steps == 0:
                if last_eval_step != step:
                    _record_eval(step, step_epoch)
            return args.max_steps > 0 and step >= args.max_steps

        for epoch in range(1, args.epochs + 1):
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            last_epoch = epoch
            epoch_start = time.perf_counter()

            def _log_fn(step_log: StepLog) -> None:
                log_step(metrics_path, step_log)

            train_metrics, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                global_step=global_step,
                log_interval=args.log_interval_steps,
                log_fn=_log_fn,
                on_step_end=_on_step_end,
                warmup_steps=args.warmup_steps,
                measure_steps=args.measure_steps,
                grad_norm_every=args.grad_norm_every,
                step_rule=args.step_rule,
                step_eoss_beta=args.step_eoss_beta,
                step_eoss_ema=args.step_eoss_ema,
                step_eoss_interval=args.step_eoss_interval,
                step_eoss_eps=args.step_eoss_eps,
                step_eoss_clip_min=args.step_eoss_clip_min,
                step_eoss_clip_max=args.step_eoss_clip_max,
                step_l0=args.step_l0,
                step_l1=args.step_l1,
                step_fstar=args.step_fstar,
                step_sps_beta=args.step_sps_beta,
                step_sps_c=args.step_sps_c,
                step_sps_max=args.step_sps_max,
                step_backtrack_c=args.step_backtrack_c,
                step_backtrack_max=args.step_backtrack_max,
                step_backtrack_rho=args.step_backtrack_rho,
                step_silver_rho=args.step_silver_rho,
                step_sagd_delta=args.step_sagd_delta,
                direction=args.direction,
                direction_beta=args.direction_beta,
                direction_eps=args.direction_eps,
                direction_beta1=args.direction_beta1,
                direction_damping=args.direction_damping,
                direction_update_every=args.direction_update_every,
                direction_max_size=args.direction_max_size,
                gn_cg_iters=args.gn_cg_iters,
                gn_cg_tol=args.gn_cg_tol,
                gn_layer_mode=args.gn_layer_mode,
                gn_layer_k=args.gn_layer_k,
                gn_update_interval=args.gn_update_interval,
                gn_layer_random_every_step=args.gn_layer_random_every_step,
                gn_layer_seed=seed,
                sophia_beta1=args.sophia_beta1,
                sophia_beta2=args.sophia_beta2,
                sophia_gamma=args.sophia_gamma,
                sophia_eps=args.sophia_eps,
                sophia_hessian_every=args.sophia_hessian_every,
                sophia_hutchinson_samples=args.sophia_hutchinson_samples,
                muon_beta=args.muon_beta,
                muon_eps=args.muon_eps,
                muon_ns_iters=args.muon_ns_iters,
                muon_scale_mode=args.muon_scale_mode,
                muon_rms_scale=args.muon_rms_scale,
                muon_hidden_size=args.muon_hidden_size,
                clip_mode=args.clip_mode,
                clip_rho=args.clip_rho,
                clip_alpha=args.clip_alpha,
                sparsity=args.sparsity,
                sparsity_lambda=args.sparsity_lambda,
                sparsity_update_interval=args.sparsity_update_interval,
                anderson_memory=args.anderson_memory,
                anderson_interval=args.anderson_interval,
                anderson_damping=args.anderson_damping,
                anderson_lambda=args.anderson_lambda,
                anderson_state=anderson_state,
                diagnostics=args.diagnostics,
            )
            log_epoch(metrics_path, "train", epoch, global_step, train_metrics)

            total_step_time_s += train_metrics.step_time_total_s
            total_step_count += train_metrics.step_time_count

            if args.eval_interval_epochs > 0 and epoch % args.eval_interval_epochs == 0:
                if last_eval_step != global_step:
                    _record_eval(global_step, epoch)

            epoch_time_sec = time.perf_counter() - epoch_start
            append_jsonl(
                metrics_path,
                {
                    "type": "epoch_timing",
                    "epoch": epoch,
                    "global_step": global_step,
                    "epoch_time_sec": float(epoch_time_sec),
                },
            )

            if args.early_stop == "first":
                if any(v is not None for v in targets_hit.values()):
                    break
            elif max_target is not None and targets_hit.get(max_target) is not None:
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if last_eval_step != global_step:
            _record_eval(global_step, last_epoch)

        mean_step_time_sec = None
        if total_step_count > 0:
            mean_step_time_sec = total_step_time_s / total_step_count

        summary = {
            "seed": seed,
            "gn_layer_seed": seed,
            "steps_per_epoch": len(train_loader),
            "mean_step_time_sec": mean_step_time_sec,
            "targets": summarize_targets(targets, targets_hit, mean_step_time_sec),
            "total_steps": global_step,
            "total_time_sec": float(time.perf_counter() - run_start),
        }
        write_json(seed_dir / "summary.json", summary)
        aggregated["per_seed"].append(summary)

    aggregated_targets: Dict[str, Any] = {}
    for target in targets:
        target_key = str(target)
        steps_vals: List[float] = []
        time_vals: List[float] = []
        cost_vals: List[float] = []
        for seed_summary in aggregated["per_seed"]:
            hit = seed_summary["targets"].get(target_key)
            if hit is None:
                continue
            steps_vals.append(hit["steps_to_target"])
            time_vals.append(hit["time_to_target_sec"])
            if hit["cost_to_target_sec"] is not None:
                cost_vals.append(hit["cost_to_target_sec"])
        aggregated_targets[target_key] = {
            "steps_to_target": aggregate_values(steps_vals),
            "time_to_target_sec": aggregate_values(time_vals),
            "cost_to_target_sec": aggregate_values(cost_vals),
        }

    aggregated["targets"] = aggregated_targets
    write_json(run_root / "summary.json", aggregated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
