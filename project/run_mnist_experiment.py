from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime
from pathlib import Path

import torch
import torchvision

from src.data import DataConfig, get_dataset_loaders
from src.models import CNNClassifier, CNNWithOBL, MLPClassifier, MLPWithOBL
from src.operator_basis import OperatorBasisLayer, build_obl_config
from src.train import count_parameters, evaluate, set_seed, train_one_epoch


MODEL_CHOICES = ("mlp", "cnn", "mlp-obl", "cnn-obl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset experiments for ROS-ALTH operator basis layer")
    parser.add_argument("--dataset", choices=("mnist", "fashion-mnist", "fashion_mnist", "cifar10"), default="mnist")
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--beta-l1", type=float, default=0.0)
    parser.add_argument("--operator-dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta-init", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cnn-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-hidden-layers", type=int, default=2)
    parser.add_argument("--mlp-hidden-dims", type=str, default="")
    parser.add_argument("--cnn-fc-layers", type=int, default=2)
    parser.add_argument("--cnn-fc-dims", type=str, default="")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--obl-profile", choices=("mini", "full"), default="full")
    parser.add_argument("--obl-seed", type=int, default=-1)
    parser.add_argument("--obl-norm", choices=("layernorm", "rmsnorm"), default="layernorm")
    parser.add_argument("--obl-programs", type=int, default=-1)
    return parser.parse_args()

def _parse_dims(raw: str, fallback: int, count: int) -> list[int]:
    if raw:
        return [int(v.strip()) for v in raw.split(",") if v.strip()]
    return [fallback] * count


def _build_obl_config(input_dim: int, args: argparse.Namespace) -> object:
    seed = None if args.obl_seed < 0 else args.obl_seed
    overrides: dict[str, object] = {
        "gamma": args.gamma,
        "beta_init": args.beta_init,
        "operator_dropout": args.operator_dropout,
        "norm_type": args.obl_norm,
        "seed": seed,
    }
    if args.obl_programs >= 0:
        overrides["num_programs"] = args.obl_programs
    return build_obl_config(input_dim, profile=args.obl_profile, **overrides)


def build_model(args: argparse.Namespace, input_dim: int, input_shape: tuple[int, int, int]) -> torch.nn.Module:
    mlp_dims = _parse_dims(args.mlp_hidden_dims, args.hidden_dim, args.mlp_hidden_layers)
    cnn_dims = _parse_dims(args.cnn_fc_dims, args.cnn_hidden_dim, args.cnn_fc_layers)

    if args.model == "mlp":
        return MLPClassifier(input_dim=input_dim, hidden_dims=mlp_dims)
    if args.model == "cnn":
        return CNNClassifier(input_shape=input_shape, fc_dims=cnn_dims)

    if args.model == "mlp-obl":
        obl_config = _build_obl_config(mlp_dims[0], args)
        return MLPWithOBL(input_dim=input_dim, hidden_dims=mlp_dims, obl_config=obl_config)
    if args.model == "cnn-obl":
        obl_config = _build_obl_config(cnn_dims[0], args)
        return CNNWithOBL(input_shape=input_shape, fc_dims=cnn_dims, obl_config=obl_config)

    raise ValueError(f"unsupported model: {args.model}")


def get_beta_l1(model: torch.nn.Module) -> torch.Tensor:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, OperatorBasisLayer):
            total = total + module.beta_l1()
    return total


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


def setup_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(device_str)
    raise ValueError(f"unsupported device: {device_str}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = setup_device(args.device)
    seeds = parse_seeds(args)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{timestamp}-{args.model}"

    project_root = Path(__file__).resolve().parent
    output_root = (project_root / args.output_root).resolve()
    run_dir = output_root / run_id

    data_config = DataConfig(
        data_dir=str((project_root / args.data_dir).resolve()),
        batch_size=args.batch_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        seed=args.data_seed,
        dataset=args.dataset,
    )

    torch_config = None
    config_buffer = StringIO()
    with redirect_stdout(config_buffer):
        torch.__config__.show()
    torch_config = config_buffer.getvalue()

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_config": torch_config.strip(),
        "torchvision": torchvision.__version__,
        "device": str(device),
        "num_threads": torch.get_num_threads(),
        "mkldnn": torch.backends.mkldnn.enabled,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "deterministic": args.deterministic,
    }

    config = {
        "run_id": run_id,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "seeds": seeds,
        "data_seed": args.data_seed,
        "val_size": args.val_size,
        "num_workers": args.num_workers,
        "num_threads": args.num_threads,
        "device": str(device),
        "beta_l1": args.beta_l1,
        "operator_dropout": args.operator_dropout,
        "gamma": args.gamma,
        "beta_init": args.beta_init,
        "hidden_dim": args.hidden_dim,
        "cnn_hidden_dim": args.cnn_hidden_dim,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "mlp_hidden_dims": args.mlp_hidden_dims,
        "cnn_fc_layers": args.cnn_fc_layers,
        "cnn_fc_dims": args.cnn_fc_dims,
        "obl_profile": args.obl_profile,
        "obl_seed": args.obl_seed,
        "obl_norm": args.obl_norm,
        "obl_programs": args.obl_programs,
        "dataset": args.dataset,
        "data_dir": data_config.data_dir,
    }

    write_json(run_dir / "config.json", config)
    write_json(run_dir / "env.json", env)

    summary = {"run_id": run_id, "model": args.model, "seeds": []}
    test_accuracies: list[float] = []
    test_losses: list[float] = []
    final_step_times: list[float] = []
    final_throughputs: list[float] = []
    wall_times: list[float] = []

    metrics_path = run_dir / "metrics.jsonl"

    for seed in seeds:
        set_seed(seed, deterministic=args.deterministic)
        train_loader, val_loader, test_loader, data_info = get_dataset_loaders(data_config)

        input_dim = int(data_info["channels"] * data_info["height"] * data_info["width"])
        input_shape = (int(data_info["channels"]), int(data_info["height"]), int(data_info["width"]))
        model = build_model(args, input_dim=input_dim, input_shape=input_shape).to(device)
        trainable_params = count_parameters(model)
        total_params = count_parameters(model, include_buffers=True)

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
            )

        best_val = 0.0
        seed_start = time.perf_counter()
        last_train_metrics = None
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                beta_l1_weight=args.beta_l1,
                beta_l1_fn=get_beta_l1 if args.beta_l1 > 0 else None,
            )
            last_train_metrics = train_metrics
            val_metrics = evaluate(model, val_loader, device)
            best_val = max(best_val, val_metrics.accuracy)
            epoch_end = time.perf_counter()
            epoch_time_sec = epoch_end - epoch_start

            append_jsonl(
                metrics_path,
                {
                    "seed": seed,
                    "epoch": epoch,
                    "split": "train",
                    "loss": train_metrics.loss,
                    "accuracy": train_metrics.accuracy,
                    "step_time_ms": train_metrics.step_time_ms,
                    "throughput": train_metrics.throughput,
                    "epoch_time_sec": epoch_time_sec,
                    "samples": train_metrics.samples,
                },
            )
            append_jsonl(
                metrics_path,
                {
                    "seed": seed,
                    "epoch": epoch,
                    "split": "val",
                    "loss": val_metrics.loss,
                    "accuracy": val_metrics.accuracy,
                    "samples": val_metrics.samples,
                },
            )

        test_metrics = evaluate(model, test_loader, device)
        seed_end = time.perf_counter()
        seed_wall_time_sec = seed_end - seed_start
        append_jsonl(
            metrics_path,
            {
                "seed": seed,
                "epoch": args.epochs,
                "split": "test",
                "loss": test_metrics.loss,
                "accuracy": test_metrics.accuracy,
                "samples": test_metrics.samples,
            },
        )

        seed_summary = {
            "seed": seed,
            "trainable_param_count": trainable_params,
            "total_param_count": total_params,
            "best_val_accuracy": best_val,
            "test_loss": test_metrics.loss,
            "test_accuracy": test_metrics.accuracy,
            "wall_time_sec": seed_wall_time_sec,
            "final_train_step_time_ms": 0.0 if last_train_metrics is None else last_train_metrics.step_time_ms,
            "final_train_throughput": 0.0 if last_train_metrics is None else last_train_metrics.throughput,
            "data_info": data_info,
        }

        if args.save_model:
            torch.save(model.state_dict(), run_dir / f"model-seed{seed}.pt")

        summary["seeds"].append(seed_summary)
        test_accuracies.append(test_metrics.accuracy)
        test_losses.append(test_metrics.loss)
        if last_train_metrics is not None:
            final_step_times.append(last_train_metrics.step_time_ms)
            final_throughputs.append(last_train_metrics.throughput)
        wall_times.append(seed_wall_time_sec)

    def stats(values: list[float]) -> dict:
        if not values:
            return {}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std}

    summary["aggregate"] = {
        "test_accuracy": stats(test_accuracies),
        "test_loss": stats(test_losses),
        "final_train_step_time_ms": stats(final_step_times),
        "final_train_throughput": stats(final_throughputs),
        "wall_time_sec": stats(wall_times),
    }

    write_json(run_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
