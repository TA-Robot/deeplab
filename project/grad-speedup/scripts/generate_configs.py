from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grad-speedup configs")
    parser.add_argument("--out-dir", type=str, default="configs")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def write_config(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def base_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "run": {"output_root": "../runs/grad-speedup"},
        "dataset": {
            "name": "cifar10",
            "data_dir": args.data_dir,
            "val_size": args.val_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": 123,
            "download": args.download,
        },
        "model": {"name": args.model},
        "train": {
            "epochs": args.epochs,
            "deterministic": args.deterministic,
            "device": args.device,
            "seeds": [0, 1, 2],
        },
        "logging": {
            "log_interval_steps": 100,
            "eval_interval_epochs": 1,
            "warmup_steps": 50,
            "measure_steps": 200,
            "grad_norm_every": 0,
        },
        "targets": [0.85, 0.90, 0.92, 0.94],
        "early_stop": "max",
        "modules": {
            "step_control": {"name": "none"},
            "direction": {"name": "none"},
            "clip": {"mode": "none"},
            "outer": {"name": "none"},
        },
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    configs: List[tuple[str, Dict[str, Any]]] = []

    # Baseline SGD
    sgd = base_config(args)
    sgd["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
    }
    configs.append(("baseline-sgd.json", sgd))

    # Baseline Adam
    adam = base_config(args)
    adam["optimizer"] = {
        "type": "adam",
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 1e-4,
    }
    configs.append(("baseline-adam.json", adam))

    # Module C: L0L1-GD on SGD
    l0l1 = base_config(args)
    l0l1["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    l0l1["modules"]["step_control"] = {"name": "l0l1", "l0": 1.0, "l1": 0.1}
    configs.append(("modc-l0l1-sgd.json", l0l1))

    # Module C: SPS on SGD
    sps = base_config(args)
    sps["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    sps["modules"]["step_control"] = {"name": "sps", "fstar": 0.0}
    configs.append(("modc-sps-sgd.json", sps))

    # Module C: SPS + momentum on SGD
    sps_momentum = base_config(args)
    sps_momentum["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    sps_momentum["modules"]["step_control"] = {
        "name": "sps-momentum",
        "fstar": 0.0,
        "sps_beta": 0.9,
        "sps_c": 1.0,
    }
    configs.append(("modc-sps-momentum-sgd.json", sps_momentum))

    # Module C: adaptive backtracking on SGD
    backtrack = base_config(args)
    backtrack["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    backtrack["modules"]["step_control"] = {
        "name": "adaptive-backtracking",
        "backtrack_c": 0.1,
        "backtrack_max": 10,
        "backtrack_rho": 0.5,
    }
    configs.append(("modc-adaptive-backtracking-sgd.json", backtrack))

    # Module C: stochastic adaptive GD without descent on SGD
    sagd = base_config(args)
    sagd["optimizer"] = {
        "type": "sgd",
        "lr": 1e-3,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    sagd["modules"]["step_control"] = {"name": "sagd", "sagd_delta": 1e-2}
    configs.append(("modc-sagd-sgd.json", sagd))

    # Module C: silver stepsizes on SGD
    silver = base_config(args)
    silver["optimizer"] = {
        "type": "sgd",
        "lr": 0.1,
        "momentum": 0.0,
        "weight_decay": 5e-4,
    }
    silver["modules"]["step_control"] = {"name": "silver", "silver_rho": 2.414213562373095}
    configs.append(("modc-silver-sgd.json", silver))

    for name, cfg in configs:
        write_config(out_dir / name, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
