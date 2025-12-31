#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def _list_dev_nodes() -> list[str]:
    return sorted(str(p) for p in Path("/dev").glob("nvidia*"))


def collect_report() -> dict:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict[str, object] = {
        "timestamp": timestamp,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "os": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PATH": os.environ.get("PATH"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        },
        "dev_nodes": _list_dev_nodes(),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvidia_smi_list": _run(["nvidia-smi", "-L"]),
    }

    try:
        import torch

        torch_info: dict[str, object] = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_version": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }
        if torch_info["cuda_available"]:
            names = []
            for idx in range(torch.cuda.device_count()):
                try:
                    names.append(torch.cuda.get_device_name(idx))
                except Exception as exc:  # noqa: BLE001
                    names.append(f"error: {exc!r}")
            torch_info["device_names"] = names
        report["torch"] = torch_info

        try:
            config_buf = []
            torch.__config__.show(buf=config_buf.append)
            report["torch_config"] = "".join(config_buf).strip()
        except Exception as exc:  # noqa: BLE001
            report["torch_config_error"] = repr(exc)
    except Exception as exc:  # noqa: BLE001
        report["torch_error"] = repr(exc)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GPU environment diagnostics")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Defaults to project/runs/diagnostics/<timestamp>-gpu-env.json",
    )
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print JSON to stdout")
    args = parser.parse_args()

    report = collect_report()
    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parent.parent / "runs" / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stamp = str(report["timestamp"]).replace("-", "").replace(":", "").replace("T", "-").replace("Z", "")
        output_path = out_dir / f"{safe_stamp}-gpu-env.json"

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    if args.print_report:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
