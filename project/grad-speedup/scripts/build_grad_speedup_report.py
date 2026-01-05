from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grad-speedup aggregate report")
    parser.add_argument("--runs-dir", type=str, default="../runs/grad-speedup")
    parser.add_argument("--output", type=str, default="reports/grad-speedup-report.json")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_get(summary: Dict[str, Any], key: str, default: Any = None) -> Any:
    return summary.get(key, default)


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()
    output_path = Path(args.output).resolve()

    runs: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = read_json(summary_path)
            summary["run_dir"] = str(run_dir)
            runs.append(summary)

            run_id = _safe_get(summary, "run_id")
            model = _safe_get(summary, "model")
            optimizer = _safe_get(summary, "optimizer")
            per_seed = _safe_get(summary, "per_seed", [])
            for seed_summary in per_seed:
                seed = seed_summary.get("seed")
                mean_step_time = seed_summary.get("mean_step_time_sec")
                targets = seed_summary.get("targets", {})
                for target_key, target_data in targets.items():
                    if target_data is None:
                        row = {
                            "run_id": run_id,
                            "seed": seed,
                            "model": model,
                            "optimizer": optimizer,
                            "mean_step_time_sec": mean_step_time,
                            "target": target_key,
                            "steps_to_target": None,
                            "time_to_target_sec": None,
                            "cost_to_target_sec": None,
                        }
                    else:
                        row = {
                            "run_id": run_id,
                            "seed": seed,
                            "model": model,
                            "optimizer": optimizer,
                            "mean_step_time_sec": mean_step_time,
                            "target": target_key,
                            "steps_to_target": target_data.get("steps_to_target"),
                            "time_to_target_sec": target_data.get("time_to_target_sec"),
                            "cost_to_target_sec": target_data.get("cost_to_target_sec"),
                        }
                    csv_rows.append(row)

    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        "runs": runs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    csv_path = output_path.with_suffix(".csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
