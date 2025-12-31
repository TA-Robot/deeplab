from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Stats:
    mean: float
    std: float


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compute_stats(values: list[float]) -> Stats | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return Stats(mean=mean, std=0.0)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return Stats(mean=mean, std=math.sqrt(variance))


def stats_dict(values: list[float]) -> dict:
    stats = compute_stats(values)
    if stats is None:
        return {}
    return {"mean": stats.mean, "std": stats.std}


def group_metrics(metrics: list[dict]) -> dict:
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in metrics:
        grouped[(row["epoch"], row["split"])].append(row)
    curves: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for (epoch, split), rows in sorted(grouped.items(), key=lambda item: item[0][0]):
        curves[split]["epoch"].append(epoch)
        for key in ("accuracy", "loss"):
            values = [r[key] for r in rows if key in r]
            stats = compute_stats(values)
            curves[split][key].append(stats.mean if stats else None)
            curves[split][f"{key}_std"].append(stats.std if stats else None)
        if split == "train":
            for key in ("step_time_ms", "throughput", "epoch_time_sec"):
                values = [r.get(key) for r in rows if key in r]
                values = [v for v in values if v is not None]
                stats = compute_stats(values)
                curves[split][key].append(stats.mean if stats else None)
                curves[split][f"{key}_std"].append(stats.std if stats else None)
    return curves


def epoch_to_threshold(curve: dict[str, list], threshold: float) -> int | None:
    epochs = curve.get("epoch", [])
    accuracies = curve.get("accuracy", [])
    for epoch, acc in zip(epochs, accuracies):
        if acc is not None and acc >= threshold:
            return epoch
    return None


def load_run(run_dir: Path) -> dict:
    config = read_json(run_dir / "config.json")
    env = read_json(run_dir / "env.json")
    summary = read_json(run_dir / "summary.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")

    curves = group_metrics(metrics)
    seed_summaries = summary.get("seeds", [])
    param_counts = []
    trainable_counts = []
    for item in seed_summaries:
        total = item.get("total_param_count") or item.get("param_count")
        trainable = item.get("trainable_param_count")
        if total is not None:
            param_counts.append(total)
        if trainable is not None:
            trainable_counts.append(trainable)
    wall_times = [s.get("wall_time_sec") for s in seed_summaries if s.get("wall_time_sec") is not None]

    data_info = None
    if seed_summaries:
        data_info = seed_summaries[0].get("data_info")

    val_curve = curves.get("val", {})
    epoch_97 = epoch_to_threshold(val_curve, 0.97)

    run = {
        "run_id": summary.get("run_id"),
        "model": summary.get("model"),
        "config": config,
        "env": env,
        "summary": summary,
        "curves": curves,
        "param_count": stats_dict([float(v) for v in param_counts if v is not None]),
        "trainable_param_count": stats_dict([float(v) for v in trainable_counts if v is not None]),
        "wall_time_sec": stats_dict([float(v) for v in wall_times if v is not None]),
        "epoch_to_97": epoch_97,
        "data_info": data_info,
    }
    run["derived"] = derive_metrics(run)
    return run


def compare_runs(base: dict, variant: dict) -> dict:
    base_acc = base["summary"].get("aggregate", {}).get("test_accuracy", {}).get("mean")
    var_acc = variant["summary"].get("aggregate", {}).get("test_accuracy", {}).get("mean")
    base_step = base["summary"].get("aggregate", {}).get("final_train_step_time_ms", {}).get("mean")
    var_step = variant["summary"].get("aggregate", {}).get("final_train_step_time_ms", {}).get("mean")
    base_wall = base["summary"].get("aggregate", {}).get("wall_time_sec", {}).get("mean")
    var_wall = variant["summary"].get("aggregate", {}).get("wall_time_sec", {}).get("mean")
    base_param = base.get("param_count", {}).get("mean")
    var_param = variant.get("param_count", {}).get("mean")

    def pct_change(a: float | None, b: float | None) -> float | None:
        if a is None or b is None or a == 0:
            return None
        return (b - a) / a * 100.0

    return {
        "test_accuracy_delta": None if base_acc is None or var_acc is None else (var_acc - base_acc),
        "step_time_delta_pct": pct_change(base_step, var_step),
        "wall_time_delta_pct": pct_change(base_wall, var_wall),
        "param_delta_pct": pct_change(base_param, var_param),
        "param_ratio": None if base_param is None or var_param is None or base_param == 0 else (var_param / base_param),
    }


def build_insights(runs: dict[str, dict]) -> list[str]:
    insights: list[str] = []
    for pair_name, (base_key, var_key) in {
        "MLP": ("mlp", "mlp-obl"),
        "CNN": ("cnn", "cnn-obl"),
    }.items():
        if base_key not in runs or var_key not in runs:
            continue
        cmp = compare_runs(runs[base_key], runs[var_key])
        acc_delta = cmp.get("test_accuracy_delta")
        step_delta = cmp.get("step_time_delta_pct")
        wall_delta = cmp.get("wall_time_delta_pct")
        param_delta = cmp.get("param_delta_pct")
        if acc_delta is not None:
            insights.append(
                f"{pair_name} accuracy delta (OBL - baseline): {acc_delta * 100:.2f} pp"
            )
        if step_delta is not None:
            insights.append(f"{pair_name} step time delta: {step_delta:.1f}%")
        if wall_delta is not None:
            insights.append(f"{pair_name} wall time delta: {wall_delta:.1f}%")
        if param_delta is not None:
            insights.append(f"{pair_name} param delta: {param_delta:.1f}%")

    for key, run in runs.items():
        derived = run.get("derived", {})
        gap = derived.get("generalization_gap")
        if gap is not None:
            insights.append(f"{key} generalization gap (train - test): {gap * 100:.2f} pp")
        test_std = derived.get("test_accuracy_std")
        if test_std is not None:
            insights.append(f"{key} test accuracy std: {test_std * 100:.2f} pp")

    best_key = None
    best_score = None
    for key, run in runs.items():
        acc = run["summary"].get("aggregate", {}).get("test_accuracy", {}).get("mean")
        wall = run["summary"].get("aggregate", {}).get("wall_time_sec", {}).get("mean")
        if acc is None or wall is None or wall <= 0:
            continue
        score = acc / wall
        if best_score is None or score > best_score:
            best_score = score
            best_key = key
    if best_key:
        insights.append(f"Best accuracy per wall-time: {best_key}")

    return insights


def build_analysis_sections(runs: dict[str, dict], comparisons: dict[str, dict], guardrail: float) -> list[dict]:
    sections: list[dict] = []

    accuracy_items = []
    for key, run in runs.items():
        acc = run.get("summary", {}).get("aggregate", {}).get("test_accuracy", {}).get("mean")
        std = run.get("summary", {}).get("aggregate", {}).get("test_accuracy", {}).get("std")
        if acc is None:
            accuracy_items.append(f"{key}: no test accuracy")
            continue
        status = "pass" if acc >= guardrail else "below"
        accuracy_items.append(
            f"{key}: {acc * 100:.2f}% (+/- {0.0 if std is None else std * 100:.2f} pp) [{status}]"
        )
    sections.append({"title": "Accuracy & Guardrail", "items": accuracy_items})

    efficiency_items = []
    for key, cmp in comparisons.items():
        step = cmp.get("step_time_delta_pct")
        wall = cmp.get("wall_time_delta_pct")
        param = cmp.get("param_delta_pct")
        if step is not None:
            efficiency_items.append(f"{key} step time delta: {step:.1f}%")
        if wall is not None:
            efficiency_items.append(f"{key} wall time delta: {wall:.1f}%")
        if param is not None:
            efficiency_items.append(f"{key} parameter delta: {param:.1f}%")
    if not efficiency_items:
        efficiency_items.append("No baseline comparisons available.")
    sections.append({"title": "Efficiency vs Baseline", "items": efficiency_items})

    stability_items = []
    for key, run in runs.items():
        derived = run.get("derived", {})
        test_std = derived.get("test_accuracy_std")
        val_std = derived.get("val_accuracy_std_max")
        if test_std is None and val_std is None:
            continue
        stability_items.append(
            f"{key}: test std {('-' if test_std is None else f'{test_std * 100:.2f} pp')}, "
            f"max val std {('-' if val_std is None else f'{val_std * 100:.2f} pp')}"
        )
    if not stability_items:
        stability_items.append("No stability stats available.")
    sections.append({"title": "Stability Across Seeds", "items": stability_items})

    convergence_items = []
    for key, run in runs.items():
        epoch = run.get("epoch_to_97")
        if epoch is None:
            convergence_items.append(f"{key}: val accuracy never reached {guardrail * 100:.1f}%")
        else:
            convergence_items.append(f"{key}: reaches {guardrail * 100:.1f}% at epoch {epoch}")
    sections.append({"title": "Convergence Speed", "items": convergence_items})

    comparability_items = []
    for key, cmp in comparisons.items():
        ratio = cmp.get("param_ratio")
        if ratio is None:
            continue
        delta_pct = (ratio - 1.0) * 100.0
        if abs(delta_pct) <= 10.0:
            comparability_items.append(f"{key}: param ratio {ratio:.2f} (within 10%)")
        else:
            comparability_items.append(f"{key}: param ratio {ratio:.2f} ({delta_pct:+.1f}%), consider rebalancing")
    if not comparability_items:
        comparability_items.append("No param comparability data.")
    sections.append({"title": "Comparability Check", "items": comparability_items})

    return sections


def derive_metrics(run: dict) -> dict:
    agg = run.get("summary", {}).get("aggregate", {})
    test_acc = agg.get("test_accuracy", {}).get("mean")
    test_std = agg.get("test_accuracy", {}).get("std")
    wall = agg.get("wall_time_sec", {}).get("mean")
    param_mean = run.get("param_count", {}).get("mean")

    train_curve = run.get("curves", {}).get("train", {})
    final_train_acc = None
    if train_curve.get("accuracy"):
        final_train_acc = train_curve["accuracy"][-1]
    generalization_gap = None
    if final_train_acc is not None and test_acc is not None:
        generalization_gap = final_train_acc - test_acc

    acc_per_wall = None
    if test_acc is not None and wall:
        acc_per_wall = test_acc / wall

    acc_per_param = None
    if test_acc is not None and param_mean:
        acc_per_param = test_acc / param_mean

    val_curve = run.get("curves", {}).get("val", {})
    val_std_values = [v for v in val_curve.get("accuracy_std", []) if v is not None]
    val_accuracy_std_max = max(val_std_values) if val_std_values else None

    return {
        "final_train_accuracy": final_train_acc,
        "generalization_gap": generalization_gap,
        "accuracy_per_wall_time": acc_per_wall,
        "accuracy_per_param": acc_per_param,
        "test_accuracy_std": test_std,
        "val_accuracy_std_max": val_accuracy_std_max,
    }


def infer_dataset(run: dict) -> str:
    data_info = run.get("data_info") or {}
    dataset = data_info.get("dataset")
    if dataset:
        return dataset
    config = run.get("config") or {}
    return config.get("dataset") or "unknown"


def infer_group_id(run: dict, dataset: str) -> str:
    run_id = run.get("run_id") or ""
    model = run.get("model")
    if run_id and dataset and model:
        suffix = f"-{dataset}-{model}"
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return run_id or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dashboard report from run artifacts")
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--output", type=str, default="dashboard/data/report.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: p.name)
    if args.limit > 0:
        run_dirs = run_dirs[-args.limit :]

    runs_by_dataset: dict[str, dict[str, dict[str, dict]]] = defaultdict(dict)
    run_mtime: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for run_dir in run_dirs:
        try:
            run = load_run(run_dir)
        except FileNotFoundError:
            continue
        model_key = run.get("model")
        if not model_key:
            continue
        dataset = infer_dataset(run)
        group_id = infer_group_id(run, dataset)
        summary_path = run_dir / "summary.json"
        mtime = summary_path.stat().st_mtime
        current_mtime = run_mtime[dataset][group_id].get(model_key)
        if current_mtime is None or mtime >= current_mtime:
            runs_by_dataset[dataset].setdefault(group_id, {})[model_key] = run
            run_mtime[dataset][group_id][model_key] = mtime

    guardrail = 0.97
    datasets_report: dict[str, dict[str, Any]] = {}
    for dataset, groups in runs_by_dataset.items():
        group_reports: dict[str, Any] = {}
        dataset_info = None
        group_mtimes: dict[str, float] = {}
        for group_id, runs in groups.items():
            comparisons = {}
            if "mlp" in runs and "mlp-obl" in runs:
                comparisons["mlp"] = compare_runs(runs["mlp"], runs["mlp-obl"])
            if "cnn" in runs and "cnn-obl" in runs:
                comparisons["cnn"] = compare_runs(runs["cnn"], runs["cnn-obl"])

            data_info = None
            for run in runs.values():
                if run.get("data_info"):
                    data_info = run["data_info"]
                    break
            if dataset_info is None and data_info:
                dataset_info = data_info

            group_reports[group_id] = {
                "run_group": group_id,
                "runs": runs,
                "comparisons": comparisons,
                "insights": build_insights(runs),
                "analysis_sections": build_analysis_sections(runs, comparisons, guardrail),
                "guardrail": guardrail,
                "dataset": data_info,
            }
            group_mtimes[group_id] = max(run_mtime[dataset][group_id].values())

        latest_group = None
        if group_mtimes:
            latest_group = max(group_mtimes.items(), key=lambda item: item[1])[0]

        if dataset_info is None:
            dataset_info = {"dataset": dataset}

        datasets_report[dataset] = {
            "groups": group_reports,
            "dataset": dataset_info,
            "latest_group": latest_group,
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "guardrail": guardrail,
        "datasets": datasets_report,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
