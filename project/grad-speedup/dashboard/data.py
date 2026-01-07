from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    pd = None


REQUIRED_COLUMNS = (
    "run_id",
    "seed",
    "model",
    "optimizer",
    "step_rule",
    "direction",
    "clip_mode",
    "sparsity",
)


def _require_pandas():
    if pd is None:
        raise ModuleNotFoundError("pandas is required for dashboard.data; install pandas to use load_all_runs.")
    return pd


def load_all_runs(
    runs_dir: str | Path,
    *,
    baseline_run_id: Optional[str] = None,
    return_targets: bool = False,
    queue_file: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load grad-speedup run artifacts into DataFrames.

    Args:
        runs_dir: Directory containing run subdirectories.
        baseline_run_id: Optional run_id to compute speedup_vs_baseline in runs_df.
        return_targets: When True, returns a targets_df as the 4th value.
    """
    pd_module = _require_pandas()
    runs_dir = _resolve_runs_dir(Path(runs_dir))
    run_records: List[Dict[str, Any]] = []
    epoch_records: List[Dict[str, Any]] = []
    step_records: List[Dict[str, Any]] = []
    target_records: List[Dict[str, Any]] = []

    if not runs_dir.is_dir():
        runs_df = pd_module.DataFrame()
        epochs_df = pd_module.DataFrame()
        steps_df = pd_module.DataFrame()
        targets_df = pd_module.DataFrame()
        _ensure_columns(runs_df, REQUIRED_COLUMNS)
        _ensure_columns(epochs_df, REQUIRED_COLUMNS + ("epoch_elapsed_time_sec",))
        _ensure_columns(steps_df, REQUIRED_COLUMNS + ("step_elapsed_time_sec",))
        _ensure_columns(
            targets_df,
            REQUIRED_COLUMNS
            + (
                "target",
                "steps_to_target",
                "time_to_target_sec",
                "cost_to_target_sec",
                "time_to_target",
                "cost_to_target",
            ),
        )
        if return_targets:
            return runs_df, epochs_df, steps_df, targets_df
        return runs_df, epochs_df, steps_df

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue

        config = _read_json(run_dir / "config.json")
        env = _read_json(run_dir / "env.json")
        summary_path = run_dir / "summary.json"
        summary = _read_json(summary_path)
        run_summary_present = summary_path.exists()

        run_id = _pick_run_id(run_dir, config, summary)
        base_meta = _build_run_meta(run_id, run_dir, config, summary, env)

        seed_dirs = _collect_seed_dirs(run_dir)
        seed_summaries = _index_seed_summaries(summary)
        seeds = _collect_seeds(config, summary, seed_dirs)
        if not seeds:
            seeds = [None]

        for seed in seeds:
            seed_summary = seed_summaries.get(seed)
            if seed_summary is None:
                seed_dir = seed_dirs.get(seed)
                if seed_dir is not None:
                    seed_summary = _read_json(seed_dir / "summary.json")

            run_record = dict(base_meta)
            run_record["seed"] = seed
            run_record.update(_extract_seed_summary(seed_summary))
            run_records.append(run_record)

            _append_target_rows(
                target_records,
                base_meta=base_meta,
                seed=seed,
                seed_summary=seed_summary,
                config_targets=config.get("targets"),
            )

            seed_dir = seed_dirs.get(seed)
            if seed_dir is None:
                _update_status_fields(
                    run_record,
                    seed_summary=seed_summary,
                    run_summary_present=run_summary_present,
                    last_step=None,
                    last_epoch=None,
                    last_eval_step=None,
                )
                continue
            metrics_path = seed_dir / "metrics.jsonl"
            if not metrics_path.exists():
                _update_status_fields(
                    run_record,
                    seed_summary=seed_summary,
                    run_summary_present=run_summary_present,
                    last_step=None,
                    last_epoch=None,
                    last_eval_step=None,
                )
                continue

            mean_step_time_sec = seed_summary.get("mean_step_time_sec") if seed_summary else None
            epoch_time_by_epoch: Dict[int, float] = {}
            epoch_start_idx = len(epoch_records)
            running_step_time_sec = 0.0
            last_step = None
            last_epoch = None
            last_eval_step = None
            for entry in _read_jsonl(metrics_path):
                entry_type = entry.get("type")
                if entry_type == "epoch_timing":
                    epoch = _coerce_int(entry.get("epoch"))
                    if epoch is not None and entry.get("epoch_time_sec") is not None:
                        epoch_time_by_epoch[epoch] = entry.get("epoch_time_sec")
                    continue

                record = dict(base_meta)
                record["seed"] = seed
                record.update(entry)
                if entry_type == "epoch":
                    last_epoch = _coerce_int(entry.get("epoch")) or last_epoch
                    if str(entry.get("split")).lower() in ("test", "val", "validation"):
                        last_eval_step = _coerce_int(entry.get("global_step")) or last_eval_step
                    epoch_records.append(record)
                elif entry_type == "step":
                    last_step = _coerce_int(entry.get("global_step")) or last_step
                    step_time_ms = _coerce_float_or_none(entry.get("step_time_ms"))
                    if step_time_ms is None:
                        record["step_elapsed_time_sec"] = None
                    else:
                        running_step_time_sec += step_time_ms / 1000.0
                        record["step_elapsed_time_sec"] = running_step_time_sec
                    step_records.append(record)

            for idx in range(epoch_start_idx, len(epoch_records)):
                epoch_record = epoch_records[idx]
                epoch = _coerce_int(epoch_record.get("epoch"))
                if epoch is not None and epoch in epoch_time_by_epoch:
                    epoch_record["epoch_time_sec"] = epoch_time_by_epoch[epoch]
                _derive_epoch_metrics(epoch_record)
            _attach_epoch_elapsed_time(epoch_records, epoch_start_idx, mean_step_time_sec)
            _update_status_fields(
                run_record,
                seed_summary=seed_summary,
                run_summary_present=run_summary_present,
                last_step=last_step,
                last_epoch=last_epoch,
                last_eval_step=last_eval_step,
            )

    runs_df = pd_module.DataFrame(run_records)
    epochs_df = pd_module.DataFrame(epoch_records)
    steps_df = pd_module.DataFrame(step_records)
    targets_df = pd_module.DataFrame(target_records)

    _attach_run_level_metrics(runs_df, epochs_df)
    _add_time_convenience_columns(runs_df)

    _ensure_columns(
        runs_df,
        REQUIRED_COLUMNS
        + (
            "status",
            "progress_pct",
            "progress_steps",
            "last_step",
            "last_epoch",
            "last_eval_step",
            "max_steps",
            "eval_interval_steps",
            "eval_interval_epochs",
        ),
    )
    _ensure_columns(epochs_df, REQUIRED_COLUMNS + ("epoch_elapsed_time_sec",))
    _ensure_columns(steps_df, REQUIRED_COLUMNS + ("step_elapsed_time_sec",))
    _ensure_columns(
        targets_df,
        REQUIRED_COLUMNS
        + (
            "target",
            "steps_to_target",
            "time_to_target_sec",
            "cost_to_target_sec",
            "time_to_target",
            "cost_to_target",
        ),
    )

    if queue_file:
        queue_path = Path(queue_file)
        if not queue_path.is_absolute():
            queue_path = queue_path.resolve()
        _merge_queue_runs(runs_df, queue_path)

    if baseline_run_id:
        _add_speedup_vs_baseline(runs_df, baseline_run_id)

    if return_targets:
        return runs_df, epochs_df, steps_df, targets_df
    return runs_df, epochs_df, steps_df


def _merge_queue_runs(runs_df: pd.DataFrame, queue_path: Path) -> None:
    if not queue_path.exists():
        return
    queued = []
    for line in queue_path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        run_id = _extract_run_id(raw)
        if not run_id:
            continue
        queued.append(run_id)
    if not queued:
        return
    existing = set(runs_df["run_id"].astype(str).tolist()) if "run_id" in runs_df.columns else set()
    for run_id in queued:
        if run_id in existing:
            continue
        runs_df.loc[len(runs_df)] = {
            "run_id": run_id,
            "status": "queued",
        }


def _extract_run_id(cmd: str) -> Optional[str]:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    for idx, token in enumerate(parts):
        if token == "--run-id" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _resolve_runs_dir(runs_dir: Path) -> Path:
    runs_dir = runs_dir.expanduser()
    if runs_dir.exists():
        return runs_dir
    project_root = Path(__file__).resolve().parents[2]
    repo_root = project_root.parent
    for base in (project_root, repo_root):
        candidate = base / runs_dir
        if candidate.exists():
            return candidate
    return runs_dir


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


def _normalize_none(value: Any) -> Any:
    if value is None:
        return "none"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return "none"
    return value


def _pick_run_id(run_dir: Path, config: Dict[str, Any], summary: Dict[str, Any]) -> str:
    return str(config.get("run_id") or summary.get("run_id") or run_dir.name)


def _build_run_meta(
    run_id: str,
    run_dir: Path,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    env: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model": config.get("model") or summary.get("model"),
        "optimizer": config.get("optimizer") or summary.get("optimizer"),
        "param_mode": _normalize_none(config.get("param_mode")),
        "relora_scope": config.get("relora_scope"),
        "relora_rank": config.get("relora_rank"),
        "relora_alpha": config.get("relora_alpha"),
        "relora_init": config.get("relora_init"),
        "relora_merge_interval": config.get("relora_merge_interval"),
        "relora_warmstart_steps": config.get("relora_warmstart_steps"),
        "step_rule": _normalize_none(config.get("step_rule")),
        "direction": _normalize_none(config.get("direction")),
        "clip_mode": _normalize_none(config.get("clip_mode")),
        "sparsity": _normalize_none(config.get("sparsity")),
        "batch_size": config.get("batch_size"),
        "epochs": config.get("epochs"),
        "max_steps": config.get("max_steps"),
        "lr": config.get("lr"),
        "momentum": config.get("momentum"),
        "weight_decay": config.get("weight_decay"),
        "device": config.get("device") or env.get("device"),
        "deterministic": config.get("deterministic") if "deterministic" in config else env.get("deterministic"),
        "config_path": config.get("config_path"),
        "data_dir": config.get("data_dir"),
        "data_seed": config.get("data_seed"),
        "val_size": config.get("val_size"),
        "num_workers": config.get("num_workers"),
        "num_threads": config.get("num_threads"),
        "targets": config.get("targets"),
        "eval_interval_steps": config.get("eval_interval_steps"),
        "eval_interval_epochs": config.get("eval_interval_epochs"),
        "env_python": env.get("python"),
        "env_torch": env.get("torch"),
        "env_torchvision": env.get("torchvision"),
        "env_platform": env.get("platform"),
        "env_processor": env.get("processor"),
        "env_cpu_count": env.get("cpu_count"),
    }


def _update_status_fields(
    run_record: Dict[str, Any],
    *,
    seed_summary: Optional[Dict[str, Any]],
    run_summary_present: bool,
    last_step: Optional[int],
    last_epoch: Optional[int],
    last_eval_step: Optional[int],
) -> None:
    max_steps = _coerce_int(run_record.get("max_steps"))
    epochs = _coerce_int(run_record.get("epochs"))
    progress_steps = _coerce_int(last_step)
    progress_pct = None
    if max_steps and progress_steps is not None:
        progress_pct = min(progress_steps / max_steps, 1.0)
    elif epochs and last_epoch is not None:
        progress_pct = min(last_epoch / epochs, 1.0)

    if seed_summary is not None:
        status = "completed"
    elif run_summary_present:
        status = "completed"
    elif progress_steps is not None or last_epoch is not None:
        status = "running"
    else:
        status = "created"

    run_record["status"] = status
    run_record["progress_steps"] = progress_steps
    run_record["progress_pct"] = progress_pct
    run_record["last_step"] = progress_steps
    run_record["last_epoch"] = last_epoch
    run_record["last_eval_step"] = last_eval_step


def _collect_seed_dirs(run_dir: Path) -> Dict[int, Path]:
    seed_dirs: Dict[int, Path] = {}
    for entry in run_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("seed-"):
            continue
        seed = _coerce_int(entry.name.split("-", 1)[-1])
        if seed is None:
            continue
        seed_dirs[seed] = entry
    return seed_dirs


def _index_seed_summaries(summary: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    per_seed = summary.get("per_seed")
    if not isinstance(per_seed, list):
        return {}
    seed_map: Dict[int, Dict[str, Any]] = {}
    for item in per_seed:
        if not isinstance(item, dict):
            continue
        seed = _coerce_int(item.get("seed"))
        if seed is None:
            continue
        seed_map[seed] = item
    return seed_map


def _collect_seeds(
    config: Dict[str, Any],
    summary: Dict[str, Any],
    seed_dirs: Dict[int, Path],
) -> List[int]:
    seeds: List[int] = []
    for source in (config.get("seeds"), summary.get("seeds")):
        if isinstance(source, list):
            for value in source:
                seed = _coerce_int(value)
                if seed is not None:
                    seeds.append(seed)
    seeds.extend(seed_dirs.keys())
    return sorted(set(seeds))


def _extract_seed_summary(seed_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not seed_summary:
        return {
            "steps_per_epoch": None,
            "mean_step_time_sec": None,
            "total_steps": None,
            "total_time_sec": None,
        }
    return {
        "steps_per_epoch": seed_summary.get("steps_per_epoch"),
        "mean_step_time_sec": seed_summary.get("mean_step_time_sec"),
        "total_steps": seed_summary.get("total_steps"),
        "total_time_sec": seed_summary.get("total_time_sec"),
    }


def _append_target_rows(
    records: List[Dict[str, Any]],
    *,
    base_meta: Dict[str, Any],
    seed: Optional[int],
    seed_summary: Optional[Dict[str, Any]],
    config_targets: Any,
) -> None:
    targets: Dict[str, Any] = {}
    if seed_summary and isinstance(seed_summary.get("targets"), dict):
        targets = seed_summary.get("targets") or {}
    elif isinstance(config_targets, list):
        targets = {str(value): None for value in config_targets}

    if not targets:
        return

    mean_step_time_sec = seed_summary.get("mean_step_time_sec") if seed_summary else None
    for target_key, target_data in targets.items():
        row = dict(base_meta)
        row["seed"] = seed
        row["target"] = _coerce_float(target_key)
        row["steps_to_target"] = None
        row["time_to_target_sec"] = None
        row["cost_to_target_sec"] = None
        row["target_accuracy"] = None
        row["target_epoch"] = None
        if isinstance(target_data, dict):
            row["steps_to_target"] = target_data.get("steps_to_target")
            row["time_to_target_sec"] = target_data.get("time_to_target_sec")
            row["cost_to_target_sec"] = target_data.get("cost_to_target_sec")
            row["target_accuracy"] = target_data.get("accuracy")
            row["target_epoch"] = target_data.get("epoch")
        _derive_target_metrics(row, mean_step_time_sec)
        row["time_to_target"] = row.get("time_to_target_sec")
        row["cost_to_target"] = row.get("cost_to_target_sec")
        records.append(row)


def _derive_target_metrics(row: Dict[str, Any], mean_step_time_sec: Optional[float]) -> None:
    # Keep target timing as recorded by the run; do not derive from mean step time.
    return


def _derive_epoch_metrics(row: Dict[str, Any]) -> None:
    dense_flops = row.get("dense_flops")
    effective_flops = row.get("effective_flops")
    row["effective_flops_ratio"] = _safe_div(effective_flops, dense_flops)

    attempts = _coerce_float(row.get("line_search_attempts"))
    accepted = _coerce_float(row.get("line_search_accepted"))
    try:
        attempts_value = float(attempts)
    except (TypeError, ValueError):
        attempts_value = None
    if attempts_value is None or attempts_value <= 0:
        row["line_search_accept_rate"] = None
    else:
        row["line_search_accept_rate"] = _safe_div(accepted, attempts_value)

    precond_update = row.get("precond_update_time_s")
    precond_apply = row.get("precond_apply_time_s")
    precond_total = None
    if precond_update is not None or precond_apply is not None:
        precond_total = float(precond_update or 0.0) + float(precond_apply or 0.0)

    denom = row.get("epoch_time_sec")
    if denom is None:
        step_time_ms = row.get("step_time_ms")
        steps = row.get("steps")
        if step_time_ms is not None and steps:
            denom = (step_time_ms * steps) / 1000.0

    row["precond_overhead"] = _safe_div(precond_total, denom)


def _estimate_epoch_time_sec(row: Dict[str, Any], mean_step_time_sec: Optional[float]) -> Optional[float]:
    epoch_time_sec = _coerce_float_or_none(row.get("epoch_time_sec"))
    if epoch_time_sec is not None:
        return epoch_time_sec

    steps = _coerce_float_or_none(row.get("steps"))
    if steps is None:
        return None

    step_time_ms = _coerce_float_or_none(row.get("step_time_ms"))
    if step_time_ms is not None:
        return (step_time_ms * steps) / 1000.0

    mean_step_time = _coerce_float_or_none(mean_step_time_sec)
    if mean_step_time is None:
        return None
    return mean_step_time * steps


def _attach_epoch_elapsed_time(
    epoch_records: List[Dict[str, Any]],
    start_idx: int,
    mean_step_time_sec: Optional[float],
) -> None:
    epoch_durations: Dict[int, Optional[float]] = {}
    for idx in range(start_idx, len(epoch_records)):
        record = epoch_records[idx]
        epoch = _coerce_int(record.get("epoch"))
        if epoch is None:
            continue
        duration = _estimate_epoch_time_sec(record, mean_step_time_sec)
        if epoch not in epoch_durations or (epoch_durations[epoch] is None and duration is not None):
            epoch_durations[epoch] = duration

    running_epoch_time_sec: Optional[float] = 0.0
    epoch_elapsed: Dict[int, Optional[float]] = {}
    for epoch in sorted(epoch_durations):
        duration = epoch_durations[epoch]
        if duration is None or running_epoch_time_sec is None:
            epoch_elapsed[epoch] = None
            running_epoch_time_sec = None
        else:
            running_epoch_time_sec += duration
            epoch_elapsed[epoch] = running_epoch_time_sec

    for idx in range(start_idx, len(epoch_records)):
        record = epoch_records[idx]
        epoch = _coerce_int(record.get("epoch"))
        if epoch is None:
            record["epoch_elapsed_time_sec"] = None
        else:
            record["epoch_elapsed_time_sec"] = epoch_elapsed.get(epoch)


def _add_speedup_vs_baseline(runs_df: pd.DataFrame, baseline_run_id: str) -> None:
    if "mean_step_time_sec" not in runs_df.columns:
        return
    baseline = runs_df[runs_df["run_id"] == baseline_run_id]
    if baseline.empty:
        return
    baseline_mean = pd.to_numeric(baseline["mean_step_time_sec"], errors="coerce").dropna()
    if baseline_mean.empty:
        return
    baseline_value = float(baseline_mean.mean())
    runs_df["speedup_vs_baseline"] = baseline_value / pd.to_numeric(
        runs_df["mean_step_time_sec"], errors="coerce"
    )


def _attach_run_level_metrics(runs_df: pd.DataFrame, epochs_df: pd.DataFrame) -> None:
    if runs_df.empty or epochs_df.empty:
        return
    if "run_id" not in runs_df.columns or "run_id" not in epochs_df.columns:
        return
    if "seed" not in runs_df.columns or "seed" not in epochs_df.columns:
        return

    epochs = epochs_df.copy()
    if "split" not in epochs.columns:
        return

    test_epochs = epochs[epochs["split"].astype(str).isin(["test", "val", "validation"])]
    if test_epochs.empty:
        return

    if "epoch" in test_epochs.columns:
        test_epochs = test_epochs.sort_values("epoch")
    elif "global_step" in test_epochs.columns:
        test_epochs = test_epochs.sort_values("global_step")

    final_rows = (
        test_epochs.groupby(["run_id", "seed"], dropna=False)
        .tail(1)[["run_id", "seed", "accuracy", "loss", "epoch", "global_step"]]
        .rename(
            columns={
                "accuracy": "final_test_acc",
                "loss": "final_test_loss",
                "epoch": "final_test_epoch",
                "global_step": "final_test_step",
            }
        )
    )

    best_rows = (
        test_epochs.groupby(["run_id", "seed"], dropna=False)["accuracy"]
        .max()
        .reset_index()
        .rename(columns={"accuracy": "best_test_acc"})
    )

    merged = runs_df.merge(final_rows, on=["run_id", "seed"], how="left")
    merged = merged.merge(best_rows, on=["run_id", "seed"], how="left")
    for col in merged.columns:
        runs_df[col] = merged[col]


def _add_time_convenience_columns(runs_df: pd.DataFrame) -> None:
    if runs_df.empty:
        return
    if "mean_step_time_sec" in runs_df.columns:
        runs_df["mean_step_time_ms"] = pd.to_numeric(runs_df["mean_step_time_sec"], errors="coerce") * 1000.0


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
