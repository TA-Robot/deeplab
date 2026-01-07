# Grad-Speedup Dashboard Spec

Goal
- Provide a rich, interactive dashboard for exploring experiment runs and comparing methods.
- Enable multi-metric tradeoff analysis (speed, quality, cost, stability) with minimal friction.

Scope
- Data source: run artifacts under `project/runs/grad-speedup/`.
- Code lives under `project/grad-speedup/dash/` (primary) and `project/grad-speedup/scripts/`.
- `project/grad-speedup/dashboard/` is treated as legacy (Streamlit prototype).
- No external services; run locally.

Data inputs (per run)
- `config.json`: run configuration and hyperparameters.
- `env.json`: device, torch version, CPU info, deterministic flag.
- `summary.json`: cost-to-target summary, steps/time, mean step time.
- `seed-*/metrics.jsonl`: step and epoch metrics.

Primary entities
- Run: `run_id`, model, optimizer, module choices (step_rule, direction, clip, sparsity, anderson).
- Seed: per-run seed subfolder.
- Step log: per logging interval (loss, acc, lr, step_size, grad_norm, curvature, step_time_ms, line_search stats).
- Epoch log: train/test loss/acc + aggregated stats (throughput, step size stats, precond stats, memory, sparsity).

Derived metrics
- Speed: mean step time, throughput, time-to-target, steps-to-target, cost-to-target.
- Quality: test accuracy at epoch, accuracy at targets.
- Stability: grad_norm, curvature, line_search acceptance rate, step_size distribution.
- Efficiency: effective_flops / dense_flops, sparsity_fraction, precond overhead (update/apply time).
- Memory: max_memory_bytes (if logged).
- Relative deltas: baseline-normalized ratios per target (time/cost/steps).

Dashboard UI (Plotly Dash)
- Left sidebar (controls): runs dir, queue file, reload, filters (model/optimizer/step_rule/direction/seed), target accuracy.
- Plot controls: color-by, legend position, label truncation, legend table toggle, hover disable toggle.
- Tabs:
  1) Overview
     - KPI cards: run coverage + best/median time-to-target.
     - Scatter: time-to-target (or cost-to-target) vs quality at selected target.
     - Leaderboard table (sortable/filterable) with run metadata + status/progress; selecting a row or clicking the scatter picks the run.
     - Bar chart: top-N fastest runs (target-specific).
  2) Compare
     - Select multiple runs.
     - Overlay curves: test acc vs epoch/time; train loss vs global_step.
     - Legend table (optional) shows full labels when legend is truncated.
  3) Run Detail
     - Run selector + per-run metadata, config/env view.
     - Curves + per-target summary.
  4) Diagnostics
     - Grad norm / curvature / sparsity curves (when logged).

Export
- Allow CSV export of run table and selected curves.

Non-goals
- No remote deployment; no real-time training streaming.

Implementation notes
- Prefer reading raw JSONL on demand with caching.
- Avoid committing generated artifacts; use `project/runs/grad-speedup/_dashboard_cache/` if caching is needed.

Success criteria
- A user can filter runs, compare at least 3 methods, and interpret speed/quality tradeoffs in under 2 minutes.
- Supports all metrics currently logged in `metrics.jsonl` and `summary.json`.
