# Grad-Speedup Dashboard Spec

Goal
- Provide a rich, interactive dashboard for exploring experiment runs and comparing methods.
- Enable multi-metric tradeoff analysis (speed, quality, cost, stability) with minimal friction.

Scope
- Data source: run artifacts under `project/runs/grad-speedup/`.
- Code lives under `project/grad-speedup/dashboard/` and `project/grad-speedup/scripts/`.
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

Dashboard UI (Streamlit)
- Sidebar filters: run_id, model, optimizer, step_rule, direction, clip_mode, sparsity, anderson, date range.
- Tabs:
  1) Overview
     - Run table (sortable): run_id, model, optimizer, modules, mean step time, targets reached.
     - Scatter: cost-to-target vs accuracy (select target).
     - Bar: speedup vs baseline for selected target (if baseline chosen).
  2) Run Detail
     - Select run + seed.
     - Curves: train loss/acc vs step/time; test acc vs epoch.
     - Step diagnostics: step_size, grad_norm, curvature, lr vs step/time.
     - Step time distribution (hist + p50/p90).
  3) Compare
     - Overlay curves across runs (test acc vs epoch/time, train loss vs step/time).
     - Summary table with deltas vs baseline.
  4) Diagnostics
     - Line search acceptance/reject counts, avg iters.
     - Preconditioner update/apply counts and time.
     - Sparsity fraction and effective FLOPs.
  5) System
     - Device/torch info per run; deterministic flags.

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
