# Task Ticket: Grad-Speedup Dashboard Data Layer

## 1) Meta
- ticket_id: task-20260105-grad-speedup-dashboard-data
- role/agent: implementer-grad-speedup-dashboard-data
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 180 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Implement a data ingestion layer that reads run artifacts and produces clean
pandas DataFrames for the dashboard (runs, steps, epochs, targets).

## 3) Background / Context
Run artifacts are stored under project/runs/grad-speedup. Each run has config,
summary, env, and seed-level metrics.jsonl. The dashboard needs consistent
dataframes and derived metrics (speed/quality/efficiency).

## 4) Scope
In scope:
- Create data loader module (suggested: project/grad-speedup/dashboard/data.py).
- Parse config.json, env.json, summary.json, metrics.jsonl.
- Return:
  - runs_df (one row per run_id or per seed)
  - epochs_df (type=epoch logs)
  - steps_df (type=step logs)
- Add derived metrics: cost_to_target, time_to_target, speedup vs baseline (optional),
  effective_flops_ratio, line_search_accept_rate, precond_overhead.

Out of scope:
- UI rendering (handled by dashboard-ui ticket).

## 5) Requirements
Must:
- Be robust to missing fields (nulls in metrics).
- Avoid hard-coded run IDs.
- Keep IO local (no network).

## 6) Acceptance Criteria
- [ ] A simple import + call loads all existing runs without exceptions.
- [ ] DataFrames include run_id, seed, model, optimizer, step_rule, direction, clip_mode, sparsity.
- [ ] Derived metrics are computed when inputs are present.

## 7) Implementation Notes
Suggested files:
- project/grad-speedup/dashboard/data.py
- project/grad-speedup/scripts/aggregate_runs.py (optional helper)

## 8) Commands
```
cd project/grad-speedup
python - <<'PY'
from dashboard.data import load_all_runs
runs, epochs, steps = load_all_runs('project/runs/grad-speedup')
print(runs.head())
PY
```

## 9) Deliverables
- Data loader module + optional aggregation script.

## 10) Risks
- Large metrics.jsonl files may be heavy; add caching or sampling if needed.
