# Task Ticket: Grad-Speedup Reporting v2 (CSV + Aggregates)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-reporting-v2
- role/agent: implementer-grad-speedup-reporting-v2
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 120 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Extend reporting to emit a flat CSV and richer JSON aggregates for A* targets
from runs/grad-speedup. Output should be separate from ROS-ALTH reporting.

## 3) Background / Context
Current build_grad_speedup_report.py only collects per-run summaries.
We need tabular outputs for quick analysis and plotting.

## 4) Scope
In scope:
- Read runs/grad-speedup/*/summary.json.
- Emit JSON with run-level metadata and per-target stats.
- Emit CSV with one row per seed per run.

Out of scope:
- Dashboard UI integration.

## 5) Requirements
Must:
- Handle missing targets (nulls) gracefully.
- Include steps/time/cost for each A*.
- Do not modify ROS-ALTH scripts.

## 6) Acceptance Criteria
- [ ] New report script writes reports/grad-speedup-report.json and .csv
- [ ] JSON includes per-run aggregates and per-seed entries
- [ ] CSV columns documented in README

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/scripts/build_grad_speedup_report.py
- project/grad-speedup/README.md (add report usage + CSV columns)

Suggested CSV columns:
- run_id, seed, model, optimizer, mean_step_time_sec
- target, steps_to_target, time_to_target_sec, cost_to_target_sec

## 8) Commands
```
cd project/grad-speedup
python scripts/build_grad_speedup_report.py --runs-dir ../runs/grad-speedup --output reports/grad-speedup-report.json
```

## 9) Deliverables
- Updated report script and README notes.

## 10) Risks
- Inconsistent summary.json formats; add defensive parsing.
