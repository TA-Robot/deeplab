# Task Ticket: Dashboard Data Layer Improvements

## 1) Meta
- ticket_id: task-20260106-grad-speedup-dashboard-data
- role/agent: implementer-dashboard-data
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 90 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Improve dashboard data cleanliness and time-based metrics so UI can show per-time accuracy and avoid None/none confusion.

Success looks like:
- Categorical fields (step_rule/direction/clip_mode/sparsity) are normalized so "None"/None/"" display consistently as "none".
- Steps and epochs include elapsed time columns usable for plotting accuracy vs time.

## 3) Background / Context
User reported dashboard confusion:
- run table has direction values "none" and "None".
- wants accuracy and time at finer granularity, not only final accuracy / mean_step_time.

## 4) Scope
In scope:
- Update `project/grad-speedup/dashboard/data.py` to normalize category fields.
- Add cumulative elapsed time columns to steps and epochs records, per run_id+seed.
  - `step_elapsed_time_sec` (cumulative sum of step_time_ms) for steps.
  - `epoch_elapsed_time_sec` (cumulative sum of epoch_time_sec if present; else estimated from steps).

Out of scope:
- UI changes (handled in separate ticket).

## 5) Requirements
Must:
- No new dependencies.
- Preserve existing columns and add new ones.
- If step_time_ms missing, leave elapsed columns null.

## 6) Acceptance Criteria
- [ ] `runs_df` no longer shows both "none" and "None" for direction/step_rule/clip/sparsity.
- [ ] `steps_df` has `elapsed_time_sec` (or `step_elapsed_time_sec`) for runs with step_time_ms.
- [ ] `epochs_df` has `epoch_elapsed_time_sec` for runs with epoch_time_sec or steps.

## 7) Implementation Notes
- Add `_normalize_none(value: Any) -> str | Any` helper and apply in `_build_run_meta`.
- While parsing metrics.jsonl, keep a `running_step_time_sec` per seed and inject into step records.
- After epochs are collected, compute per run_id+seed cumulative epoch times; for epochs without epoch_time_sec, approximate using mean step time if available.

Files:
- project/grad-speedup/dashboard/data.py

## 8) Commands
No tests required. Verify with a quick local `python -c` if needed.

## 9) Deliverables
- Updated data.py with normalization + elapsed time columns.

## 10) Risks
- Ensure per-seed isolation (don’t mix elapsed time across seeds).

## 11) Reporting
Post a short summary: changes made + columns added.
