# Task Ticket: Grad-Speedup Triage (Train Metrics Zero)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-triage-train-metrics-zero
- role/agent: triage-grad-speedup
- owner: PM
- created_at: 2026-01-06
- priority: P0
- timebox: 120 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Identify why train metrics show samples=0 and global_step=0 in metrics.jsonl and summaries.

## 3) Symptoms
- metrics.jsonl contains only epoch/test/timing records (no step logs).
- Train epoch loss=0, accuracy=0, samples=0 despite non-empty dataset.
- summary.json total_steps=0, mean_step_time_sec=null.

## 4) Scope
- Investigate train_one_epoch loop execution and logging.
- Confirm DataLoader iteration inside run script.
- Check any unintended early returns or skips.

## 5) Acceptance Criteria
- Root cause identified.
- Fix proposed (code or config), with a smoke run showing non-zero samples and step logs.
