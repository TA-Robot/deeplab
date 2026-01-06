# Task Ticket: Grad-Speedup Reporting

## 1) Meta
- ticket_id: task-20260105-grad-speedup-reporting
- role/agent: implementer-grad-speedup-reporting
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 90 min for first working slice
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Provide reporting for the grad-speedup track that computes steps-to-target and
time-to-target for CIFAR-10. Output should be separate from the ROS-ALTH dashboard.

## 3) Background / Context
Existing build_report.py is tailored to OBL baseline vs OBL variants. The new
track requires target-threshold metrics per the grad-speedup spec.

## 4) Scope
In scope:
- Add a dedicated report builder (new script) or extend build_report.py safely.
- Compute T(A*), W(A*), and C(A*) for A* in {0.85, 0.90, 0.92, 0.94}.
- Output JSON under project/grad-speedup/reports/.

Out of scope:
- UI changes to the existing dashboard.

## 5) Requirements
Must:
- Works with runs in project/runs/grad-speedup/
- Per-run summary includes thresholds and mean/std across seeds

Should:
- Handle missing thresholds gracefully (null values)

## 6) Acceptance Criteria
- [ ] Report script runs and produces a JSON file for grad-speedup runs.
- [ ] JSON includes thresholds with steps/time/cost values.
- [ ] No changes required to the ROS-ALTH report unless explicitly needed.

## 7) Implementation Notes
Suggested approach:
- Create project/grad-speedup/scripts/build_grad_speedup_report.py
- Parse metrics.jsonl for train/val accuracy by epoch
- Convert epoch to steps via steps_per_epoch from metrics
- Compute time-to-target using epoch_time_sec sums

Files to touch:
- project/grad-speedup/scripts/build_grad_speedup_report.py (new)

## 8) Commands
```
cd project/grad-speedup
python scripts/build_grad_speedup_report.py --runs-dir ../runs/grad-speedup --output reports/grad-speedup-report.json
```

## 9) Deliverables
- New reporting script and brief usage notes in project/grad-speedup/README.md

## 10) Risks
- Incomplete metrics if runs terminate early; must handle missing data.
