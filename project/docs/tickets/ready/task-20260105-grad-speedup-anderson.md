# Task Ticket: Grad-Speedup Module E (Anderson Acceleration)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-anderson
- role/agent: implementer-grad-speedup-anderson
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 180 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
DEPRECATED. Superseded by paper-accurate GGNC + Anderson ticket
(`task-20260105-grad-speedup-ggnc-anderson.md`).

## 3) Background / Context
Anderson is an outer acceleration module and must include a fallback path.

## 4) Scope
This ticket is superseded. Use the combined GGNC + Anderson ticket to verify the
paper algorithm and update the implementation accordingly.

## 5) Requirements
Must:
- Default behavior unchanged when disabled.
- Log usage count and fallback count.

## 6) Acceptance Criteria
- [ ] Runs without instability on small-cnn smoke run.
- [ ] Logs include accel metrics.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-anderson --anderson-memory 3 --anderson-interval 10
```

## 9) Deliverables
- Anderson module and docs.

## 10) Risks
- Numerical instability; keep m small and add damping.
