# Task Ticket: Grad-Speedup Module D (GGNC Clip)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-ggnc
- role/agent: implementer-grad-speedup-ggnc
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 120 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
DEPRECATED. Superseded by paper-accurate GGNC + Anderson ticket
(`task-20260105-grad-speedup-ggnc-anderson.md`).

## 3) Background / Context
GGNC is part of the stability module family and should be optional.

## 4) Scope
This ticket only covered L2 clipping, which is not paper-accurate for GGNC.
Use the newer ticket for implementation details.

## 5) Requirements
Must:
- Default behavior unchanged when mode is "none".
- CPU and CUDA safe.

## 6) Acceptance Criteria
- [ ] Clip applied only when enabled.
- [ ] metrics.jsonl includes clip stats.
- [ ] README documents flags.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-ggnc --clip-mode global --clip-rho 1.0
```

## 9) Deliverables
- GGNC implementation and docs.

## 10) Risks
- Performance overhead if per-layer stats are heavy.
