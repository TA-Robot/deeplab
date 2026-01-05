# Task Ticket: Grad-Speedup Module B (Direction Preconditioner v1)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-module-b
- role/agent: implementer-grad-speedup-module-b
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 180 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
DEPRECATED. Superseded by paper-accurate direction method tickets
(`task-20260105-grad-speedup-direction-soap-shampoo.md` and
`task-20260105-grad-speedup-direction-sophia-muon.md`).

## 3) Background / Context
This ticket described a minimal placeholder preconditioner. It does not satisfy
the paper-accuracy requirement and should not be executed.

## 4) Scope
In scope:
- Implement one direction module that re-scales gradients or parameter updates.
- Register it under a simple name (e.g., "diag-precond").
- Add CLI/config wiring.

Out of scope:
- Shampoo/SOAP/Muon full implementations.

## 5) Requirements
Must:
- Default behavior unchanged when module is "none".
- Module exposes update frequency and epsilon/damping.
- Log any module stats (e.g., scale factor mean/p50/p90).

Should:
- Allow per-parameter or per-layer scaling with minimal overhead.

## 6) Acceptance Criteria
- [ ] New module option works on CIFAR-10 smoke run.
- [ ] metrics.jsonl includes module stats when enabled.
- [ ] README documents the new module flag.

## 7) Implementation Notes
Suggested implementation:
- Maintain EMA of grad^2 (like RMS) but use it only as a preconditioner.
- Apply scaling to grads before optimizer.step().
- Update EMA every step or every K steps.

Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py
- project/grad-speedup/README.md

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-mod-b --step-rule none --direction diag-precond
```

## 9) Deliverables
- Module B implementation and docs.

## 10) Risks
- Double-counting with optimizer adaptivity; keep scope minimal.
