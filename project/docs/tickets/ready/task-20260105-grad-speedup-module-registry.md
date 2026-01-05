# Task Ticket: Grad-Speedup Module Registry Skeleton

## 1) Meta
- ticket_id: task-20260105-grad-speedup-module-registry
- role/agent: implementer-grad-speedup-modules
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 120 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Introduce a minimal module registry and hook points for future Module B/D/E
without changing current baseline behavior.

## 3) Background / Context
We need a clean interface for adding curvature preconditioners and accelerators
later, while keeping baselines stable.

## 4) Scope
In scope:
- Define a module interface for step modification and optional state.
- Register modules by name in a small factory.
- Wire registry into train loop with no-op default.

Out of scope:
- Implementing actual Module B or Anderson acceleration.

## 5) Requirements
Must:
- Default path unchanged when no module is selected.
- Registry lives in project/grad-speedup/src/.
- CLI accepts a module name but defaults to "none".

## 6) Acceptance Criteria
- [ ] New registry with at least a "none" module.
- [ ] No change to baseline run results when module is "none".
- [ ] README documents new module hook point.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/modules.py (new)
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py

Suggested interface:
- module = build_module(name, config)
- module.before_step(state, grads, stats) -> (maybe update lr or delta)

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-mod-none
```

## 9) Deliverables
- Module registry and hook path documented.

## 10) Risks
- Over-engineering; keep it minimal and additive.
