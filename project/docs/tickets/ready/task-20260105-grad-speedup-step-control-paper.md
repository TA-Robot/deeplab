# Task Ticket: Grad-Speedup Step-Control Methods (Paper-Accurate)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-step-control-paper
- role/agent: implementer-grad-speedup-step-control
- owner: PM
- created_at: 2026-01-05
- priority: P0
- timebox: 240 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Replace heuristic step-control implementations with paper-accurate methods.
Step-control must match the algorithms from the cited papers.

## 3) Background / Context
Current "l0l1" and "eoss" modules are heuristics and not guaranteed to match
paper definitions. This ticket upgrades them to exact algorithms.

## 4) Scope
In scope:
- Implement paper-accurate step-control methods:
  - (L0,L1)-smoothness methods (normalized gradient / Polyak stepsize variants)
  - Adaptive Backtracking Line Search (ICLR 2025)
  - Stochastic Polyak Step-size with Momentum (ICLR 2025)
  - Silver step sizes (COLT 2025 / JMLR 2025)
  - Stochastic Adaptive GD Without Descent (2024)
- Remove or rename any heuristic modules to avoid confusion.
- Update configs/CLI to select exact methods by name.
- Update logging to record step size, backtracking iterations, acceptance stats.

Out of scope:
- Direction/preconditioning methods (SOAP/Sophia/Muon/etc).

## 5) Requirements
Must:
- Use exact update rule from the paper (cite section/equation in code comments or docs).
- Default behavior unchanged when step-control is disabled.
- CPU and CUDA safe; deterministic behavior if seed set.

## 6) Acceptance Criteria
- [ ] All listed step-control methods can run on 1-epoch CIFAR-10 smoke run.
- [ ] metrics.jsonl includes step size and acceptance stats.
- [ ] README documents flags and points to method-conformance.md.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/src/modules.py
- project/grad-speedup/scripts/run_cifar10.py
- project/grad-speedup/README.md
- project/docs/grad-speedup/method-conformance.md (if missing details)

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-step-control --step-rule <method>
```

## 9) Deliverables
- Paper-accurate step-control modules + docs.

## 10) Risks
- Some methods may require extra evaluations / line search overhead.
