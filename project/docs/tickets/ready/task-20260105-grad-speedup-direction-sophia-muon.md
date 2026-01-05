# Task Ticket: Grad-Speedup Direction Methods (Sophia + Muon)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-direction-sophia-muon
- role/agent: implementer-grad-speedup-sophia-muon
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 240 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Implement Sophia and Muon per their papers, with exact update rules and
hyperparameters surfaced in configs.

## 3) Background / Context
Sophia and Muon are key direction methods from the temp survey and must be
implemented accurately to evaluate step reductions.

## 4) Scope
In scope:
- Implement Sophia (arXiv:2305.14342) with diagonal Hessian estimator and clipping.
- Implement Muon (arXiv:2502.16982) with orthogonalization-based update and scaling.
- Add config flags for paper-specified hyperparameters.
- Add logging for curvature/clipping stats (Sophia) and orthogonalization iterations (Muon).

Out of scope:
- Distributed Muon variants or GPU kernels beyond PyTorch.

## 5) Requirements
Must:
- Conform to paper update rules; no heuristics.
- Default behavior unchanged when not enabled.
- Keep module exclusive (only one direction method active).

## 6) Acceptance Criteria
- [ ] Sophia and Muon can run on CIFAR-10 smoke run (1 epoch).
- [ ] metrics.jsonl includes method-specific stats.
- [ ] README documents flags and configuration.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/src/modules.py
- project/grad-speedup/scripts/run_cifar10.py
- project/grad-speedup/README.md

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-sophia --direction sophia
```

## 9) Deliverables
- Sophia and Muon implementations and docs.

## 10) Risks
- HVP/orthogonalization cost may dominate on small models; measure overhead.
