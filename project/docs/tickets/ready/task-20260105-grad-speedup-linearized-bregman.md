# Task Ticket: Grad-Speedup Dynamic Sparsity (Linearized Bregman)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-linearized-bregman
- role/agent: implementer-grad-speedup-linearized-bregman
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 240 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Implement linearized Bregman / DessiLBI-style sparsity updates per the papers,
with metrics for sparsity level and effective FLOPs.

## 3) Background / Context
Dynamic sparsity is a key compute-reduction lever in the temp survey.
We need a paper-accurate implementation for CIFAR-10 experiments.

## 4) Scope
In scope:
- Implement linearized Bregman iteration (arXiv:1405.2380) or DessiLBI (arXiv:1905.09449).
- Expose lambda/threshold and update interval in config.
- Log sparsity fraction, effective FLOPs, and mask update frequency.

Out of scope:
- Multilevel mirror descent (separate ticket once paper located).

## 5) Requirements
Must:
- Paper-accurate update rule (dual variable update + shrinkage).
- Default behavior unchanged when disabled.

## 6) Acceptance Criteria
- [ ] Linearized Bregman runs on CIFAR-10 smoke run (1 epoch).
- [ ] metrics.jsonl includes sparsity fraction and effective FLOPs.
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
python scripts/run_cifar10.py --epochs 1 --run-id smoke-lb --sparsity linbreg
```

## 9) Deliverables
- Linearized Bregman implementation and docs.

## 10) Risks
- Masking may reduce accuracy; report both theoretical and effective FLOPs.
