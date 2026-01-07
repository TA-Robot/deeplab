# Task Ticket: Grad-Speedup Combo Stage 2 (Direction Sweep)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-combo-stage2-direction-sweep
- role/agent: experimenter-grad-speedup
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 180 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Run direction-only sweep (step_control=none) to rank direction methods before pairwise combos.

## 3) Preconditions
- Paper-accurate direction methods merged (SOAP, Shampoo, Sophia, Muon, diag-precond).
- Smoke-validated and logged in experiment-log.md.

## 4) Scope
- Model: small-cnn
- Dataset: CIFAR-10
- Budget: 1 epoch, 1 seed, GPU
- Directions: none, diag-precond, shampoo, soap, sophia, muon

## 5) Requirements
- Use step_rule=none, clip=none, sparsity=none, outer=none.
- Log run IDs in project/docs/experiment-log.md.
- Output artifacts under project/runs/grad-speedup.

## 6) Acceptance Criteria
- [ ] All 6 runs complete.
- [ ] Summary of ranking included in notes (best-to-worst by cost-to-target or loss@1epoch).

## 7) Suggested Commands
```
python scripts/run_cifar10.py --model small-cnn --epochs 1 --device cuda:0 --step-rule none --direction <dir> --run-id <run_id>
```
