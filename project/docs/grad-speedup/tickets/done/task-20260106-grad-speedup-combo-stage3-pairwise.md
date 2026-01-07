# Task Ticket: Grad-Speedup Combo Stage 3 (Step-Control × Direction)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-combo-stage3-pairwise
- role/agent: experimenter-grad-speedup
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 240 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Run pairwise combos for the top-2 step-control methods vs all direction methods.

## 3) Preconditions
- Stage 1 complete (step-control sweep).
- Stage 2 complete (direction sweep).
- Top-2 step-control methods identified in PM notes.

## 4) Scope
- Model: small-cnn
- Dataset: CIFAR-10
- Budget: 1 epoch, 1 seed, GPU
- Step-control: top-2 from Stage 1 that allow direction != none.
- Directions: none, diag-precond, shampoo, soap, sophia, muon

## 5) Requirements
- clip=none, sparsity=none, outer=none.
- Respect paper-accurate constraints: adaptive-backtracking, sps-momentum, sagd require direction=none.
- Log run IDs in project/docs/experiment-log.md.
- Output artifacts under project/runs/grad-speedup.

## 6) Acceptance Criteria
- [ ] All runs complete (2 x 6 = 12).
- [ ] Best pair identified by cost-to-target (or loss@1epoch).

## 7) Suggested Commands
```
python scripts/run_cifar10.py --model small-cnn --epochs 1 --device cuda:0 \
  --step-rule <step_rule> --direction <dir> --run-id <run_id>
```
