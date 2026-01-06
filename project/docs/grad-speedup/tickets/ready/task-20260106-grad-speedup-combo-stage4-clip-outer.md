# Task Ticket: Grad-Speedup Combo Stage 4 (Clip/Outer Sweep)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-combo-stage4-clip-outer
- role/agent: experimenter-grad-speedup
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 180 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Run clip + Anderson sweeps on the best pair from Stage 3.

## 3) Preconditions
- Stage 3 complete; top-1 pair identified.
- GGNC + Anderson implemented and smoke-validated.

## 4) Scope
- Model: small-cnn
- Dataset: CIFAR-10
- Budget: 1 epoch, 1 seed, GPU
- Clip: none, ggnc-global, ggnc-layerwise
- Outer: none, anderson

## 5) Requirements
- direction/step-control fixed to best pair.
- Log run IDs in project/docs/experiment-log.md.
- Output artifacts under project/runs/grad-speedup.

## 6) Acceptance Criteria
- [ ] All runs complete (3 x 2 = 6).
- [ ] Best combo identified by cost-to-target (or loss@1epoch).

## 7) Suggested Commands
```
python scripts/run_cifar10.py --model small-cnn --epochs 1 --device cuda:0 \
  --step-rule <step_rule> --direction <dir> --clip-mode <clip> --anderson-memory <m> \
  --run-id <run_id>
```
