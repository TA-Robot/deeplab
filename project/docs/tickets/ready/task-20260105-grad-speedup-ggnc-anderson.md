# Task Ticket: Grad-Speedup Stability + Outer Accel (GGNC + Anderson)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-ggnc-anderson
- role/agent: implementer-grad-speedup-ggnc-anderson
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 240 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Implement GGNC and Anderson acceleration according to their papers, replacing
simplified versions currently in the codebase.

## 3) Background / Context
The existing GGNC and Anderson implementations are minimal and may not match
paper definitions. We need paper-accurate implementations for valid evaluation.

## 4) Scope
In scope:
- GGNC (arXiv:2506.01913): implement generalized clipping operator and step rule.
- Anderson acceleration (arXiv:1809.02341): verify coefficients, residual definition,
  and stabilization (damping/fallback) against the paper.
- Update config flags and logging to match paper parameters.

Out of scope:
- Non-Euclidean norm families not described in the paper.

## 5) Requirements
Must:
- Paper-accurate algorithm steps.
- Default behavior unchanged when disabled.
- Safe fallback for numerical instability.

## 6) Acceptance Criteria
- [ ] GGNC and Anderson run on CIFAR-10 smoke run (1 epoch).
- [ ] metrics.jsonl includes GGNC clip stats and Anderson acceptance/failure counts.
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
python scripts/run_cifar10.py --epochs 1 --run-id smoke-ggnc-anderson --clip-mode ggnc --anderson-memory 3
```

## 9) Deliverables
- Paper-accurate GGNC + Anderson implementations.

## 10) Risks
- GGNC norms may require per-layer norms with extra overhead.
