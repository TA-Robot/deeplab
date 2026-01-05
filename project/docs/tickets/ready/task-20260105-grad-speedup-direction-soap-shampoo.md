# Task Ticket: Grad-Speedup Direction Methods (SOAP + Shampoo)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-direction-soap-shampoo
- role/agent: implementer-grad-speedup-soap-shampoo
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 240 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Implement SOAP and Shampoo per the primary papers with paper-accurate update rules
and preconditioning cadence, suitable for CIFAR-10 experiments.

## 3) Background / Context
Module B requires true direction/preconditioning methods. The existing placeholder
(diag preconditioner) is insufficient. SOAP and Shampoo are prioritized.

## 4) Scope
In scope:
- Implement Shampoo (Kronecker preconditioning) as per arXiv:1802.09568.
- Implement SOAP as per arXiv:2409.11321 (Adam in Shampoo eigenbasis).
- Add config flags for preconditioning update frequency and damping.
- Log per-layer preconditioner stats and overhead.

Out of scope:
- Distributed Shampoo.

## 5) Requirements
Must:
- Conform to paper algorithm steps; do not substitute RMS/Adam approximations.
- Keep module exclusive (one direction method at a time).
- Default behavior unchanged when not enabled.

## 6) Acceptance Criteria
- [ ] SOAP and Shampoo can run on CIFAR-10 smoke run (1 epoch).
- [ ] metrics.jsonl includes preconditioner update counts and timing.
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
python scripts/run_cifar10.py --epochs 1 --run-id smoke-soap --direction soap
```

## 9) Deliverables
- SOAP and Shampoo implementations and docs.

## 10) Risks
- Preconditioner cost; may need small-layer fallback.
