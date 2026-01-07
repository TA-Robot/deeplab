# Task Ticket: Layerwise Gauss–Newton (implementation)

## 1) Meta

- ticket_id: task-20260106-grad-speedup-layerwise-gn
- role/agent: implementer-layerwise-gn
- owner（manager）: codex-pm
- created_at: 2026-01-06
- priority: P1
- timebox: 180min for first working prototype
- workspace_scope: project/grad-speedup/
- related
  - issue: add layerwise GN preconditioning to grad-speedup
  - depends_on: task-20260106-grad-speedup-gn-paperpack

## 2) Goal / Desired Outcome

Implement a layerwise Gauss–Newton (GGN) preconditioner that can be enabled as a module in the CIFAR-10 runner.
The update must be paper-accurate and plug into the existing modular optimizer pipeline.

### What success looks like
- A new module (or direction wrapper) performs layerwise GN preconditioning and can be toggled via CLI/config.
- Logging includes GN diagnostics (e.g., damping, per-layer scale stats).
- A small smoke run on CIFAR-10 completes without NaN/inf and emits GN stats.
- Paper-accurate equations are referenced in method-conformance.

## 3) Background / Context

The temp investigation prioritizes GN (full and layerwise) as a candidate to reduce steps-to-target.
We need a pragmatic layerwise variant before tackling full GN.

## 4) Scope

### In scope
- Add a layerwise GN/GGN preconditioner implementation.
- CLI/config wiring to enable it (e.g., `--gn-layerwise` or `--direction gn-layerwise`).
- Logging of key stats.
- Minimal smoke run instructions.

### Out of scope
- Full (cross-layer) GN.
- New experiments beyond a short smoke.

## 5) Requirements

### Must
- Paper-accurate update rule (cite the local GN PDF added by the paperpack task).
- Per-layer preconditioner (block-diagonal / layerwise approximation).
- Damping parameter and stability guard (e.g., epsilon or clipping).
- Works with ResNet-18 CIFAR-10 training loop.

### Should
- Support configurable update frequency (e.g., every K steps).
- Keep overhead bounded (avoid full per-parameter matrix ops).

### Nice
- Optional low-rank approximation for large layers (if trivial).

## 6) Acceptance Criteria

- [ ] New module wired via config + CLI.
- [ ] `run_cifar10.py` can enable it without code changes.
- [ ] Smoke run completes and logs GN stats.
- [ ] `method-conformance.md` references the GN paper and update.

## 7) Implementation Notes

Suggested approach:
- Use empirical Fisher / GGN approximation per layer as described in the GN paper.
- Use diagonal or low-rank approximation per layer to keep cost manageable.
- Insert into existing modular pipeline in `project/grad-speedup/src/modules.py` (or similar).
- Update `project/grad-speedup/src/train.py` to compute the preconditioned update.

## 8) Commands

```bash
cd project

# smoke (example)
python grad-speedup/scripts/run_cifar10.py --model resnet18 --epochs 1 --max-steps 200 --batch-size 128 --optimizer sgd --lr 0.1 --seed 0 --device cuda:0 --run-id smoke-gn-layerwise
```

## 9) Deliverables

- Code changes under `project/grad-speedup/`
- Updated config + CLI docs (if needed)
- Short note in `project/docs/grad-speedup/devlog.md` with the smoke command

## 10) Risks / Edge Cases

- Incorrect curvature estimation leads to instability; ensure damping + clipping.
- Added overhead may overwhelm step time; log cost stats.

## 11) Open Questions

- Confirm which GN paper / equations to follow (must match paperpack task).

## 12) Constraints / Guardrails

- Allowed paths: project/grad-speedup/ only
- Dependency changes: require manager approval

## 13) Reporting

- Report after implementation and after smoke run with logs and key changes.
