# Task Ticket: Paper-accurate Layerwise GN (prototype)

## 1) Meta

- ticket_id: task-20260106-grad-speedup-gn-layerwise-paper-accurate
- role/agent: implementer-gn-layerwise-exact
- owner（manager）: codex-pm
- created_at: 2026-01-06
- priority: P1
- status: parked (compute overhead; keep as reference only)
- timebox: 3–4h for prototype + design note
- workspace_scope: project/grad-speedup/
- related
  - paper: arXiv:2510.09378 (full + layerwise GN)

## 2) Goal / Desired Outcome

Implement a **paper-accurate** layerwise GN update as described in arXiv:2510.09378, Section 6.3:
per-layer Taylor expansion, solve second-order Taylor of the loss per-layer, merge updates, apply line search.

### What success looks like
- New direction name (e.g., `gn-layerwise-exact`) that is explicitly paper-accurate.
- Uses per-layer GN (block-diagonal) instead of diagonal Fisher/EMA(g^2).
- Has line-search step for merged update (can reuse existing adaptive-backtracking or a simple Armijo).
- Small smoke run completes without NaN/inf and logs GN stats.

## 3) Background / Context

The current `gn-layerwise` implementation in main is a proxy (diag Fisher/EMA(g^2)). We need a method that
matches the paper’s definition for validation and experiments.

## 4) Scope

### In scope
- Add `gn-layerwise-exact` direction.
- Compute per-layer GN update (block-diagonal) as in the paper.
- Merge per-layer updates and apply a line search.
- Document design in `project/docs/grad-speedup/gn-layerwise-design.md`.

### Out of scope
- Full GN across all layers.
- Large-scale LLM specific optimizations.

## 5) Requirements

### Must
- Paper-accurate per-layer GN definition (Section 6.3, pages 8–9).
- Use G = J^T H_l J for each layer (Section 4.2, page 4).
- Apply a line search after merging per-layer updates (paper uses line search).

### Should
- Provide a minimal, computationally feasible approximation for CIFAR-10 (e.g., low-rank / CG).
- Fall back to proxy only if explicitly flagged and documented as non-accurate.

## 6) Acceptance Criteria

- [ ] `--direction gn-layerwise-exact` runs (CIFAR-10, short smoke).
- [ ] Implementation and method-conformance updated with page references.
- [ ] Design note updated with exact equations + approximations used.

## 7) Implementation Notes

Possible approach:
- For each layer, compute Gv via per-layer Jacobian / loss Hessian (GGN) and solve via CG.
- Use a small number of CG iterations to keep cost bounded.
- Merge updates across layers; run Armijo-style line search on the merged update.

## 8) Commands

```bash
cd project
python grad-speedup/scripts/run_cifar10.py --model resnet18 --epochs 1 --max-steps 200 --batch-size 128 --optimizer sgd --lr 0.1 --seed 0 --device cuda:0 --direction gn-layerwise-exact --run-id smoke-gn-layerwise-exact
```

## 9) Deliverables

- Code changes under `project/grad-speedup/`
- Updated `method-conformance.md` and `gn-layerwise-design.md`
- Smoke run note in `devlog.md`

## 10) Risks / Edge Cases

- GGN computation overhead.
- Numerical instability in per-layer solves.

## 11) Open Questions

- Acceptable approximation level for per-layer solve (CG vs closed form for small layers).
