# Task Ticket: GN-lite layer selection (top/bottom/random k)

## 1) Meta

- **ticket_id**: `task-20260106-grad-speedup-gn-lite-layer-select`
- **role/agent**: `implementer-gn-lite-layer-select`
- **owner（manager）**: `codex-pm`
- **created_at**: 2026-01-06
- **priority**: P1
- **timebox**: 2–3h for implementation + smoke
- **workspace_scope**: project/grad-speedup/
- **related**
  - parent: `task-20260106-grad-speedup-gn-layerwise-paper-accurate` (reference impl exists)

## 2) Goal / Desired Outcome

Add **GN-lite** variants that apply the layerwise Gauss–Newton update to only a subset of layers,
to reduce wall-clock cost while retaining some second-order benefits.

These are **not** paper-accurate and must be labeled as experimental.

### What success looks like

- New selector options for `gn-layerwise-exact` (or a separate `gn-layerwise-lite`) to choose:
  - **top‑k layers**, **bottom‑k layers**, or **random‑k layers per step**.
- Non-selected layers fall back to standard SGD gradient update.
- Deterministic randomness (seeded by run seed + global_step) for the random‑k mode.
- Logging captures which layers were selected and GN timing stats per step.

## 3) Background / Context

Full `gn-layerwise-exact` is too slow for experiments (per-step cost in seconds).
We need a pragmatic, controlled subset approach to test whether GN benefits can be captured cheaply.

## 4) Scope

### In scope

- Add CLI/config args for layer selection:
  - `--gn-layer-mode` in {`all`, `topk`, `bottomk`, `randomk`}
  - `--gn-layer-k` (int)
  - `--gn-layer-random-every-step` (bool, default True)
- Define “layer” as **parameter tensors with ndim >= 2** (conv/linear weights),
  ordered by model parameter order.
- Apply GN only to selected layers; others use their plain gradients.
- Log selection metadata per layer (e.g., `gn_selected`, `gn_layer_mode`).
- Update README + method-conformance to mark GN‑lite as experimental (not paper-accurate).

### Out of scope

- Full GN or layerwise GN performance claims.
- Any claim of paper alignment for GN‑lite.

## 5) Requirements

### Must

- Deterministic random‑k selection per step (seed = run seed + global_step).
- GN applied only to selected layers; non-selected layers are unchanged SGD grads.
- Works with CIFAR‑10 runner.
- Safe defaults: `gn-layer-mode=all` keeps existing behavior.

### Should

- Allow excluding 1D params (bias/BN) from candidate list.
- Log selected layer count per step.

## 6) Acceptance Criteria

- [ ] `python grad-speedup/scripts/run_cifar10.py ... --direction gn-layerwise-exact --gn-layer-mode topk --gn-layer-k 5`
      runs for max_steps=1 on CUDA without error.
- [ ] `randomk` mode produces different selected layers across steps but is deterministic for a fixed seed.
- [ ] README + method-conformance updated to describe GN‑lite (experimental).

## 7) Implementation Notes

Suggested approach:
- Build `eligible_params = [(name, param, grad)]` where `param.ndim >= 2`.
- Compute `selected_names` based on `gn_layer_mode`:
  - topk: last k in eligible list
  - bottomk: first k in eligible list
  - randomk: sample k using local RNG seeded by (seed + global_step)
- Use selected list for GN solve; for others, set `param.grad = grad.detach()`.

## 8) Commands

```bash
cd project
python grad-speedup/scripts/run_cifar10.py --model resnet18 --epochs 1 --max-steps 1 --batch-size 32 \
  --optimizer sgd --lr 0.1 --seed 0 --device cuda:0 \
  --direction gn-layerwise-exact --gn-layer-mode topk --gn-layer-k 5 --run-id smoke-gn-lite-topk
```

## 9) Deliverables

- Code changes under `project/grad-speedup/`
- Docs update: `project/docs/grad-speedup/method-conformance.md` and `project/grad-speedup/README.md`
- Note in `project/docs/grad-speedup/devlog.md`

## 10) Risks / Edge Cases

- Random selection could destabilize training; must be reproducible.
- If `gn-layer-k` > eligible layers, fall back to all eligible.

## 11) Open Questions

- Confirm definition of “top/bottom” (model depth vs parameter order) if ambiguous.

## 12) Constraints / Guardrails

- **Allowed paths**: project/grad-speedup/ only
- **Dependency changes**: Not allowed
- **Dangerous operations**: None

## 13) Reporting

- Provide changes list + smoke command output.
- Note any surprises around layer selection or determinism.
