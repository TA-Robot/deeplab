# Task Ticket: Grad-Speedup EoSS Step-Control

## 1) Meta
- ticket_id: task-20260106-grad-speedup-eoss-step-control
- role/agent: implementer-grad-speedup-eoss
- owner: PM
- created_at: 2026-01-06
- priority: P0
- timebox: 4h
- workspace_scope: project/grad-speedup/
- related:
  - spec: project/docs/grad-speedup/cifar10-implementation-spec.md
  - conformance: project/docs/grad-speedup/method-conformance.md

## 2) Goal / Desired Outcome
Add EoSS step-control as a selectable step_rule so the base 72-condition grid can run.

Success:
- `--step-rule eoss` works and changes the effective step size.
- HVP-based curvature estimate is computed every K steps, EMA-smoothed, and clipped.
- Metrics log includes step_size and curvature stats for EoSS runs.

## 3) Background / Context
The spec defines EoSS as the base grid step-control rule. It is currently missing in code.
We must implement it before grid generation and baseline/grid runs start.

## 4) Scope
In scope:
- Add `eoss` to SUPPORTED_STEP_RULES and CLI/config schema.
- Implement curvature estimation via HVP along gradient direction.
- Implement EMA smoothing + clipping to produce an effective step size.
- Log curvature and step_size statistics.

Out of scope:
- Changing other step rules.
- Any direction/preconditioning methods.

## 5) Requirements
Must:
- Step-rule `eoss` is available in CLI/config.
- HVP computation uses the current mini-batch gradients.
- Step size rule: eta = beta * 2 / (s_hat + eps), clipped to [eta_min, eta_max].
- Update frequency controlled by `step_eoss_interval`.

Should:
- Store state in optimizer param_group (EMA, last curvature).
- Respect `--grad-norm-every` if used for logging.

## 6) Acceptance Criteria
- [ ] `python project/grad-speedup/scripts/run_cifar10.py --step-rule eoss ...` runs without error.
- [ ] step_size values change over time and remain within clip bounds.
- [ ] metrics.jsonl includes curvature and step_size entries for train/epoch.
- [ ] No regressions for existing step rules.

## 7) Implementation Notes
Suggested approach:
- Add new CLI args: `--step-eoss-beta`, `--step-eoss-ema`, `--step-eoss-interval`,
  `--step-eoss-eps`, `--step-eoss-clip-min`, `--step-eoss-clip-max`.
- In `train_one_epoch`, when step_rule == "eoss":
  - Compute gradient g (already available).
  - Every K steps: compute HVP along g (using autograd.grad with grad_outputs=g).
  - Compute s = (g·H g) / (||g||^2 + eps).
  - Update EMA s_hat.
  - Compute eta and set group lr accordingly (store base_lr in group).
- Add curvature logging to StepLog and TrainMetrics (mean/p50/p90).

Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/src/modules.py
- project/grad-speedup/scripts/run_cifar10.py

## 8) Commands
```bash
cd project/grad-speedup
python scripts/run_cifar10.py --step-rule eoss --max-steps 2000 --seed 0 --device cuda:0
```

## 9) Deliverables
- Code changes implementing EoSS step-control.
- Brief note in method-conformance if behavior deviates from spec.

## 10) Risks / Edge Cases
- HVP overhead may be high; ensure low frequency defaults.
- HVP can be unstable if gradients are zero; guard with eps and fall back.

## 11) Open Questions
- Should HVP use the same batch or a smaller micro-batch for stability?
- Default clip bounds for eta (use conservative defaults).

## 12) Constraints / Guardrails
- No new dependencies.
- Keep changes minimal and isolated.
