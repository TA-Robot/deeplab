# Task Ticket: Grad-Speedup Optimizer Hooks

## 1) Meta
- ticket_id: task-20260105-grad-speedup-optimizer
- role/agent: implementer-grad-speedup-optimizer
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 120 min for first working slice
- workspace_scope: project/grad-speedup/
- related:
  - issue: n/a
  - pr/branch: n/a
  - commits: n/a

## 2) Goal / Desired Outcome
Add modular optimizer/step-size control hooks to support the grad-speedup CIFAR-10 experiment
in the isolated grad-speedup codebase (no ROS-ALTH dependencies). The initial focus is
Module C (step-size control) with 1-2 variants that are mathematically motivated and easy
to reproduce.

Success looks like:
- New CLI flags for step-size control (Module C) are available in the grad-speedup runner.
- Training logs include step-time, throughput, and any new module metrics (grad norm, curvature).
- No changes to default behavior when new flags are not set.

## 3) Background / Context
We are creating a new experiment track (grad-speedup) that evaluates algorithmic speedups.
See project/docs/grad-speedup/cifar10-implementation-spec.md for requirements.

## 4) Scope
In scope:
- Add Module C implementations (1-2 variants) and integrate into the training loop.
- Add config/CLI plumbing for the new module(s).
- Extend metrics logging to capture module-specific statistics.

Out of scope:
- Implementing full Shampoo/SOAP/Muon or large new model architectures.
- Updating dashboard/reporting (handled by another ticket).

## 5) Requirements
Must:
- Default behavior is unchanged when new flags are not provided.
- New module(s) must be reproducible with fixed seeds.
- Metrics include grad norm and any curvature proxy if computed.

Should:
- Step-size control uses per-step stats with low overhead.

Nice:
- Add a small unit-style check in code (asserts) for numeric stability.

## 6) Acceptance Criteria
- [ ] Baseline CLI runs unchanged (no behavior change without new flags).
- [ ] New CLI flags appear in config.json and are logged.
- [ ] metrics.jsonl includes grad_norm (and optional curvature field) for train split.
- [ ] New module variant(s) can be enabled via CLI and run without errors.

## 7) Implementation Notes
Suggested approach:
- Implement Module C as an optional step-size rule in project/grad-speedup/src/train.py.
- Add flags to project/grad-speedup/scripts/run_cifar10.py to select a step rule and parameters.

Recommended Module C variants:
- L0-L1 smooth step scaling: lr_eff = lr / (L0 + L1 * grad_norm)
- EoSS-like stability rule: estimate directional curvature via HVP every N steps
  and cap lr based on 2 / curvature (use epsilon guards)

Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py
- (optional) new helper module under project/grad-speedup/src/

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --run-id smoke-grad-speedup
```

## 9) Deliverables
- Code changes implementing Module C hooks and CLI flags.
- Notes on parameter defaults for reproducibility.

## 10) Risks / Edge Cases
- HVP computation can be slow; use low frequency or small sample.
- Curvature estimates can be zero or negative; guard with eps and fallbacks.

## 11) Open Questions
- Which Module C variant should be the default for v1?
- Do we want step-size rules to apply to both SGD and Adam?

## 12) Constraints / Guardrails
- Allowed paths: project/ only.
- Dependency changes: not allowed without approval.
- No destructive operations.

## 13) Reporting
- What I changed
- Evidence (command run or log excerpt)
- Next
- Blockers
