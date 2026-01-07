# Grad-Speedup Combination Experiment Plan (Spec-Aligned)

Purpose
- Enumerate and evaluate all combinable modules in a controlled grid.
- Keep the base grid free of heavy direction/preconditioning methods (handled in a separate track).

Gates (must pass before entering grid)
- Paper-audited spec in project/docs/grad-speedup/method-conformance.md
- Paper-accurate implementation merged + smoke run logged in project/docs/experiment-log.md

Base grid definition (72 conditions)
- Base optimizer: {SGD+Momentum, AdamW} (2)
- Step control: {None, EoSS, Adaptive Backtracking} (3)
- Geometry/clip: {None, GGNC global, GGNC layerwise} (3)
- Anderson: {None, On} (2)
- Sparsity: {None, LinBreg} (2)
- Total: 2 * 3 * 3 * 2 * 2 = 72

Execution stages
Stage A: Single-module tuning (1 seed)
- Fix hyperparameters for EoSS, Backtracking, GGNC, Anderson, LinBreg.
- Budget: max_steps=7000, eval_interval_steps=200.
- Output: recommended hyperparameter defaults for grid.

Stage B: 72-condition sweep (1 seed)
- Budget: max_steps=7000, eval_interval_steps=200.
- Rank by time-to-target at A* = 0.80 and 0.85 (both reported); also report 0.90+ if any runs hit it.

Stage C: Promotion (multi-seed)
- Top-10 configs from Stage B.
- Budget: max_steps=14000, seeds {0,1,2}.
- Output: mean/std for steps/time/cost-to-target at A* thresholds (primary: 0.85; secondary: 0.90+).

Separate track (direction / preconditioning)
- Layerwise GN and SOAP fast-check run in parallel to Stage A/B.
- These are not part of the base 72 grid; report separately and only combine later if stable.

Run naming
- `YYYYMMDD-grad-speedup-grid-<optimizer>-<step>-<clip>-<anderson>-<sparsity>-<model>-seed<seed>`
- Example: `20260106-grad-speedup-grid-sgd-eoss-ggnc-layerwise-anderson-linbreg-resnet18-seed0`

Notes
- Do not mix multiple step-control methods in one run.
- Do not combine multiple direction/preconditioning methods.
- All runs must log per-target steps/time/cost + step-time statistics.
- Baseline regime: for fixed-LR/no-schedule comparisons, use SGD momentum=0.0 (mom0) as the baseline; keep “scheduled baseline” runs separate.
