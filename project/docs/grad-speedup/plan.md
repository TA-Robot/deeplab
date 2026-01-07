# Grad-Speedup Delivery Plan (PM)

Goal
- Deliver a paper-accurate, isolated CIFAR-10 benchmark framework for speedup methods.
- Produce reproducible results with time/steps/cost-to-target metrics and quality guardrails.
- Align experiments to the CIFAR-10 implementation spec (module stacking + 72-condition grid).

Scope
- Work lives under project/grad-speedup and project/docs/grad-speedup.
- No dependency on other experiments or shared code.
- Primary focus: algorithmic modules (step control, geometry/clip, outer accel, sparsity).
- Direction/preconditioning (SOAP/GN/etc) is a separate track, not part of the base 72 grid.

Operating constraints
- FP32 only, fixed augmentation, fixed batch size (128) unless explicitly in a new series.
- Step-based budgets are primary: max_steps is the hard cap; eval is on fixed step cadence.
- Every module must be paper-audited before it is treated as valid.

Phases and tasks (parallelized)

Phase 0: Spec lock + alignment (PM)
- Treat project/docs/grad-speedup/cifar10-implementation-spec.md as the source of truth.
- Update system/plan/combination/critical-path docs to match spec and remove old stage logic.
- Mark SOAP-heavy runs as legacy and out-of-scope for the base grid.

Phase 1: Paper audit and conformance matrix (continuous)
- Core methods (base grid): EoSS step control, Adaptive Backtracking, GGNC, Anderson, LinBreg.
- Track B (direction/preconditioning): Layerwise GN (priority), SOAP (fast check), Full GN (upper bound).
- Output: method-conformance entries with equations, hyperparameters, and constraints.

Phase 2: Implementation tickets (parallel)
- Step-control: implement EoSS (HVP-based curvature with EMA + clipping).
- Geometry/clip: validate GGNC global/layerwise behavior + logging.
- Outer accel: validate Anderson stability/telemetry.
- Sparsity: validate LinBreg path + FLOPs accounting.
- Direction track: implement Layerwise GN and SOAP fast-check harness.

Phase 3: Experiment infrastructure
- Generate 72-condition config grid (SGD/AdamW × step_control × clip × anderson × sparsity).
- Add queue-based runner so runs can be appended and executed sequentially.
- Ensure summary aggregation emits steps/time/cost-to-target per A*.

Phase 4: Execution (step-based)
- Baseline runs (SGD/AdamW) with max_steps and eval_interval_steps=200.
- Single-module tuning (1 seed, max_steps=7000) to fix hyperparameters.
- 72-condition sweep (1 seed, max_steps=7000) for ranking.
- Promote top-10 (max_steps=14000, seeds 0/1/2) for final comparison.
- Direction track (Layerwise GN / SOAP fast check) runs in parallel, but reported separately.

Phase 5: Reporting + dashboard
- Update experiment-log.md with run IDs + status.
- Dashboard must support time-to-target, steps-to-target, cost-to-target, and learning curves.
- No narrative report required; only structured summaries.

Critical-path reference
- See project/docs/grad-speedup/critical-path.md for dependency ordering and lanes.

Decision gates
- No module enters the grid until paper-accurate behavior is verified.
- If baseline fails to reach A*=0.85 within the current promotion budget, adjust baseline before running the grid.
- If A*>=0.90 is required, run a separate “scheduled baseline” series and compare within that regime.
- If any module destabilizes training, it stays out of combinations until fixed.

Next actions (immediate)
- Update spec-aligned docs + create new tickets for EoSS, grid generator, queue runner, Layerwise GN.
- Freeze old SOAP-heavy experiments as legacy and stop using them for decisions.
- Start 1-seed baseline with step-based eval cadence if not already running.
