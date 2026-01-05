# Grad-Speedup Delivery Plan (PM)

Goal
- Deliver a paper-accurate, isolated CIFAR-10 benchmark framework for speedup methods.
- Produce reproducible results with time/steps/cost-to-target metrics and quality guardrails.

Scope
- Work lives under project/grad-speedup and project/docs/grad-speedup.
- No dependency on other experiments or shared code.

Phases and tasks (parallelized)

Phase 0: Inventory and scope lock
- Confirm isolation rules and directory layout.
- Freeze target method list (from temp materials).
- Define experiment acceptance criteria and budget.

Phase 1: Paper audit and conformance matrix (continuous)
- For each method, capture:
  - Primary paper(s) and venue/year
  - Algorithm steps and equations
  - Required hyperparameters and defaults
  - Computational/memory complexity
  - Known caveats / stability notes
- Produce a method conformance matrix with implementation requirements.

Phase 2: Detailed design tickets (continuous)
- One ticket per method or method family.
- Each ticket includes:
  - spec summary (equations, update rules)
  - config schema updates
  - logging/metrics requirements
  - unit/smoke tests
  - integration dependencies

Phase 3: Implementation (paper-accurate) + overlapping runs
- Wave A (baseline + step-control): implement step-control methods, run baseline and single-module runs in parallel.
- Wave B (direction): implement SOAP/Shampoo, then Sophia/Muon; run single-module direction experiments while Wave C is built.
- Wave C (stability/outer): implement GGNC + Anderson; run combinations with the best step-control and direction choice.
- Wave D (sparsity): implement Linearized Bregman; run compute-reduction combos.

Critical path reference
- See project/docs/grad-speedup/critical-path.md for dependency ordering and lanes.
- Keep modules exclusive per category.

Phase 4: Verification and baselines (rolling)
- Smoke tests for each method immediately after implementation (CPU).
- Short baseline runs run in the background while new methods are implemented.
- Promote to multi-seed baselines once method correctness is verified.

Phase 5: Reporting and decisions (per wave)
- Update experiment-log.md and decisions.md.
- Produce summary report CSV/JSON.
- Flag wins/losses and recommend next iterations.
