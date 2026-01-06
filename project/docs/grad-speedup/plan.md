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

Phase 3b: Combination experiments (gated, staged)
- Gate: each method must be paper-audited + smoke-validated before entering combos.
- Stage 1: Step-control sweep (completed; smoke suite done).
- Stage 2: Direction sweep (direction-only, step_rule=none).
- Stage 3: Pairwise combos (Step-control × Direction)
  - Select top-2 step-control methods by cost-to-target at target=0.85 (or lowest loss at 1 epoch).
  - Combine with each direction: {none, diag-precond, shampoo, soap, sophia, muon}.
  - Budget: 1 epoch, 1 seed, small-cnn (GPU).
  - Constraint: adaptive-backtracking, sps-momentum, sagd require direction=none (paper-accurate).
- Stage 4: Add stability/outer (GGNC/Anderson)
  - Use top-1 pair from Stage 3.
  - Sweep clip: {none, ggnc-global, ggnc-layerwise}; outer: {none, anderson}.
  - Budget: 1 epoch, 1 seed, small-cnn (GPU).
- Stage 5: Promote winners to multi-seed + longer epochs
  - 3 seeds, 5 epochs, resnet18.
  - Report cost-to-target and guardrail accuracy.

Critical path reference
- See project/docs/grad-speedup/critical-path.md for dependency ordering and lanes.
- Keep modules exclusive per category.

Phase 4: Verification and baselines (rolling)
- Smoke tests for each method immediately after implementation (CPU).
- Short baseline runs run in the background while new methods are implemented.
- Promote to multi-seed baselines once method correctness is verified.
- Prefer step-based budgets (max_steps=14,000) with eval_interval_steps=1000 for curve fidelity.

Phase 5: Reporting and decisions (per wave)
- Update experiment-log.md and decisions.md.
- Produce summary report CSV/JSON.
- Flag wins/losses and recommend next iterations.

Phase 5b: Promotion + ablation (post-Stage4)
- Promote Stage4 winner to longer runs (resnet18, max_steps=14,000, seeds 0/1/2).
- Run ablations to attribute gains:
  - baseline (SGD)
  - l0l1 only
  - soap only
  - l0l1 + soap (no anderson)
  - l0l1 + soap + anderson (winner)
- Compare accuracy vs time/steps + cost-to-target (step-based).

Phase 6: Dashboard (parallel)
- Build a local dashboard for run comparison and diagnostics.
- Define data schema and derived metrics.
- Provide a Streamlit UI with filtering, overlays, and export.

Phase 7: Follow-up experiments (post-Stage6)
- Sensitivity sweeps on the Stage6 winner:
  - L0/L1 grid (step_l0, step_l1), SOAP update frequency, Anderson memory/interval.
  - Budget: 1,000–2,000 steps, 1 seed on small-cnn; promote top-2 to resnet18 (max_steps=14,000).
- Efficiency profiling:
  - Fixed measurement window (warmup_steps + measure_steps) for baseline vs winner.
  - Report step_time_ms, throughput, and cost-to-target from metrics.jsonl.
- Stability checks:
  - Repeat winner on 2nd data seed to ensure robustness of gains.

Step-based reset plan (Jan 6)
- R0: Clear pre-step-based artifacts (completed).
- R1: Step-based baseline (resnet18, max_steps=14,000, eval_interval_steps=1000, seeds 0/1/2).
- R2: Step-based winner (l0l1+soap+anderson, same budget).
- R3: Step-based ablations (l0l1-only, soap-only, l0l1+soap).
- R4: Update dashboard + compare curves (accuracy vs time/steps, time-to-target).
- R5: Decide whether to proceed to sensitivity sweeps or adjust method configs.

Next experiment plan (step-based)
- Baseline:
  - 20260106-grad-speedup-step-baseline-resnet18-maxsteps14000-seeds012
- Winner:
  - 20260106-grad-speedup-step-l0l1-soap-anderson-resnet18-maxsteps14000-seeds012
- Ablations:
  - 20260106-grad-speedup-step-l0l1-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-soap-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012

Post-ablation plan (next development + experiments)

Gate: baseline + winner + all ablations complete with step-based logs and summaries.

Development tasks (in order)
- D1: Dashboard validation pass on step-based runs
  - Ensure elapsed-time axes populate and plots render (acc vs time/steps).
  - Confirm `none/None` normalization and label/hover behavior.
- D2: Metrics QA + sanity checks
  - Verify eval cadence at 1000 steps.
  - Check targets reached and time-to-target calculations for each run.
- D3: Report minimal comparison table
  - Produce a small CSV/JSON summary (run_id, acc@max_steps, steps/time/cost-to-target).

Experiment plan (after D1–D3)
- E1: Sensitivity sweep (small-cnn, 1 seed, step-based 2,000 steps)
  - L0/L1 grid around (1.0, 0.1)
  - SOAP update frequency sweep (update_every in {1, 5, 10})
  - Anderson memory/interval sweep (m in {0, 3, 5}, interval in {1, 5})
- E2: ResNet18 full sweep mirroring small-cnn stages (step-based)
  - Stage1r: Step-control sweep (max_steps=7,000, 1 seed).
  - Stage2r: Direction sweep (max_steps=7,000, 1 seed).
  - Stage3r: Pairwise combos (max_steps=7,000, 1 seed).
  - Stage4r: Clip/outer sweep (max_steps=7,000, 1 seed).
- E3: Promote top-2 configs from E1/E2 to ResNet18
  - max_steps=14,000, seeds 0/1/2.
- E4: Step-based combo sweep (if E3 shows gains)
  - Re-run Stage3/Stage4 combo logic on ResNet18 with step budgets.
- E5: Stability check
  - Repeat best config with data_seed=456 to validate robustness.

Decision points
- If baseline < target thresholds at max_steps, adjust optimizer schedule or batch size before expanding sweeps.
- If winner gains are not attributable in ablations, pause and re-audit method implementations.

Near-term plan (next 2 weeks)

Week 1 (unblockers + specs)
- Ingest primary papers or provide exact equations for core methods (paper pack now local).
- Upgrade method-conformance.md with paper-accurate update rules + hyperparameters.
- Convert pending tickets into implementation-ready specs (per method family).
- Validate run artifact locations and update experiment-log.md accordingly.

Week 2 (paper-accurate implementations + smoke runs)
- Implement Step-Control (Wave A) and run smoke/baseline validation.
- Implement Direction methods (Wave B) and run smoke checks.
- Implement GGNC + Anderson (Wave C) and run smoke checks.
- Promote top candidates to multi-seed runs; document results.
