# Grad-Speedup Critical Path (PM)

Purpose
- Visualize dependencies and parallel lanes to minimize total time-to-result.
- Keep paper-accuracy as the gating constraint for implementations.

Critical Path (must complete in order)
1) Paper audit for core methods (EoSS, L0L1, Adaptive Backtracking, GGNC, Anderson, SOAP, Sophia, Muon)
2) Method conformance matrix updated with exact update rules + hyperparameters
3) Paper-accurate implementation for Step-Control (Wave A)
4) Paper-accurate implementation for Direction methods (Wave B)
5) Paper-accurate implementation for GGNC + Anderson (Wave C)
6) Step-based reset + baseline (max_steps=14,000, eval_interval_steps=1000)
7) Step-based winner + ablations (same budget)
8) ResNet18 full sweep mirroring small-cnn stages (step-based, max_steps=7,000)
9) Multi-seed validation on top configs

Parallelization note (2026-01-06)
- Stage4 clip/outer “runner prep” can proceed in parallel with
  the train-metrics triage and GGNC/Anderson implementation.
- Actual Stage4 execution remains gated on:
  - train-metrics fix (non-zero samples/global_step)
  - GGNC + Anderson merged + smoke validated
  - top-1 pair from Stage3 (re-ranked after metrics fix)
9) Final report and decision log

Status snapshot (2026-01-06)
- Step 1: unblocked (paper pack stored under project/docs/grad-speedup/papers/; SAGD corrected to arXiv:2509.14969).
- Step 2: in progress (core methods updated; remaining audits pending).
- Step 3: in progress (step-control updated; SAGD Variant III implemented, pending review).
- Step 4-5: queued (awaiting audited specs).
- Step 6: complete (step-based baseline finished).
- Step 7: in progress (step-based winner running; ablations queued).
- Step 8: queued (resnet18 sweep mirroring small-cnn stages).

Parallel Lanes (run concurrently)
Lane A: Paper audit + conformance updates
- Deliverable: method-conformance.md entries with algorithm steps and parameter definitions.

Lane B: Step-Control implementation (Wave A)
- Depends on: Lane A for step-control methods.
- Runs: baseline + step-control only experiments.

Lane C: Direction methods (Wave B)
- Depends on: Lane A for direction methods.
- Runs: direction-only experiments after build.

Lane D: Stability/Outer acceleration (Wave C)
- Depends on: Lane A for GGNC/Anderson.
- Runs: combinations with best step-control + direction.

Lane E: Sparsity (Wave D)
- Depends on: Lane A for linearized Bregman.
- Runs: compute-reduction combos once method is verified.

Lane F: Reporting + experiment ops
- Runs continuously: baseline runs, report generation, logging, runbook updates.

Lane G: Dashboard
- Build local dashboard for run comparison and diagnostics (independent, can run now).

Gates
- No experiment is accepted unless its method’s paper-accurate spec is present in
  method-conformance.md and validated in code review.
- Legacy placeholder modules may only be used for smoke testing infrastructure.

Current Parallelization Plan
- Paper audit (Lane A) active using local paper pack.
- Step-control implementation queued (Lane B) pending audit completion for remaining methods.
- Direction methods queued (Lane C) pending audit completion for remaining methods.
- GGNC + Anderson queued (Lane D) pending audit completion for remaining methods.
- LinBreg queued (Lane E) awaiting LinBreg paper confirmation.
- Baseline/Smoke runs completed; artifacts are under project/runs/grad-speedup (Lane F).
