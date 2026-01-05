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
6) Baseline + single-module experiments (per wave)
7) Multi-seed validation on top configs
8) Final report and decision log

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

Gates
- No experiment is accepted unless its method’s paper-accurate spec is present in
  method-conformance.md and validated in code review.
- Legacy placeholder modules may only be used for smoke testing infrastructure.

Current Parallelization Plan
- Paper audit (Lane A) in progress.
- Step-control implementation queued (Lane B) awaiting audit notes.
- Direction methods queued (Lane C) awaiting audit notes.
- GGNC + Anderson queued (Lane D) awaiting audit notes.
- LinBreg queued (Lane E) awaiting audit notes.
- Baseline/Smoke runs in background (Lane F).
