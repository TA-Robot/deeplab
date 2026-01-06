# Grad-Speedup Critical Path (PM, Spec-Aligned)

Purpose
- Visualize dependencies and parallel lanes to minimize time-to-result.
- Paper-accuracy is the gating constraint for all methods.

Critical Path (must complete in order)
1) Spec lock (CIFAR-10 implementation spec is source of truth)
2) Method conformance for base grid methods (EoSS, Backtracking, GGNC, Anderson, LinBreg)
3) Implement EoSS step control (HVP-based curvature) + smoke validation
4) Grid config generator for 72 conditions + queue runner
5) Step-based baseline runs (SGD/AdamW; eval_interval_steps=1000)
6) 1-seed 72-condition sweep (max_steps=7000)
7) Promote top-10 configs to 3-seed runs (max_steps=14000)
8) Decision on winners + follow-up hyperparameter sweeps

Parallel lanes (run concurrently)
Lane A: Paper audit + method conformance updates
- Deliverable: method-conformance.md entries with equations and constraints.

Lane B: Step-control implementation
- EoSS step rule, plus validation of Backtracking settings.

Lane C: Geometry/Outer/Sparsity validation
- GGNC, Anderson, LinBreg correctness + logging.

Lane D: Grid infrastructure
- Config generator, queue runner, and aggregation scripts.

Lane E: Direction track (separate)
- Layerwise GN implementation and SOAP fast-check harness.
- Reported separately; no coupling to base grid unless stable.

Lane F: Reporting + dashboard
- Dashboard must show time-to-target and learning curves without legend overlap.

Gates
- No experiment is accepted unless the method’s paper-accurate spec is present in method-conformance.md.
- Direction/preconditioning methods are excluded from the base grid until validated.

Status snapshot (2026-01-06)
- Spec alignment: in progress (plan/system/combination docs updated).
- Base-grid methods: EoSS implementation pending; Backtracking/GGNC/Anderson/LinBreg exist but need spec validation.
- Grid infra: generator + queue runner pending update to 72-condition grid.
- Direction track: Layerwise GN ticket created; SOAP heavy runs deprecated.
