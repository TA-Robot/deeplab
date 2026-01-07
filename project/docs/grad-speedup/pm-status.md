# Grad-Speedup PM Status

Date: 2026-01-07
Owner: PM

Current state
- Spec alignment: updated plan/system/combination docs to the CIFAR-10 spec (72-condition grid).
- Baseline: SGD+mom=0.9 was weak under fixed-LR/no-schedule; added mom=0.0 baseline which is substantially stronger (now use mom=0.0 as the default baseline for step-control comparisons).
- Base-grid methods: current evidence suggests EoSS is not competitive in this fixed-LR/no-schedule regime; GGNC(alpha=1) and LinBreg are viable candidates for combination sweeps.
- Direction track: Layerwise GN / GN-lite experiments were run for profiling but are currently deprioritized (compute overhead dominates).
- Dashboard: Dash UI in place; supports time/steps/cost-to-target and learning curves, with legend table + hover-disable controls.

Primary docs
- CIFAR-10 spec: project/docs/grad-speedup/cifar10-implementation-spec.md
- Plan: project/docs/grad-speedup/plan.md
- Combination plan: project/docs/grad-speedup/combination-plan.md
- Critical path: project/docs/grad-speedup/critical-path.md
- Method conformance: project/docs/grad-speedup/method-conformance.md

Tickets (running)
- None (Layerwise GN track parked after compute-cost review)

Experiments
- Legacy SOAP-focused runs are no longer used for decision-making.
- Momentum ablation completed; mom0 baseline is adopted for the no-schedule regime.

Next PM actions
- Re-run the active comparison series with a tighter eval cadence (eval_interval_steps=200) to make time-to-target curves usable.
- Start 72-condition grid once the stage-A tuning defaults are fixed (or reduce the grid if the tuning indicates clear dominance relations).
- Shift dashboard comparisons to time-to-target (drop mean step time as primary).
- Drop l0l1-only as a primary candidate unless it beats mom=0.0 baseline (current data suggests it doesn't).
- Treat GGNC alpha<1.0 as unsafe (alpha=0.2 failed); keep alpha=1.0 until re-derived from paper.

Risks / blockers
- Grid generator still pending for full 72-condition sweep.
