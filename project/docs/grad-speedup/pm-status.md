# Grad-Speedup PM Status

Date: 2026-01-07
Owner: PM

Current state
- Spec alignment: updated plan/system/combination docs to the CIFAR-10 spec (72-condition grid).
- Base-grid methods: EoSS implemented; screen2000 validation complete; tune2000 sweep running.
- Direction track: Layerwise GN paperpack + reference implementation done; excluded from active experiments due to heavy compute.
- GN-lite: gn-layerwise-exact now supports top/bottom/random-k layer selection with logging for profiling.
- Dashboard: Dash UI in place; supports time/steps/cost-to-target with legend table.

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
- Screen2000 baseline sweep done; tune2000 sweeps running via queue runner.

Next PM actions
- Monitor tune2000 results and promote top configs to 7k/14k runs.
- Start 72-condition grid once generator is ready (if still required).
- Shift dashboard comparisons to time-to-target (drop mean step time as primary).

Risks / blockers
- Grid generator still pending for full 72-condition sweep.
