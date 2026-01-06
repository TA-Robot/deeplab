# Grad-Speedup PM Status

Date: 2026-01-06
Owner: PM

Current state
- Spec alignment: updated plan/system/combination docs to the CIFAR-10 spec (72-condition grid).
- Base-grid methods: Backtracking/GGNC/Anderson/LinBreg exist but need spec validation; EoSS implementation pending.
- Direction track: Layerwise GN ticket created; SOAP heavy runs treated as legacy and out-of-scope for base grid.
- Dashboard: Dash UI in place; supports time/steps/cost-to-target with legend table.

Primary docs
- CIFAR-10 spec: project/docs/grad-speedup/cifar10-implementation-spec.md
- Plan: project/docs/grad-speedup/plan.md
- Combination plan: project/docs/grad-speedup/combination-plan.md
- Critical path: project/docs/grad-speedup/critical-path.md
- Method conformance: project/docs/grad-speedup/method-conformance.md

Tickets (running)
- EoSS step-control implementation (HVP + EMA)
- 72-condition grid config generator
- Queue runner update (grid support)
- Layerwise GN implementation (separate direction track)

Experiments
- Legacy SOAP-focused runs are no longer used for decision-making.
- Base-grid runs have not started yet (waiting on EoSS + grid generator).

Next PM actions
- Create/assign sub-agent tickets for EoSS, grid generator, queue runner, Layerwise GN.
- Update experiment-log.md with new base-grid run plan once configs exist.
- Kick off baseline runs (SGD/AdamW) with max_steps + eval_interval_steps=1000.

Risks / blockers
- EoSS implementation is the critical blocker for the base grid.
- LinBreg and GGNC need paper-conformance verification before inclusion in grid.
