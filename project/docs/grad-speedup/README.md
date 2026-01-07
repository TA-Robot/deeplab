# Grad-Speedup Track

This track evaluates algorithmic speedups for gradient-based training without
relying on mainstream implementation tricks (mixed precision, FlashAttention,
or generic step-count tuning). The focus is CIFAR-10 experiments that compare
single modules and module combinations with a shared baseline.

Primary docs
- CIFAR-10 implementation spec: project/docs/grad-speedup/cifar10-implementation-spec.md
- Experiment brief: project/docs/experiment-20260105-grad-speedup-cifar10.md
- Temp materials record: project/docs/grad-speedup/temp-materials-summary.md
- Method conformance matrix: project/docs/grad-speedup/method-conformance.md
- Delivery plan: project/docs/grad-speedup/plan.md
- Combination plan: project/docs/grad-speedup/combination-plan.md
- Critical path: project/docs/grad-speedup/critical-path.md
- PM status: project/docs/grad-speedup/pm-status.md
- Devlog: project/docs/grad-speedup/devlog.md
- System spec: project/docs/grad-speedup/system-spec.md
- Tickets (grad-speedup only): project/docs/grad-speedup/tickets/ready/
- Paper pack: project/docs/grad-speedup/papers/README.md
- Dashboard spec: project/docs/grad-speedup/dashboard-spec.md
- Dashboard implementation (Dash): project/grad-speedup/dash/app.py
- Legacy Streamlit dashboard (deprecated): project/grad-speedup/dashboard/README.md

Scope notes
- Primary metrics are time-to-target / steps-to-target / cost-to-target.
- Base grid is a 72-condition sweep (SGD/AdamW × step-control × clip × anderson × sparsity).
- Direction/preconditioning methods (SOAP/GN/etc) are tracked separately.

Isolation
- Grad-speedup code and scripts live under project/grad-speedup/ only.
- Do not import or modify project/src or run_mnist_experiment.py for this track.
- Artifacts and logs are written under project/runs/grad-speedup/.

Module families
- Base optimizer: SGD+Momentum or AdamW.
- Step control: None, EoSS, Adaptive Backtracking (Silver optional).
- Geometry/clip: GGNC (global or layerwise).
- Outer accel: Anderson.
- Sparsity: LinBreg.
- Direction/preconditioning (SOAP/GN) is a separate track.
