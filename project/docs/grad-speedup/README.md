# Grad-Speedup Track

This track evaluates algorithmic speedups for gradient-based training without
relying on mainstream implementation tricks (mixed precision, FlashAttention,
or generic step-count tuning). The focus is CIFAR-10 experiments that compare
single modules and module combinations with a shared baseline.

Primary docs:
- CIFAR-10 implementation spec: project/docs/grad-speedup/cifar10-implementation-spec.md
- Experiment brief: project/docs/experiment-20260105-grad-speedup-cifar10.md
- Temp materials record: project/docs/grad-speedup/temp-materials-summary.md
- Method conformance matrix: project/docs/grad-speedup/method-conformance.md
- Delivery plan: project/docs/grad-speedup/plan.md
- Critical path: project/docs/grad-speedup/critical-path.md
- System spec: project/docs/grad-speedup/system-spec.md

Scope notes:
- We compare time-to-target / steps-to-target / cost-to-target, not just final accuracy.
- We keep data, model, and training budgets fixed; only algorithmic modules change.

Isolation
- Grad-speedup code and scripts live under project/grad-speedup/ only.
- Do not import or modify project/src or run_mnist_experiment.py for this track.

Module families (from research materials):
- Module A: compute reduction (e.g., structured sparsity or linearized-Bregman style updates).
- Module B: update direction (e.g., curvature-aware preconditioning).
- Module C: step-size control (e.g., stability- or curvature-aware step rules).
- Module D: external acceleration (e.g., Anderson / nonlinear acceleration).

Artifacts and logs for this track should be written under project/runs/grad-speedup/.
