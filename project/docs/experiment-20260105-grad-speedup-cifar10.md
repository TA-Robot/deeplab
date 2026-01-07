# Experiment Brief: 20260105-grad-speedup-cifar10

ID: 20260105-grad-speedup-cifar10
Owner: TBD
Date: 2026-01-05
Hypothesis:
  Algorithmic modules that control step size or curvature (Module C and/or B)
  reduce cost-to-target on CIFAR-10 by >= 1.5x without > 0.5 pp accuracy loss.
Baseline:
  Commit: TBD (grad-speedup isolated codebase)
  Config: CIFAR-10, batch 128, seeds 0,1,2, FP32, fixed augmentation,
          baseline model from project/grad-speedup/src with SGD (momentum 0.0 for fixed-LR/no-schedule) and Adam.
Change:
  Add modular step-size control (Module C) and/or curvature-aware updates (Module B)
  per project/docs/grad-speedup/cifar10-implementation-spec.md.
Primary metric:
  Time-to-target W(A*) and steps-to-target T(A*) for A* in {0.80, 0.85, 0.90, 0.92, 0.94}.
Quality guardrail:
  Test accuracy within 0.5 pp of baseline at matched target thresholds.
Acceptance criteria:
  - At least one A* achieves >= 1.5x cost-to-target improvement vs baseline.
  - No training instability (NaNs, divergence).
  - Accuracy drop <= 0.5 pp at target.
  - 3 seeds; report mean and std.
Budget:
  GPU only, max wall clock 6 hours total.
Notes:
  - Use FP32 only; no mixed precision or kernel-level optimizations.
  - Data augmentation fixed: RandomCrop(32, padding=4), RandomHorizontalFlip(p=0.5).
  - Early stop once highest A* threshold is reached (or use “first” mode for screening).
  - Keep this track isolated from ROS-ALTH code paths.
