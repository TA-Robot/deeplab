# CIFAR-10 Implementation Spec (Grad-Speedup Track)

Goal
- Implement and compare algorithmic speedups for gradient methods as modular components.
- Evaluate single modules and combinations on CIFAR-10 using shared baselines.
- Focus on time-to-target / steps-to-target / cost-to-target rather than best final accuracy.

Cost model
- Total cost C = T * c
  - T: steps-to-target (optimizer steps to reach a target accuracy)
  - c: average cost per step (wall time or FLOPs proxy)

Primary metrics (must report)
- Steps-to-target: T(A*) for A* in {0.85, 0.90, 0.92, 0.94}
- Time-to-target: W(A*) (wall time to reach A*)
- Cost-to-target: C(A*) = T(A*) * mean step time
- Train/val curves: loss, accuracy, step time, throughput

Secondary metrics (recommended)
- Peak GPU memory
- Gradient norm stats (mean, p50, p90)
- Optional curvature proxy if used (e.g., directional curvature from HVP)

Fixed conditions
- Precision: FP32 only (no mixed precision)
- Data augmentation (train only): RandomCrop(32, padding=4), RandomHorizontalFlip(p=0.5)
- Normalize: CIFAR-10 mean/std
- Determinism: fixed seeds; record determinism flag and library versions
- Budget: max epochs 200 (or max steps 100k), early stop when highest A* reached

Dataset
- CIFAR-10: train 50,000 / test 10,000; val split from train (fixed seed)
- Batch size: 128 (optional secondary run at 256 or 512 as separate experiment)
- DataLoader: pin_memory=True, drop_last=True, num_workers fixed per run

Model
- Primary baseline: clean-slate CIFAR-10 model defined under project/grad-speedup/src
  (e.g., ResNet-18 CIFAR variant or a small CNN)
- Optional follow-up: add a second baseline model in the same isolated codebase

Module families (composable)
- Module A (compute reduction): structured sparsity or linearized-Bregman style updates
  - Initial status: planned, not in v1
- Module B (update direction): curvature-aware or preconditioned updates
  - Candidates: Shampoo/SOAP/Muon/Sophia (v1 should pick one feasible option)
- Module C (step-size control): stability- or curvature-aware step rules
  - Candidates: EoSS/Batch-Sharpness step rule, L0-L1 smooth step scaling, adaptive line search
- Module D (external acceleration): Anderson / nonlinear acceleration
  - Initial status: planned, add after C is stable

Experiment grid (v1)
- Baseline: SGD (momentum 0.9) and Adam (no scheduler)
- Module C only: baseline + step-size control (1-2 variants)
- Module B only: baseline replaced with a curvature-aware update (if implemented)
- Module B + C: combined

Logging requirements
- Save config.json, env.json, metrics.jsonl, summary.json
- summary.json must include steps-to-target and time-to-target per threshold
- Include run_id format: YYYYMMDD-grad-speedup-cifar10-<variant>
- Output root: project/runs/grad-speedup/

Acceptance criteria (initial)
- For at least one A* in {0.90, 0.92}: cost-to-target improves by >= 1.5x vs baseline
- Accuracy drop at target <= 0.5 pp
- No training instability (NaNs, divergence)

Notes
- Avoid mainstream optimizations (AMP, FlashAttention, generic LR schedule tuning).
- Keep the grad-speedup codebase isolated; do not reuse project/src or run_mnist_experiment.py.
