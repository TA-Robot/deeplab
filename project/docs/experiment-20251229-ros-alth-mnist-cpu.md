# Experiment Brief: 20251229-ros-alth-mnist-cpu

ID: 20251229-ros-alth-mnist-cpu
Owner: TBD
Date: 2025-12-29
Hypothesis:
  Operator Basis Layer (K=59, full mixture with per-operator Norm)
  is stable on CPU MNIST and achieves comparable accuracy with
  controlled runtime overhead.
Baseline:
  Two CPU MNIST baselines with identical optimizer/batch size:
  - MLP baseline
  - CNN baseline
Change:
  Insert multi-layer Operator Basis Layers after each hidden block using the
  full operator library from ROS-ALTH_02 (including RBF, attention, softsort,
  sinkhorn, diffusion, and random program composition).
Runs:
  - MLP baseline vs MLP+OBL (full library)
  - CNN baseline vs CNN+OBL (full library)
Primary metric:
  Step time (ms/step) and throughput (samples/sec) on CPU.
Quality guardrail:
  Test accuracy >= baseline - 0.5% (target >= 97%).
Acceptance criteria:
  - No training instability (loss divergence or NaNs).
  - Accuracy >= 97% and within 0.5% of baseline.
  - Runtime overhead <= 1.5x baseline step time.
  - 3 seeds; report mean and std.
Resource budget:
  CPU only, max wall clock 2 hours total.
Notes:
  - Use fixed seeds and record environment (CPU model, BLAS, OS).
  - Keep K=59 only (no growth schedule in this first run).
  - Initialize new betas near 0 and set small residual scale gamma.
  - Keep dataset and preprocessing identical across runs.
  - Use train/val/test split and only evaluate test at the end.
  - Default architectures:
    - MLP: 784-256-256-10 (ReLU + dropout)
    - CNN: 2x conv(32/64) + fc(256) + fc(128) + dropout
  - OBL profile: full (multi-family + random programs)
  - Record parameter counts for each run; if gaps are large, adjust hidden dims and rerun.
Protocol:
  - Fixed train/val split seed for all runs
  - 3 seeds per configuration, report mean/std
  - No hyperparameter tuning on test metrics
  - Log configs, environment, and per-epoch metrics to `project/runs/`

References:
- project/docs/ros-alth/ROS-ALTH_01_overview.md
- project/docs/ros-alth/ROS-ALTH_02_operator_library.md
- project/docs/ros-alth/ROS-ALTH_03_training_evaluation.md

Operator library coverage (full profile):
- Frequency: shared RFF sin/cos over multiple scales
- Polynomial + rational + gate operators
- Diffusion + Gaussian blur
- RBF prototypes and soft neighbor aggregation
- SoftSort and Sinkhorn relaxations
- Log-sum-exp soft pooling
- Random program composition (depth 2-4 with residual skips)
