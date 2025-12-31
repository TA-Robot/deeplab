# Experiment Brief: 20251230-ros-alth-gpu-mnist

ID: 20251230-ros-alth-gpu-mnist
Owner: TBD
Date: 2025-12-30
Hypothesis:
  On GPU, the full Operator Basis Layer (K=59) remains stable on MNIST and
  keeps accuracy parity while keeping step-time overhead <= 1.5x baseline.
Baseline:
  Commit: e481a31b949a13c856b22a765bd14dee9b9b0322
  Config: MLP/CNN baselines, 5 epochs, batch 128, seeds 1,2,3, device cuda:0
  Dataset: torchvision MNIST (downloaded to project/data)
Change:
  Insert multi-layer Operator Basis Layers after each hidden block using the
  full operator library from ROS-ALTH_02.
Primary metric:
  Step time (ms/step) and throughput (samples/sec) on GPU.
Quality guardrail:
  Test accuracy within 0.5% of baseline for the same epochs.
Acceptance criteria:
  - No training instability (loss divergence or NaNs).
  - Accuracy >= baseline - 0.5%.
  - Runtime overhead <= 1.5x baseline step time.
  - 3 seeds; report mean and std.
Budget:
  GPU only, max wall clock 2 hours total.
Notes:
  - Record GPU model/driver/CUDA/cuDNN from env.json.
  - Use fixed data seed and deterministic flag.
  - Runs: MLP/CNN baselines vs MLP/CNN + OBL (full profile).
