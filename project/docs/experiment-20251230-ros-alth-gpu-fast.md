# Experiment Brief: 20251230-ros-alth-gpu-fast

ID: 20251230-ros-alth-gpu-fast
Owner: TBD
Date: 2025-12-30
Hypothesis:
  Removing O(D^2) operators (SoftSort/Sinkhorn/Attention) and adding lightweight
  cross-feature mixing (low-rank bilinear, group mix, permuted blur) reduces
  step-time overhead while keeping accuracy within guardrails.
Baseline:
  Commit: e481a31b949a13c856b22a765bd14dee9b9b0322
  Config: MLP/CNN baselines, 5 epochs, batch 128, seeds 1,2,3, device cuda:0
  Dataset: torchvision MNIST/Fashion-MNIST/CIFAR-10 (downloaded to project/data)
Change:
  Use `--obl-profile fast` (no SoftSort/Sinkhorn/Attention; add low-rank/group mix/permute blur).
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
  GPU only, max wall clock 4 hours total.
Notes:
  - Compare against existing GPU baselines for each dataset.
  - Runs: MLP/CNN baselines vs MLP/CNN + OBL (fast profile).
