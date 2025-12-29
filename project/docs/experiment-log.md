# Experiment Log

- 2025-12-29: 20251229-ros-alth-mnist-cpu (completed)
  - Brief: project/docs/experiment-20251229-ros-alth-mnist-cpu.md
  - Status: completed (parallel)
  - Notes: CPU-only MNIST, MLP/CNN baselines vs OBL K=32 variants
  - Run IDs:
    - 20251229-mlp-baseline-par
    - 20251229-mlp-obl-par
    - 20251229-cnn-baseline-par
    - 20251229-cnn-obl-par
  - Report: project/dashboard/data/report.json
  - Summary: project/docs/experiment-20251229-ros-alth-mnist-cpu-summary.md

- 2025-12-29: 20251229-ros-alth-mnist-cpu-full (running)
  - Brief: project/docs/experiment-20251229-ros-alth-mnist-cpu.md
  - Status: running (parallel)
  - Notes:
    - full operator library + multi-layer OBL
    - OBL runs restarted after fixing SoftSort/Sinkhorn batch matmul
  - Run IDs:
    - 20251229-zz-mlp-baseline-full
    - 20251229-zz-cnn-baseline-full
    - 20251229-zzzz-mlp-obl-full
    - 20251229-zzzz-cnn-obl-full
