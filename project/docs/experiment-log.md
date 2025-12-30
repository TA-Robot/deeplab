# Experiment Log

- 2025-12-30: 20251229-ros-alth-mnist-cpu (running)
  - Brief: project/docs/experiment-20251229-ros-alth-mnist-cpu.md
  - Status: running (parallel)
  - Notes:
    - full operator library + multi-layer OBL (K=59)
    - OBL SoftSort/Sinkhorn batch matmul fix applied before this rerun
  - Run IDs:
    - 20251229-ros-alth-mnist-cpu-mlp-baseline
    - 20251229-ros-alth-mnist-cpu-cnn-baseline
    - 20251229-ros-alth-mnist-cpu-mlp-obl
    - 20251229-ros-alth-mnist-cpu-cnn-obl

- 2025-12-29: 20251229-ros-alth-mnist-cpu (superseded)
  - Brief: project/docs/experiment-20251229-ros-alth-mnist-cpu.md
  - Status: superseded (pilot parallel run)
  - Notes: CPU-only MNIST, MLP/CNN baselines vs OBL K=32 variants
  - Run IDs:
    - 20251229-mlp-baseline-par
    - 20251229-mlp-obl-par
    - 20251229-cnn-baseline-par
    - 20251229-cnn-obl-par

- 2025-12-29: 20251229-ros-alth-mnist-cpu-full (superseded)
  - Brief: project/docs/experiment-20251229-ros-alth-mnist-cpu.md
  - Status: superseded (pre-fix runs)
  - Notes:
    - full operator library + multi-layer OBL
    - OBL runs restarted after fixing SoftSort/Sinkhorn batch matmul
  - Run IDs:
    - 20251229-zz-mlp-baseline-full
    - 20251229-zz-cnn-baseline-full
    - 20251229-zzzz-mlp-obl-full
    - 20251229-zzzz-cnn-obl-full
