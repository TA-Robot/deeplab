# Grad-Speedup System Spec (Isolated, Spec-Aligned)

Purpose
- Build an isolated, reproducible codebase to evaluate algorithmic speedups for CIFAR-10.
- Focus on time-to-target / steps-to-target / cost-to-target, not only final accuracy.

Scope
- In scope: data pipeline, models, training loop, module hooks, logging, reporting, run scripts.
- Out of scope: ROS-ALTH code, shared project/src, and mixed precision / FlashAttention.

Isolation rules
- All grad-speedup code lives under project/grad-speedup/.
- Do not import or modify project/src or run_mnist_experiment.py.
- Outputs go under project/runs/grad-speedup/ only.

Directory layout
- project/grad-speedup/src: models, data, training, module hooks.
- project/grad-speedup/scripts: runners, grid generators, queue runners.
- project/grad-speedup/configs: JSON/YAML experiment configs.
- project/grad-speedup/reports: generated summaries.
- project/docs/grad-speedup: specs, plans, experiment notes.

Run IDs and output structure
- run_id format: YYYYMMDD-grad-speedup-cifar10-<variant>
- Output root default: project/runs/grad-speedup/
- Per run:
  - config.json, env.json
  - summary.json (aggregated across seeds)
  - seed-<seed>/metrics.jsonl and seed-<seed>/summary.json

Data pipeline
- Dataset: CIFAR-10 (train 50k, test 10k), 32x32x3.
- Train augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip(0.5).
- Normalize: CIFAR-10 mean/std.
- DataLoader: pin_memory=True, drop_last=True for train.

Models
- Primary: ResNet-18 CIFAR variant (conv1 3x3, stride 1, no maxpool).
- Optional: WideResNet-28-10 or ResNet-20/32 for robustness checks.

Training loop
- Loss: CrossEntropyLoss.
- Optimizers: SGD+Momentum, AdamW.
- Baseline note: for fixed-LR/no-schedule comparisons, treat SGD momentum=0.0 (mom0) as the default baseline; SGD momentum=0.9 is a separate “scheduled baseline” series.
- Determinism: optional flag, record in env.json.
- Step-based control (primary):
  - max_steps is a hard cap on total optimizer steps.
  - eval_interval_steps drives evaluation cadence (default 200).
- Epoch-based control (compat):
  - epochs still supported; avoid double-eval when step eval is enabled.
- Early stop:
  - mode "max": stop when highest target accuracy is hit.
  - mode "first": stop when any target is hit (screening).

Step-time measurement
- Warmup steps: first 50 steps excluded (configurable).
- Measurement steps: next 200 steps averaged (configurable).
- Report mean step time (ms) and throughput; prefer torch.cuda.Event on GPU.

Targets and metrics
- Targets: A* in {0.80, 0.85, 0.90, 0.92, 0.94} (configurable; short budgets typically use 0.80/0.85).
- For each target, record:
  - Steps-to-target T(A*)
  - Time-to-target W(A*)
  - Cost-to-target C(A*) = T(A*) * mean_step_time
- Train metrics: loss, accuracy, step time stats.
- Optional metrics: grad norm percentiles, curvature percentiles, data loader wait time, peak GPU memory.

Logging format (metrics.jsonl)
- step records: type=step, split=train, epoch, global_step, loss, accuracy, lr, step_size,
  grad_norm, curvature, step_time_ms, line_search_iters, line_search_accepted.
- epoch records: type=epoch, split=train/test, epoch, global_step, loss, accuracy, samples,
  step_time_ms, throughput, steps, step_size stats, grad_norm stats, curvature stats.
- timing records: type=epoch_timing, epoch, global_step, epoch_time_sec.

Module stacking rules (spec-aligned)
- Base optimizer (exclusive): SGD+Momentum or AdamW.
- Step control (exclusive): None, EoSS, Adaptive Backtracking (Silver optional).
- Geometry/clip (optional): GGNC global or layerwise.
- Outer acceleration (optional): Anderson.
- Sparsity (optional): LinBreg.
- Direction/preconditioning (SOAP, GN, etc) is a separate track and not part of the base 72 grid.

Config schema (current)
- run: {run_id, output_root}
- dataset: {name, data_dir, val_size, batch_size, num_workers, seed, download}
- model: {name}
- optimizer: {type, lr, momentum, weight_decay}
- train: {epochs, max_steps, deterministic, device}
- logging: {log_interval_steps, eval_interval_epochs, eval_interval_steps, warmup_steps, measure_steps, grad_norm_every}
- targets: [0.80, 0.85, 0.90, 0.92, 0.94]
- modules:
  - step_control: {name, beta, hvp_interval, ema, backtrack_c, backtrack_max, backtrack_rho}
  - clip: {mode, rho}
  - outer: {name, interval, memory, damping}
  - sparsity: {name, lambda, update_interval}

Experiment matrix (base grid)
- Base optimizer: {SGD, AdamW} (2)
- Step control: {none, EoSS, backtracking} (3)
- Clip: {none, ggnc-global, ggnc-layerwise} (3)
- Outer accel: {none, anderson} (2)
- Sparsity: {none, linbreg} (2)
- Total: 72

Reproducibility
- Record seeds, determinism flag, device, torch/torchvision versions.
- Keep configs and CLI args with every run.
