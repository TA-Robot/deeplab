# Grad-Speedup System Spec (Isolated)

Purpose
- Build an isolated, reproducible codebase to evaluate algorithmic speedups for gradient training on CIFAR-10.
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
- project/grad-speedup/scripts: runners and report builders.
- project/grad-speedup/configs: JSON/YAML experiment configs.
- project/grad-speedup/reports: generated summaries.
- project/docs/grad-speedup: specs and experiment notes.

Run IDs and output structure
- run_id format: YYYYMMDD-grad-speedup-cifar10-<variant>[-<tag>]
- Output root default: project/runs/grad-speedup/
- Per run:
  - config.json, env.json
  - summary.json (aggregated across seeds)
  - seed-<seed>/metrics.jsonl and seed-<seed>/summary.json

Data pipeline
- Dataset: CIFAR-10 (train 50k, test 10k), 32x32x3.
- Train augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip(0.5).
- Normalize: CIFAR-10 mean/std.
- Validation: fixed split from train with seed control (val_size configurable).
- DataLoader: pin_memory=True, drop_last=True for train.

Models
- ResNet-18 CIFAR variant (conv1 3x3, stride 1, no maxpool).
- Small CNN baseline (lightweight, for quick smoke runs).

Training loop
- Loss: CrossEntropyLoss.
- Optimizers: SGD+momentum, Adam.
- Determinism: optional flag, record in env.json.
- Early stop:
  - mode "max": stop when highest target accuracy is hit.
  - mode "first": stop when any target is hit (screening).

Step-time measurement
- Warmup steps: first 50 steps excluded (configurable).
- Measurement steps: next 200 steps averaged (configurable).
- Report: mean step time (ms) and throughput.

Targets and metrics
- Targets: A* in {0.85, 0.90, 0.92, 0.94}.
- For each target, record:
  - Steps-to-target T(A*)
  - Time-to-target W(A*)
  - Cost-to-target C(A*) = T(A*) * mean_step_time
- Train metrics: loss, accuracy, step time stats.
- Optional metrics: grad norm percentiles, curvature percentiles.
- Optional diagnostics (future): data loader wait time, peak GPU memory.

Logging format (metrics.jsonl)
- step records: type=step, split=train, epoch, global_step, loss, accuracy, lr, grad_norm, curvature, step_time_ms.
- epoch records: type=epoch, split=train/test, epoch, global_step, loss, accuracy, samples, step_time_ms, throughput, steps, grad_norm stats, curvature stats.
- timing records: type=epoch_timing, epoch, global_step, epoch_time_sec.

Module interfaces (design intent)
- Module B (direction): preconditioned direction (single choice).
- Module C (step control): step_rule in {none, l0l1, eoss, silver}.
  - l0l1: lr_eff = lr / (L0 + L1 * grad_norm).
  - eoss: estimate directional curvature via HVP every K steps and cap lr by 2/(curvature+eps), scaled by beta.
  - silver: scheduled step size based on the silver-ratio schedule (paper-accurate implementation required).
- Module D (stability/clip): GGNC optional (future).
- Module E (outer acceleration): Anderson optional (future).
- Only one module from B and C at a time; D/E are additive.

Paper-accuracy guardrail
- All module implementations must match their primary paper(s).
- See project/docs/grad-speedup/method-conformance.md for exact algorithms and references.
- The current l0l1/eoss rules are placeholders and must be replaced by paper-accurate methods
  before any results are considered valid.

Module B requirements (curvature/preconditioning)
- Module B is exclusive (one choice at a time).
- Keep memory and compute overhead explicit in logs.
- v1 candidates (pick one for implementation):
  - diagonal preconditioner (RMS-like) for a minimal baseline
  - layerwise scaling (blockwise) with configurable update frequency
- Must expose configuration fields for update frequency and damping/epsilon.
- Must not change baseline behavior when disabled.

Module D requirements (GGNC)
- Optional clip stage after gradient computation.
- Support global L2 clipping and layerwise clipping modes.
- Configuration fields: rho or max_norm, mode in {global, layerwise}.
- Must record clip coefficient stats (mean/p50/p90) if enabled.

Module E requirements (Anderson)
- Optional outer acceleration with memory m and interval K_A.
- Requires a fallback to standard update on numerical instability.
- Must log activation count and failure count.

Config schema (planned)
- run: {run_id, output_root}
- dataset: {name, data_dir, val_size, batch_size, num_workers, seed, download}
- model: {name}
- optimizer: {type, lr, momentum, weight_decay}
- train: {epochs, deterministic, device}
- logging: {log_interval_steps, eval_interval_epochs, warmup_steps, measure_steps, grad_norm_every}
- targets: [0.85, 0.90, 0.92, 0.94]
- modules:
  - step_control: {name, l0, l1, curv_every, curv_eps, eoss_beta, silver_rho}
  - direction: {name, update_every, eps, damping}
  - clip: {mode, rho}
  - outer: {name, interval, memory, damping}

Experiment matrix (v1)
- Baselines: SGD, Adam.
- Module C variants: none, l0l1, eoss.
- Module B/D/E are deferred to later tickets.

Experiment matrix (v2 placeholder)
- Base optimizer: {SGD, Adam} (2)
- Step control: {none, l0l1, eoss} (3)
- Clip: {none, ggnc-global, ggnc-layerwise} (3)
- Outer accel: {none, anderson} (2)
- Total (example): 2 * 3 * 3 * 2 = 36 (before Module B).

Reporting
- Per run summary.json should contain per-seed targets with steps/time/cost.
- Report builder aggregates summaries into a single JSON (and CSV in v2).

Baseline execution
- Smoke: 1 epoch, CPU, single seed.
- Baseline: 5 epochs, CIFAR-10, seeds 0/1/2, device=cuda if available.

Reproducibility
- Record seeds, determinism flag, device, torch/torchvision versions.
- Keep configs and CLI args with every run.
