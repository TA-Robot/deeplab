# CIFAR-10 Implementation Spec (Grad-Speedup Track, Spec-Aligned)

Goal
- Implement algorithmic speedups for gradient methods as composable modules.
- Evaluate single modules and their combinations on CIFAR-10 under fixed conditions.
- Primary focus is time-to-target / steps-to-target / cost-to-target (not just final accuracy).

Cost model
- Total cost C = T * c
  - T: steps-to-target (optimizer steps to reach target accuracy)
  - c: mean cost per step (wall time proxy)

Primary metrics (must report)
- Steps-to-target: T(A*) for A* in {0.80, 0.85, 0.90, 0.92, 0.94} (configurable; for short budgets use 0.80/0.85)
- Time-to-target: W(A*) (wall time to reach A*)
- Cost-to-target: C(A*) = T(A*) * mean step time
- Learning curves: train loss/acc, test acc vs step/epoch/time

Secondary metrics (recommended)
- Mean step time (warmup excluded) + P50/P90 if available
- Peak GPU memory
- Gradient norm stats (mean, p50, p90)
- Optional curvature proxy (if using EoSS)

Fixed conditions
- Precision: FP32 only (no mixed precision)
- Augmentation (train only): RandomCrop(32, padding=4), RandomHorizontalFlip(p=0.5)
- Normalize: CIFAR-10 mean/std
- Batch size: 128 (other batch sizes are a separate series)
- Budget: max_steps is the hard cap (default 100k); early stop on target threshold
- Baseline note (important): in the fixed-LR/no-schedule regime, SGD momentum=0.0 (mom0) is the default baseline for fair comparisons; SGD momentum=0.9 is treated as a separate “scheduled baseline” series.

Dataset
- CIFAR-10: train 50k / test 10k
- DataLoader: pin_memory=True, drop_last=True (train)

Model
- Primary baseline: ResNet-18 CIFAR variant (conv1 3x3, stride 1, no maxpool)
- Optional follow-up: WideResNet-28-10 or ResNet-20/32

Training loop requirements
- Loss: CrossEntropyLoss (no label smoothing)
- Logging cadence:
  - train metrics every N steps (default 100)
  - test acc every eval_interval_steps (default 200) and/or per epoch
  - keep eval cadence identical across compared runs (time-to-target resolution depends on it)
- Early stop modes:
  - "max": continue until highest A* reached
  - "first": stop at first threshold reached

Step-time measurement
- Use torch.cuda.Event on GPU
- Exclude warmup steps (default 50), average next K steps (default 200)
- Record mean step time; optionally P50/P90

Module architecture (composable)
- Base optimizer (exclusive): SGD+Momentum, AdamW
- Step control (exclusive): None, EoSS, Adaptive Backtracking (Silver optional)
- Geometry/clip (optional): GGNC global or layerwise
- Outer acceleration (optional): Anderson
- Sparsity (optional): LinBreg
- Direction/preconditioning (SOAP/GN/etc) is handled as a separate track and not part of the base grid

Experiment grid (base 72 conditions)
- Base optimizer: {SGD, AdamW} (2)
- Step control: {None, EoSS, Backtracking} (3)
- Geometry: {None, GGNC global, GGNC layerwise} (3)
- Anderson: {None, On} (2)
- Sparsity: {None, LinBreg} (2)
- Total: 2 * 3 * 3 * 2 * 2 = 72

Seeds and statistics
- Seeds: {0, 1, 2}
- Report mean/std and best/worst for time-to-target

Logging requirements
- Per run: config.json, env.json, metrics.jsonl, summary.json
- summary.json includes per-target steps/time/cost
- Record warmup/measure steps and hardware details

Acceptance criteria (initial)
- Baseline reaches A* = 0.85 within the current promotion budget (e.g., max_steps=14000) in the no-schedule regime
- If A* >= 0.90 is required, run a separate “scheduled baseline” series (e.g., cosine LR) and compare within that regime
- At least one configuration improves cost-to-target by >= 1.5x vs baseline at A*
- No training instability (NaNs, divergence)

Notes
- Keep the grad-speedup codebase isolated; do not reuse project/src.
- Direction/preconditioning methods (SOAP/GN) are tracked separately and should not influence base-grid decisions.
