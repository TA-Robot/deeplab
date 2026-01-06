# Temp Materials Full Record (Grad-Speedup)

Source
- Extracted from /workspace/deeplab/temp (ChatGPT share export).
- This file preserves the full method inventory and CIFAR-10 spec from the temp investigation.
- It is intentionally exhaustive and not a curated shortlist.

Notation and cost decomposition
- Parameters: theta in R^d.
- Steps to target: N (or T).
- Cost per step: c_step.
- Total cost: C_total = N * c_step.
- Relative ratios: r_T = T_new / T_base, r_c = c_new / c_base, r_C = r_T * r_c.
- Goal: r_C <= 0.01 (approx 100x total cost reduction).

1) Round 1 materials: broad levers (baseline inventory)

1.1 Lever A: reduce steps T (algorithmic progress per step)
- A1 Light second-order / curvature preconditioning
  - Sophia: diagonal Hessian estimate with clipping; reported ~50% fewer steps for GPT-style pretraining relative to Adam (approx 2x step reduction).
  - Shampoo: Kronecker-factor preconditioning; distributed implementations report ~10% overhead and AlgoPerf time-to-result improvements around 28-30%.
  - K-FAC: Fisher (natural gradient) approximation; classic second-order acceleration.
  - Expected ceiling: when AdamW is already tuned, improvements often land in ~10-30%, but 1.2x to 2x step reduction can appear on favorable tasks.
- A2 Momentum/EMA variants (lightweight but sometimes effective)
  - Adan: modified Nesterov-style momentum; reports up to ~2x cost reductions on some workloads.
  - Lion: sign-based momentum update; discovered by search; can reduce compute while maintaining quality.
  - AdEMAMix: mixes short and long EMA for adaptive memory.
  - Schedule-Free AdamW: removes explicit LR schedule; reduces tuning effort; AlgoPerf self-tuning results show competitiveness.
- A3 Learning-rate and batch strategies
  - Super-Convergence / 1cycle: large peak LR can yield order-of-magnitude speedups in some regimes, but sensitive to task and regularization.
  - Large batch with LARS (ResNet) / LAMB (BERT): wall-clock speedups via parallelism, subject to scaling efficiency.
- A4 Muon (orthogonalization-based updates)
  - "Muon is Scalable for LLM Training": orthogonalization (e.g., Newton-Schulz) for matrix parameters; reported ~52% of AdamW FLOPs for similar performance on LLMs.

1.2 Lever B: reduce cost per step c
- Mixed precision (FP16/BF16): Tensor Core acceleration; real gains depend on bottlenecks.
- FlashAttention: IO-aware attention kernel; up to ~3x in GPT-2 settings.
- 8-bit optimizer states: lower memory footprint; can allow larger batch sizes and better throughput.
- ZeRO: partitions optimizer states, gradients, and parameters; improves memory and scaling on large clusters.

1.3 Lever C: reduce distributed communication cost
- Deep Gradient Compression: gradient sparsification with correction; reported 270x-600x compression when communication is the bottleneck.

1.4 Multiplicative example (illustrative)
- Example: Sophia (~0.5x steps) * FlashAttention (~0.33x step cost) * mixed precision (~0.25x-0.5x) yields ~12x. Add distribution and batch scaling to push further.
- Key takeaway: 100x is unlikely from a single optimizer; multiple levers must multiply.

1.5 Selection axis (for narrowing later)
- Bottleneck: compute vs memory vs communication.
- Model type: Transformer vs Conv vs others.
- Cost definition: wall clock vs FLOPs vs dollars.

2) Round 2 materials: prioritized candidates for 1/100

2.1 Critical batch size (CBS)
- Large batch ideally reduces steps ~1/B, but above B* (critical batch size) the step reduction saturates and sample efficiency degrades.
- McCandlish: gradient noise scale predicts effective max batch size.
- Shallue: batch size vs steps depends on workload and tuning.
- "Critical Batch Size Revisited" (2025) suggests more direct CBS estimation.
- AllenAI OLMo blog: practical CBS measurement challenges.

2.2 Full and Layerwise Gauss-Newton as upper bound
- Full GN preconditioning on Transformers (up to 150M) reported 5.4x fewer steps vs SOAP and 16x vs Muon in large-batch regime.
- Layerwise GN (ignoring cross-layer curvature) reportedly within ~1.4x of full GN steps.
- GN continues to improve with larger batch sizes where AdamW saturates, suggesting CBS can be pushed.
- GN-PROX-LINEAR view: optimize the original loss over linearized model; suggests higher-order loss terms are small, guiding approximations.

2.3 SOAP
- SOAP reinterprets Shampoo as running Adam in the Shampoo eigenbasis, reducing hyperparameters to preconditioning frequency.
- Reported 40%+ fewer iterations and 35%+ wall-clock reduction vs AdamW in LLM pretraining; better than Shampoo.
- Temp emphasis: SOAP’s key knob is preconditioning frequency (reduce frequency without breaking, unlike Shampoo).

2.4 Muon (scalable)
- Scalable Muon adds weight decay and parameter-wise scaling; reports ~2x compute efficiency vs AdamW and distributed implementation.

2.5 Sophia
- Diagonal Hessian + clipping; GPT-2 scale reports ~2x step reduction vs Adam/AdamW.

2.6 Distributed Shampoo (AlgoPerf signal)
- AlgoPerf benchmarks algorithm changes on fixed hardware. Distributed Shampoo ranks near top with ~28-30% time-to-result improvements; strong baseline.

2.7 CLEAN (Nystrom sketch)
- Two-sided preconditioning with randomized Nystrom sketches to reduce memory; promising but under review.

2.8 Communication compression
- PowerSGD: low-rank gradient compression reduces communication; requires careful monitoring for convergence degradation (PowerSGD+ theoretical notes).

2.9 AlgoPerf as reality check
- Shows what algorithm-only gains are robust across workloads.
- Highlights tradeoff between "single-task wins" vs "robust across tasks" improvements.

3) Round 3 materials: theory-driven and newer methods (2024-2025 focus)

3.1 Full Gauss-Newton details
- Loss F(theta) = (1/|B|) sum l(f(theta; x), y).
- Jacobian J = df/dtheta, loss Hessian H_l.
- GN curvature: G(theta) = J^T H_l J. Solve G * Delta approx grad F, update theta <- theta - eta * Delta.
- Reported large-batch results: GN reached target loss in 54 steps at batch sizes far above CBS; 5.4x fewer steps than SOAP, 16x fewer than Muon.
- Layerwise GN approximates full GN surprisingly well; cross-layer curvature may not dominate.

3.2 Randomized linear algebra for scalable 2nd order
- Exact Gauss-Newton (EGN, 2025): uses low-rank linear algebra (Duncan-Guttman identity) so updates are computed via a matrix of size batch instead of parameter dimension.
- Randomized sketching for training: apply sketches to activations or curvature to reduce memory and allow larger batches or better curvature estimates.
- Research questions: design Nystrom or randomized SVD for heavy layers (Wq/Wk/Wv/FFN) and measure approximation error vs step reduction.

3.3 Step-size adaptation via local smoothness
- Armijo line search under non-uniform smoothness: can yield faster rates than fixed 1/L steps in some convex objectives (e.g., logistic regression), indicating local adaptivity matters.
- Adaptive Backtracking Line Search (ICLR 2025): shrink factor depends on violation magnitude; claims no extra computation and similar guarantees as classic backtracking.
- Stochastic Polyak Step-size with Momentum (ICLR 2025): combines SPS with heavy-ball momentum, recovering fast rates under interpolation and giving guarantees without interpolation.
- Stochastic Adaptive GD Without Descent (2025): uses only stochastic gradients to adapt step size to local geometry.

3.4 Anderson acceleration and fixed point view
- Optimization as fixed point: theta_{t+1} = F(theta_t).
- Anderson uses previous residuals r_t = F(theta_t) - theta_t to extrapolate.
- AADL provides a PyTorch implementation; open questions include stability with stochastic gradients and interaction with momentum/Adam.

3.5 Variance reduction for large models
- MARS (2025): integrates preconditioned gradients with variance reduction via scaled stochastic recursive momentum; proposes variants based on AdamW/Lion/Shampoo; reports large gains on GPT-2.

3.6 Implicit/prox/Bregman updates
- Stochastic Proximal Point Method (SPPM): implicit update x_{k+1} = x_k - gamma * grad f(x_{k+1}); more stable for stiff problems and can allow larger steps.
- MSBPG (JMLR 2025): nonconvex stochastic Bregman proximal gradient using polynomial kernels; claims automatic gradient scaling and reduced gradient explosions.
- Bregman distance D_h(x,y) = h(x) - h(y) - <grad h(y), x - y> enables non-Euclidean geometry.

3.7 ODE/SDE/control viewpoints
- PIDAO (Nature Communications 2024): PID control in optimization; claims faster convergence and improved saddle escape; ties to stability control.
- Diffusion limit of SGD: formalize SGD as SDE with noise and friction; allows principled design of LR, momentum, and noise.
- Hessian-aware SDE (SME): incorporate curvature into drift/diffusion for refined dynamics.
- Two time-scale stochastic approximation models for phenomena like grokking.

3.8 Research questions
- Can line search be approximated with periodic evaluation, large batches, or proxy losses to reduce overhead?
- How close are deep nets to non-uniform or generalized smoothness assumptions in practice?
- Which approximations to GN preserve the most benefit for the least cost?

4) Additional materials round 1: parametrization and quasi-Newton frameworks

4.1 Parametrization as implicit preconditioning (Levin)
- Reparameterize x = phi(y), optimize g(y) = f(phi(y)).
- Gradient in y: grad_y g(y) = J_phi(y)^T grad_x f(x).
- First-order update in x approximates x_{t+1} approx x_t - eta * [J_phi J_phi^T] * grad_x f(x_t).
- Conclusion: changing parametrization changes the implicit preconditioner; can approximate GN-like behavior without changing the optimizer.
- Applications: symmetry quotienting, low-rank factorization, Burer-Monteiro, neural nets.

4.2 Scieur (AISTATS 2024): GD and cubic-regularized Newton blend
- Provides global (non-asymptotic) rates that interpolate between GD and cubic-regularized Newton.
- Uses subspace minimization: x_{t+1} = x_t + D_t * alpha_t with D_t in R^{d x N}, N << d.
- Curvature in subspace is approximated by gradient differences; Anderson acceleration is related to this view.
- Practical idea: per-layer subspaces of size N ~ 4-16 to get low-rank quasi-Newton behavior.

4.3 Jiang and Mokhtari (2024): online quasi-Newton for monotone operators
- Frames Jacobian update as online optimization; gives global convergence and eventual superlinear improvement after O(d) steps in strongly monotone cases.
- Does not require Jacobian queries; stability logic is explicit.
- Potentially applicable to convex or locally PL-like subproblems in deep nets (e.g., last layers).

4.4 Block Broyden (NeurIPS 2023)
- Multi-secant (block) good/bad Broyden variants; local superlinear convergence.
- Combine with subspace minimization to form layerwise low-rank quasi-Newton updates.

4.5 Qi (2025): H-convex/H-smooth and structure matrix
- Defines H-convex/H-smooth to avoid strong convexity or Lipschitz smoothness assumptions.
- Structure matrix: A_x = (grad_theta f_theta(x))^T (grad_theta f_theta(x)); eigenvalues define a structural error metric.
- Empirical risk bounds can be expressed via local grad norm and structure error, suggesting two axes for optimization.

4.6 Bolte smooth adaptivity and MSBPG connection
- Smooth adaptivity allows Bregman-based descent without Euclidean smoothness.
- MSBPG uses polynomial kernels that satisfy smooth adaptivity; ties to the structure-matrix view.

4.7 Hypotheses derived from this round
- H1: designing parametrization phi yields implicit GN/natural-gradient-like updates and can reduce steps in large-batch regimes.
- H2: Scieur-style low-dimensional subspace + gradient-difference curvature yields cheap quasi-Newton behavior.
- H3: Layerwise low-rank block Broyden reduces structure error and accelerates convergence.

5) Additional materials round 2: stability, stepsize, and compute reduction

5.1 Edge of Stochastic Stability (EoSS) and batch sharpness
- GD stability roughly requires eta < 2 / lambda_max(H).
- EoSS for SGD: stability boundary characterized by batch sharpness (directional curvature).
- Directional curvature example: s_t = (g_t^T H_t g_t) / ||g_t||^2, estimated via HVP.
- Step control: set eta_t near 2 / (s_t + eps), with EMA smoothing and periodic estimation.

5.2 Silver step sizes (COLT 2025)
- Shows acceleration from step-size schedules alone (no momentum), extends to prox/projected settings.
- Complexity claims: smooth convex and strongly convex cases improve with a silver-ratio schedule.
- Extensions to Riemannian and Wasserstein spaces connect to natural-gradient style geometry.

5.3 Ravine structure and long Polyak steps (Mathematical Programming 2025)
- Even with quartic growth (not quadratic), adaptive step sizes can yield near-linear convergence.
- Proposal: many short GD steps plus occasional long Polyak steps; ties to ravine manifold structure.
- In practice, L_star can be approximated by best-so-far loss and clipped via stability rules (EoSS).

5.4 PIDAO (PID control)
- PID feedback (P/I/D) applied to optimization dynamics; claims faster convergence and saddle escape.
- Fits with stability-centric design (keep dynamics near safe boundary).

5.5 Learned optimizers
- muLO: applies muP to learned optimizers for better meta-generalization across scales.
- Celo: meta-training in ~24 GPU hours with broad wins; promise is lower tuning cost plus step reduction.

5.6 Prox/Bregman variants with variance reduction
- SPPM analysis under generalized smoothness (phi-smoothness) suggests stability and lower tuning.
- BSPPA combines Bregman proximal updates with SAGA/SVRG variance reduction.

5.7 Second-order variants summarized here
- Full/layerwise GN as step-reduction upper bound.
- EGN: batch-size matrix factorization for GN updates.
- Dual NGD: solve GN in residual space (size m) rather than parameter space (size n), with Nystrom preconditioning and geodesic acceleration.

5.8 Dynamic sparsity via multilevel mirror descent / linearized Bregman
- Alternates between dynamic sparsity discovery and frozen-sparsity updates.
- Claims theoretical FLOPs of ~6% of SGD (vs ~38% for standard Bregman), enabling large c reduction.
- Real wall-time gains depend on sparse kernels; report theoretical and effective FLOPs separately.

5.9 Implicit diffusion (generative models)
- Treats sampling and optimization jointly; uses bilevel optimization and implicit differentiation.
- Relevant when sampling cost dominates (diffusion models/EBMs), not general supervised SGD.

6) Additional materials round 3: generalized smoothness, clipping, and cubic methods

6.1 (L0, L1)-smoothness and L0L1-GD
- Generalized smoothness: ||grad f(x) - grad f(y)|| <= (L0 + L1 * sup ||grad f(u)||) * ||x - y||.
- L0L1-GD update: x_{k+1} = x_k - [eta / (L0 + L1 * ||grad f(x_k)||)] * grad f(x_k).
- Interpretation: smooth, principled form of gradient normalization/clipping; supports two-phase schedules (large grad then small grad).

6.2 GGNC (Generalized Gradient Norm Clipping)
- Defines sharp operator d_sharp and LMO over unit norm ball; in Euclidean case recovers classic clipping.
- Update: x_{k+1} = x_k - gamma * tau_k * [d_k]^sharp, with tau_k = min(1, rho / ||d_k||_*).
- Weight decay can be interpreted as a Frank-Wolfe short step in this framework.
- Design choice: choose norm/dual norm to match layerwise or blockwise scales.

6.3 Parameter-free and line-search connections
- Normalized SGD with Momentum can be near parameter-free under (L0, L1)-smoothness, with tradeoffs in L1 factors.
- Armijo line search can improve convergence rates under non-uniform smoothness; practical variants require low-cost acceptance tests.

6.4 Trust-region view under generalized smoothness
- First-order trust-region methods subsume clipping and normalization as special cases.
- Second-order trust-region yields complexity to second-order stationary points; negative curvature handling is explicit.
- Design goal: set trust-region radius from grad norm/noise, aligning with EoSS and sharpness.

6.5 Stochastic cubic Newton with momentum (AISTATS 2025)
- Uses momentum to stabilize noisy gradient/Hessian estimates.
- Local model: m(p) = g^T p + 0.5 p^T H p + (sigma/3) * ||p||^3.
- Claims second-order stationarity with 1-sample steps; promising for stiff problems but heavy per-step compute.

6.6 SR1 and adaptive regularized cubics (nonconvex)
- Limited-memory SR1 allows indefinite Hessian estimates; can exploit negative curvature to escape saddles.
- Adaptive regularized cubics can yield closed-form subproblem solutions in some settings.
- Noisy Broyden with line search and finite-precision tolerance has been explored (SSRN), but requires reproducibility checks.

7) Combination framework and module view (from temp synthesis)

7.1 Cost model for combinations
- r_C = r_T * r_c; 100x means r_C <= 0.01.
- Measure r_T and r_c separately to avoid double counting.

7.2 Module categories
- Module A (compute reduction): dynamic sparsity via multilevel mirror descent / linearized Bregman.
- Module B (direction / preconditioning): Full GN, Layerwise GN, SOAP, Muon, MARS (choose one).
- Module C (step-size control): EoSS, Adaptive Backtracking, Armijo, Silver stepsize (choose one).
- Module D (stability/geometry): GGNC (optional, usually one variant).
- Module E (outer acceleration): Anderson or quasi-Newton/cubic (optional, periodic).

7.3 Exclusivity and compatibility rules
- Module B is exclusive (do not stack multiple direction methods).
- Module C is exclusive (do not combine multiple step-control rules).
- Module A can wrap almost anything; D and E are optional add-ons.

7.4 Upper-bound combination math (illustrative)
- Full GN example: r_T ~ 1/16, r_c ~ 1.5 -> r_C ~ 0.094 (approx 10.7x cost reduction).
- Dynamic sparsity: r_c ~ 0.06 (approx 16.7x FLOPs reduction).
- If independent, 16.7 * 10.7 ~ 179x (upper bound). Real interactions will reduce this.

7.5 Candidate combinations
- Research-max set:
  - A: dynamic sparsity
  - B: Layerwise GN (lighter than full GN)
  - C: EoSS or Adaptive Backtracking (one)
  - D: GGNC
  - E: Scieur-style periodic acceleration
- Practical set:
  - A: dynamic sparsity
  - B: Muon or SOAP (one)
  - C: Adaptive Backtracking or EoSS (one)
  - D: GGNC

7.6 Common failure modes and mitigations
- Double counting (e.g., r_T already baked into r_c): measure separately.
- Sparsity changes curvature behavior: apply preconditioning only on active weights.
- Line-search overhead: apply periodically, not every step.
- Combinatorial explosion: tune single modules first, then fix and combine.

8) CIFAR-10 experiment specification (full)

8.1 Purpose
- Build a CIFAR-10 framework that compares single modules and combinations under identical conditions.
- Primary metrics are time-to-target, steps-to-target, and cost-to-target, not just final accuracy.

8.2 Metrics and targets
- Targets: A* in {0.85, 0.90, 0.92, 0.94}.
- Steps-to-target: T(A*).
- Time-to-target: W(A*).
- Cost-to-target: C(A*) = T(A*) * mean_step_time.
- Secondary: step time statistics, max GPU memory, theoretical and effective FLOPs, gradient norm stats, optional curvature proxies.

8.3 Fixed conditions
- Precision: FP32 only (no AMP).
- Augmentation fixed: RandomCrop(32, padding=4), RandomHorizontalFlip(0.5), Normalize; no Cutout/RandAugment in this series.
- Max epochs (e.g., 200) or max steps (e.g., 100k); early stop once targets reached.

8.4 Dataset
- CIFAR-10: train 50k, test 10k, 32x32x3.
- DataLoader: batch 128 (separate series for 256/512), num_workers fixed, pin_memory true, drop_last true.

8.5 Model
- Primary: ResNet-18 (CIFAR variant): 3x3 conv1, stride 1, padding 1, no maxpool.
- Optional: WideResNet-28-10 or ResNet-20/32 for robustness checks.

8.6 Training loop
- Loss: CrossEntropyLoss (no label smoothing).
- Logging: train metrics every N steps (e.g., 100), test acc every epoch (or half epoch).
- Early stop modes:
  - Mode A: continue until highest A* reached.
  - Mode B: stop at first threshold reached (screening).

8.7 Step-time measurement
- Use torch.cuda.Event for GPU.
- Exclude warmup (e.g., first 50 steps), average next K steps (e.g., 200) for mean step time.
- Record mean, p50/p90 (if possible), data loader wait time (if available).

8.8 FLOPs and memory
- Estimate dense FLOPs once; report effective FLOPs for sparsity.
- Record torch.cuda.max_memory_allocated() per epoch.

8.9 Architecture for module stacking
- Base optimizer provides delta_theta; wrappers modify delta.
- Interface idea:
  - base.step(grad, state) -> delta_theta
  - wrapper(delta_theta, grad, state, stats) -> delta_theta'
  - final update: theta <- theta + delta_theta
- Suggested structure:
  - configs/, models/, data/, optim/base/, optim/wrappers/, sparsity/, metrics/, runner/, scripts/, results/

8.10 Module catalog and rules
- Base optimizer (exclusive): SGD+Momentum, AdamW (K-FAC/Shampoo optional later).
- Step-size control (exclusive): None, EoSS, Adaptive Backtracking, Silver stepsize (optional).
- Geometry/clip (optional): GGNC (global or layerwise).
- Outer acceleration (optional): Anderson acceleration (interval K_A, memory m, damping, fallback).
- Sparsity (optional): Linearized Bregman / Multilevel Mirror Descent.

8.11 Step-control specifics
- EoSS: estimate directional curvature s_t via HVP every K steps and smooth by EMA.
- Example LR: eta_t = beta * 2 / (s_t + eps), with clipping.
- Adaptive Backtracking: Armijo-like acceptance, small max retries.
- Silver stepsize: implement schedule function only (no extra computation).

8.12 Sparsity specifics
- Maintain dual z, update z <- z - eta * g, primal w <- shrink(z, lambda).
- Multilevel: alternate sparsity update and frozen phases.
- Always report theoretical and effective FLOPs.

8.13 Combination grid
- Base optimizer: {SGD, AdamW} (2)
- Step control: {None, EoSS, Backtracking} (3)
- Geometry: {None, GGNC global, GGNC layerwise} (3)
- Anderson: {None, On} (2)
- Sparsity: {None, LinBreg} (2)
- Total: 2 * 3 * 3 * 2 * 2 = 72 conditions.

8.14 Seeds and statistics
- Seeds: {0, 1, 2} (3 seeds).
- Report mean, std, best/worst, and quantiles for time-to-target.

8.15 Hyperparameters (minimum search)
- Baselines:
  - batch_size 128, epochs max 200.
  - weight_decay: 5e-4 (SGD), 1e-4 (AdamW) or keep same for comparability.
  - LR: SGD ~0.1, AdamW ~1e-3.
  - Schedule: cosine fixed; add a no-schedule baseline when testing step-control modules.
- EoSS: beta in {0.5, 0.7, 0.9}, HVP interval K in {50, 200}, EMA alpha in {0.9, 0.99}.
- Backtracking: max retries M in {2, 4}, acceptance c in {1e-4, 1e-3}.
- GGNC: rho set from baseline grad norm quantile (e.g., p90).
- Anderson: memory m in {3, 5}, interval K_A in {5, 10}.
- Two-stage tuning: tune modules alone, then fix for combinations.

8.16 Logging (per run)
- Full config (args + seed + env).
- Curves: train loss/acc, test acc vs epoch/step/time.
- Per-target T(A*), W(A*), C(A*).
- Mean step time (warmup excluded), max GPU memory.

8.17 Aggregate output
- CSV: one row per condition per seed, columns for all metrics.

8.18 Acceptance criteria
- Baseline reproduces: reaches at least A* = 0.90 within budget.
- Time-to-target is stable across repeated same-seed runs.
- Combinations do not silently fail; NaN/inf or divergence must be logged as failure with diagnostics.

8.19 Risks and mitigations
- Sparsity does not speed wall time: report theoretical and effective FLOPs separately.
- HVP overhead for EoSS: reduce frequency, smaller batch for HVP, EMA smoothing.
- Anderson instability: damping, longer interval, fallback on poor conditioning.
- Combinatorial explosion: 72 conditions as base, two-stage tuning.

8.20 YAML sketch (example)
```yaml
dataset: cifar10
model:
  name: resnet18_cifar
train:
  batch_size: 128
  epochs_max: 200
  seed: 0
  precision: fp32
optimizer:
  base: sgd_momentum
  lr: 0.1
  momentum: 0.9
  weight_decay: 5.0e-4
  schedule: cosine
modules:
  step_control:
    name: eoss
    beta: 0.7
    hvp_interval: 200
    ema: 0.99
    lr_clip: [1.0e-4, 1.0]
  geometry:
    name: ggnc
    mode: layerwise
    rho_quantile: 0.9
  outer_accel:
    name: anderson
    m: 3
    interval: 10
    damping: 0.5
  sparsity:
    name: linbreg
    lambda: 1.0e-4
    update_interval: 1
logging:
  backend: tensorboard
  log_interval_steps: 100
  eval_interval_epochs: 1
targets:
  acc: [0.85, 0.90, 0.92, 0.94]
```

8.21 Recommended execution order
- Establish baselines (SGD/AdamW) with time-to-target working.
- Add wrappers one by one: GGNC -> EoSS -> Backtracking -> Anderson -> LinBreg.
- Run 72-condition grid with a single seed for screening.
- Re-run top configs with 3 seeds.
- Report separate gains for low target (0.90) vs high target (0.94).
