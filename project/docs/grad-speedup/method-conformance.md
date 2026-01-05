# Method Conformance Matrix (Grad-Speedup)

This file records paper-accurate requirements for every method mentioned in the temp materials.
Populate paper sources and implementation requirements during the audit phase.

Legend
- Status: pending | audited | designed | implemented
- Scope: core (in-scope for CIFAR-10 framework) | optional (research) | out-of-scope

Table
| Category | Method | Paper(s) | Status | Scope | Notes |
| --- | --- | --- | --- | --- | --- |
| Direction / preconditioning | Full Gauss-Newton (GN) | arXiv:2510.09378 | audited | optional | Large-batch upper bound; heavy compute |
| Direction / preconditioning | Layerwise GN | arXiv:2510.09378 | audited | optional | Layerwise block approximation |
| Direction / preconditioning | Exact Gauss-Newton (EGN) | arXiv:2405.14402 | audited | optional | Low-rank / batch-space update |
| Direction / preconditioning | Dual NGD | arXiv:2505.21404 | audited | optional | Residual-space solve |
| Direction / preconditioning | Shampoo | arXiv:1802.09568 | audited | optional | Kronecker preconditioning |
| Direction / preconditioning | Distributed Shampoo | PMLR 139 (2021) | audited | optional | Distributed variant (AlgoPerf) |
| Direction / preconditioning | SOAP | arXiv:2409.11321 | audited | optional | Shampoo eigenbasis Adam |
| Direction / preconditioning | K-FAC | arXiv:1503.05671 | audited | optional | Fisher block approx |
| Direction / preconditioning | Sophia | arXiv:2305.14342 | audited | optional | Diagonal Hessian + clipping |
| Direction / preconditioning | Muon (scalable) | arXiv:2502.16982 | audited | optional | Orthogonalization-based update |
| Direction / preconditioning | MARS | arXiv:2411.10438 | audited | optional | Variance-reduced preconditioning |
| Direction / preconditioning | Adan | arXiv:2208.06677 | audited | optional | Momentum variant |
| Direction / preconditioning | AdEMAMix | arXiv:2409.03137 | audited | optional | Mixed EMA |
| Direction / preconditioning | Lion | arXiv:2302.06675 | audited | optional | Sign momentum |
| Direction / preconditioning | Schedule-Free AdamW | arXiv:2405.15682 | audited | optional | No explicit LR schedule |
| Direction / preconditioning | LARS | arXiv:1708.03888 | audited | optional | Large-batch scaling |
| Direction / preconditioning | LAMB | arXiv:1904.00962 | audited | optional | Layerwise adaptive scaling |
| Direction / preconditioning | CLEAN (Nystrom) | OpenReview ICLR 2026 (wNh0sE9QWD) | audited | optional | Two-sided preconditioning |
| Step control | EoSS (Edge of Stochastic Stability) | arXiv:2412.20553 | audited | core | Directional curvature via HVP |
| Step control | L0L1-GD | arXiv:2410.10800 | audited | core | Generalized smoothness step rule |
| Step control | Adaptive Backtracking | arXiv:2408.13150 | audited | core | Armijo-like stochastic backtracking |
| Step control | Silver step sizes | arXiv:2309.16530; PMLR 247 (2024) | designed | optional | Step-size schedule uses v(t)=max{v:t>=F_v} and eta_t = eta*(1+rho^{v(t)-1}) |
| Step control | SPS + Momentum | arXiv:2406.04142 | audited | optional | Stochastic Polyak w/ momentum |
| Step control | Stochastic Adaptive GD Without Descent | arXiv:2405.00582 | audited | optional | Step size from gradients |
| Step control | Armijo line search | TBD | pending | optional | Non-uniform smoothness |
| Stability / geometry | GGNC | arXiv:2506.01913 | audited | core | Generalized gradient norm clipping |
| Outer acceleration | Anderson acceleration | arXiv:1809.02341 | audited | core | Fixed-point extrapolation |
| Outer acceleration | Scieur subspace method | PMLR 238 (2024) | audited | optional | Low-dim curvature via differences |
| Outer acceleration | Block Broyden | arXiv:2306.13542 | audited | optional | Multi-secant updates |
| Outer acceleration | SR1 / regularized cubic | arXiv:2405.16452 | audited | optional | Curvature with negative directions |
| Sparsity / compute | Linearized Bregman | arXiv:1405.2380; arXiv:1905.09449 | audited | optional | Dynamic sparsity operator |
| Sparsity / compute | Multilevel mirror descent | TBD | pending | optional | Alternating sparsity phases |
| Prox / Bregman | SPPM | arXiv:2502.03401 | audited | optional | Stochastic proximal point |
| Prox / Bregman | MSBPG | JMLR 2025 (24-0859) | audited | optional | Nonconvex Bregman prox |
| Prox / Bregman | BSPPA | TBD | pending | optional | Bregman + variance reduction |
| Control / ODE | PIDAO | Nature Comms 15:6221 (2024) | audited | optional | PID control in optimization |
| Compression | PowerSGD | arXiv:1905.13727 | audited | optional | Low-rank gradient compression |
| Compression | Deep Gradient Compression | arXiv:1712.01887 | audited | optional | Gradient sparsification |
| Scaling / batch | Critical Batch Size (CBS) | arXiv:1812.06162 | audited | optional | GNS / CBS estimation |
| Scaling / batch | Super-Convergence / 1cycle | arXiv:1708.07120 | audited | optional | Aggressive LR schedule |
| Scaling / batch | Large-batch scaling notes | arXiv:1708.03888; arXiv:1904.00962 | audited | optional | LARS/LAMB context |
| Efficiency | Mixed precision | arXiv:1710.03740 | audited | out-of-scope | Not in current CIFAR-10 series |
| Efficiency | FlashAttention | arXiv:2205.14135 | audited | out-of-scope | Not in current CIFAR-10 series |
| Efficiency | 8-bit optimizer states | arXiv:2110.02861 | audited | out-of-scope | Not in current CIFAR-10 series |
| Efficiency | ZeRO | arXiv:1910.02054 | audited | out-of-scope | Not in current CIFAR-10 series |
| Meta-optimizer | muLO | arXiv:2502.07645 | audited | optional | Learned optimizer | 
| Meta-optimizer | Celo | arXiv:2311.01818 | audited | optional | Learned optimizer |
| Parametrization | Implicit preconditioning via reparameterization | TBD | pending | optional | Levin viewpoint |
| Theory | Generalized smoothness / H-smooth | TBD | pending | optional | Qi / Bolte smooth adaptivity |
| Theory | Edge of stochastic stability | arXiv:2412.20553 | audited | core | Stability boundary used by EoSS |
| Theory | GN-PROX-LINEAR view | arXiv:2510.09378 | audited | optional | GN perspective |

Audit notes (in progress)

EoSS (arXiv:2412.20553)
- Paper defines Batch Sharpness / MiniBS and the EoSS regime (measurement, not an optimizer).
- Implementation requirement: add batch sharpness estimator and logging; do not claim an EoSS optimizer unless paper specifies update rule.

(L0, L1)-smoothness methods (arXiv:2410.10800)
- Paper studies gradient methods with specific step-size rules (Polyak stepsizes, normalized gradient) and an accelerated variant.
- Implementation requirement: if we expose an "(L0,L1)" module, it must match one of the paper's algorithms (not an ad-hoc heuristic).

GGNC (arXiv:2506.01913)
- Paper proposes a generalized clipping operator and a hybrid steepest-descent / conditional-gradient update.
- Implementation requirement: implement the operator and step rule exactly as specified (non-Euclidean norms and short-step behavior).

Adaptive Backtracking (arXiv:2408.13150)
- Paper defines adaptive backtracking factors based on the degree of violation of Armijo / descent-lemma criteria.
- Implementation requirement: implement adaptive factor update, not fixed shrink factor.

Silver step sizes (PMLR 247, 2024)
- Step-size schedule uses v(t) defined by Fibonacci thresholds F_v (F_0=0, F_1=1).
- Step rule: eta_t = eta * (1 + rho^{v(t)-1}), with rho>1 (silver-ratio schedule). citeturn2view0

SOAP (arXiv:2409.11321)
- Paper reframes Shampoo as Adam in the eigenbasis of Shampoo and specifies preconditioning update cadence.
- Implementation requirement: follow basis update + per-parameter moments as specified; do not approximate with simple RMS scaling.

Sophia (arXiv:2305.14342)
- Paper uses a diagonal Hessian estimate with clipping and a specific update rule.
- Implementation requirement: implement Hessian diagonal estimator and clipping exactly as defined.

Muon (arXiv:2502.16982)
- Paper introduces orthogonalization-based updates for matrix parameters with specific scaling.
- Implementation requirement: include orthogonalization routine (e.g., Newton-Schulz) and scaling rules as in paper.
