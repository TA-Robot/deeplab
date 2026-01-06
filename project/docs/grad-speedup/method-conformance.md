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
| Direction / preconditioning | SOAP | arXiv:2409.11321 | audited | optional | Algorithm 3 (SOAP) + Algorithm 4 (Eigenvectors), page 6 |
| Direction / preconditioning | K-FAC | arXiv:1503.05671 | audited | optional | Fisher block approx |
| Direction / preconditioning | Sophia | arXiv:2305.14342 | audited | optional | Algorithm 3 + Eq. (6), pages 3 & 6 |
| Direction / preconditioning | Muon (scalable) | arXiv:2502.16982 | audited | optional | Eq. (1)–(2) + scalable variants Eq. (3)–(7), pages 2–4 |
| Direction / preconditioning | MARS | arXiv:2411.10438 | audited | optional | Variance-reduced preconditioning |
| Direction / preconditioning | Adan | arXiv:2208.06677 | audited | optional | Momentum variant |
| Direction / preconditioning | AdEMAMix | arXiv:2409.03137 | audited | optional | Mixed EMA |
| Direction / preconditioning | Lion | arXiv:2302.06675 | audited | optional | Sign momentum |
| Direction / preconditioning | Schedule-Free AdamW | arXiv:2405.15682 | audited | optional | No explicit LR schedule |
| Direction / preconditioning | LARS | arXiv:1708.03888 | audited | optional | Large-batch scaling |
| Direction / preconditioning | LAMB | arXiv:1904.00962 | audited | optional | Layerwise adaptive scaling |
| Direction / preconditioning | CLEAN (Nystrom) | OpenReview ICLR 2026 (wNh0sE9QWD) | audited | optional | Two-sided preconditioning |
| Step control | EoSS (Edge of Stochastic Stability) | arXiv:2412.20553 | audited | core | Stability measurement only (no optimizer step rule) |
| Step control | L0L1-GD | arXiv:2409.14989; arXiv:2410.10800 | implemented | core | Algorithm 1 ((L0,L1)-GD), page 7 |
| Step control | SPS (Polyak step size) | arXiv:2409.14989; arXiv:2406.04142 | implemented | core | Algorithm 2 (GD-PS), page 9 |
| Step control | SPS + Momentum | arXiv:2406.04142 | implemented | core | SPS + heavy-ball momentum |
| Step control | Adaptive Backtracking | arXiv:2408.13150 | implemented | core | Armijo violation (4a) + adaptive factor (4b), page 3 |
| Step control | Silver step sizes | arXiv:2309.16530; PMLR 247 (2024) | implemented | core | Step-size schedule uses v(t)=max{v:t>=F_v} and eta_t = eta*(1+rho^{v(t)-1}) |
| Step control | Stochastic Adaptive GD Without Descent | arXiv:2509.14969 | implemented | core | Variant III adaptive step size with extra gradient eval |
| Step control | Armijo line search | TBD | pending | optional | Non-uniform smoothness |
| Stability / geometry | GGNC | arXiv:2506.01913 | audited | core | Eq. (GGNC) + Algorithm 1, pages 3–4 |
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

Audit notes (2026-01-05)

Primary papers are now stored locally under `project/docs/grad-speedup/papers/`.
Use `project/docs/grad-speedup/papers/README.md` as the paper index.

Core method specs (paper-accurate; implemented in grad-speedup)

EoSS (Edge of Stochastic Stability) (arXiv:2412.20553)
- Definition 3 (Eq. 3), page 10:
  - Batch Sharpness(θ) := E_{B∼P_b}[∇L_B(θ)^T H(L_B) ∇L_B(θ) / ||∇L_B(θ)||^2].
- Usage: diagnostic only (no optimizer update rule). Estimate via HVP along mini-batch gradient.

(L0, L1)-smoothness / (L0,L1)-GD (arXiv:2409.14989; arXiv:2410.10800)
- Algorithm 1 ((L0,L1)-GD), page 7 (arXiv:2409.14989):
  - x_{k+1} = x_k − (η / (L0 + L1 ||∇f(x_k)||)) ∇f(x_k).
- Algorithm 2 (GD-PS / Polyak stepsize), page 9 (arXiv:2409.14989):
  - x_{k+1} = x_k − ((f(x_k)−f*) / ||∇f(x_k)||^2) ∇f(x_k).
- SPS (stochastic Polyak) uses the same rule with mini-batch loss/gradients.
- Hyperparameters: L0, L1, η for (L0,L1)-GD; f* (or estimate) for GD-PS/SPS.

SPS + Momentum (arXiv:2406.04142)
- Algorithm 1 (SHB with MomSPS_max), Appendix B:
  - γ_t = (1−β) min{ c (f_{S_t}(x_t)−ℓ*_S_t) / ||∇f_{S_t}(x_t)||^2, γ_b }.
  - x_{t+1} = x_t − γ_t ∇f_{S_t}(x_t) + β (x_t − x_{t−1}).
- Hyperparameters: β, c, γ_b, lower bound ℓ* (f*); ε for numerical safety.

Stochastic Adaptive GD Without Descent (arXiv:2509.14969)
- Algorithm 1 (page 2): extra gradient evaluation on previous batch.
  - Evaluate ∇f_{ξ_k}(x_k) and ∇f_{ξ_{k-1}}(x_k).
- Local Lipschitz estimate (Eq. 4): L̂_{k−1} = ||∇f_{ξ_{k-1}}(x_k) − ∇f_{ξ_{k-1}}(x_{k−1})|| / ||x_k − x_{k−1}||.
- Step size (Variant III, Eq. 4):
  - λ_k = min( 1 / (2√2 L̂_{k−1} k^{1/2+δ}),
    λ_{k−1} √(1 + (1 − 1/k^{1/2+δ}) (λ_{k−1}/λ_{k−2})) ).
- Initialization (Algorithm 1): λ_0 given; λ_1 = ||x_1 − x_0|| / (2√2 ||∇f_{ξ_0}(x_1) − ∇f_{ξ_0}(x_0)||).
- Update: x_{k+1} = x_k − λ_k ∇f_{ξ_k}(x_k).
- Hyperparameters: λ_0, δ ∈ (0, 1/2).

Adaptive Backtracking Line Search (arXiv:2408.13150)
- Armijo condition (Section 2.4, page 3):
  - F(x_k + α_k d_k) − F(x_k) ≤ c α_k ⟨∇F(x_k), d_k⟩.
- Violation (Eq. 4a): v(α_k) := [F(x_k + α_k d_k) − F(x_k)] / [c α_k ⟨∇F(x_k), d_k⟩].
- Adaptive factor (Eq. 4b): ρ̂(v(α_k)) := max(ε, ρ^{(1−c)/(1−c v(α_k))}).
- Backtracking loop (Algorithm 2): while v(α_k) < 1, set α_k ← ρ̂(v(α_k)) α_k.
- Hyperparameters: c, ρ, ε.

GGNC (Generalized Gradient Norm Clipping) (arXiv:2506.01913)
- Method definition (Eq. GGNC), page 3:
  - τ_k = min{1, ρ / ||d_k||_*}, x_{k+1} = x_k − γ τ_k [d_k]^♯.
  - Equivalent: x_{k+1} = x_k + γ η_k lmo(d_k) with η_k = min{ρ, ||d_k||_*}.
- Algorithm 1 (GGNC), page 4:
  - d_k = α_k ∇f(x_k, ξ_k) + (1−α_k) d_{k−1}, v_k = −lmo(d_k),
    η_k = min{ρ, ⟨d_k, v_k⟩}, x_{k+1} = x_k − γ η_k v_k.
- Hyperparameters: γ, ρ, α_k.

SOAP (arXiv:2409.11321)
- Algorithm 3 (SOAP), page 6:
  - Rotate gradients with Q_L, Q_R: G′ = Q_L^T G Q_R.
  - M ← β1 M + (1−β1) G; M′ = Q_L^T M Q_R.
  - V ← β2 V + (1−β2)(G′⊙G′).
  - N′ = M′ / sqrt(V̂ + ε); N = Q_L N′ Q_R^T; W ← W − η N.
  - L ← β2 L + (1−β2) G G^T; R ← β2 R + (1−β2) G^T G.
  - Every f steps: update eigenvectors via Algorithm 4 (power iteration + QR).
- Hyperparameters: β1, β2, ε, preconditioning frequency f.

Sophia (arXiv:2305.14342)
- Algorithm 3 (page 3) + Eq. (6) (page 6):
  - m_t = β1 m_{t−1} + (1−β1) g_t.
  - Hessian EMA: if t mod k = 1, h_t = β2 h_{t−k} + (1−β2) ĥ_t; else h_t = h_{t−1}. (Eq. 5)
  - Weight decay: θ_t ← θ_t − η_t λ θ_t.
  - Update: θ_{t+1} = θ_t − η_t * clip(m_t / max{γ h_t, ε}, 1). (Eq. 6)
- Hyperparameters: β1, β2, γ, ε, k, λ.

Muon (scalable) (arXiv:2502.16982)
- Base update (Eq. 1), page 2:
  - M_t = μ M_{t−1} + ∇L_t(W_{t−1})
  - O_t = Newton-Schulz(M_t) ≈ (M_t M_t^T)^{−1/2} M_t
  - W_t = W_{t−1} − η O_t
- Newton-Schulz iteration (Eq. 2), page 3:
  - X_0 = M_t / ||M_t||_F
  - X_k = a X_{k−1} + b (X_{k−1} X_{k−1}^T) X_{k−1} + c (X_{k−1} X_{k−1}^T)^2 X_{k−1}
  - a=3.4445, b=−4.7750, c=2.0315 (paper defaults).
- Scalable variants (Eq. 3–7): add weight decay and update RMS scaling; see pages 3–4.
- Hyperparameters: μ, η, N (NS steps), λ, scaling mode.

Silver step sizes (PMLR 247, 2024)
- eta_t = eta * (1 + rho^{v(t)-1}), v(t) = max{v: t ≥ F_v}, F_0=0, F_1=1.

Implementation note (current repo state)
- Step-control implementations in `project/grad-speedup/src/train.py` follow the equations above for L0L1-GD, SPS (Polyak step size without extra η), SPS+Momentum (MomSPS_max), ABLS, Silver, and SAGD.
- EoSS remains diagnostic only (no optimizer update rule).
- SGD momentum is zeroed for GD-based step rules (l0l1, sps, silver, adaptive-backtracking, sagd); sps-momentum uses its own beta.
- For paper-accurate step-control runs, keep direction/clip/sparsity disabled; code enforces direction='none' for adaptive-backtracking, sps-momentum, sagd, and sparsity='none' for adaptive-backtracking/sagd.

Pending method specs (awaiting primary paper extraction)
- Armijo line search (TBD)
- Multilevel mirror descent (TBD)
- BSPPA (TBD)
- Parametrization / implicit preconditioning (TBD)
- Generalized smoothness / H-smooth (TBD)
