# Muon Improvement Notes (Temp Summary)

Status: draft derived from temp notes; needs primary paper confirmation.

Goal
- Combine curvature-aware, non-diagonal preconditioning (inverse square root via Newton-Schulz)
  with Muon orthogonalization, keeping memory to the smaller side (min(m,n)) per layer.

Algorithm sketch (per layer)
- Given gradient G in R^{m x n} for a weight matrix W.
- Choose side = right if n <= m, else left (one-sided V).
- Form curvature proxy:
  - Right side: V = G^T G.
  - Left side: V = G G^T.
- Damping: V_eps = V + eps I.
- Normalize V_eps to keep eigenvalues <= 1 (trace or Frobenius scaling).
- Approximate inverse square root with Newton-Schulz for K steps to get P ~= V_eps^{-1/2}.
- Precondition:
  - Right side: G_pre = G P.
  - Left side: G_pre = P G.
- Muon orthogonalization on G_pre:
  - O = (G_pre G_pre^T)^{-1/2} G_pre (or right-side variant if m < n).
- Update: W <- W - eta * O (optionally with Muon RMS scaling).

Relation to existing methods
- Muon: set P = I (or K = 0) and keep only the orthogonalization step.
- ASGO: skip Muon orthogonalization and use only G_pre.
- If V = G^T G and Muon orthogonalizes on the left, the combination approximates two-sided whitening.

Hyperparameters (proposed)
- precond_side: auto | left | right
- precond_ns_iters: K (Newton-Schulz steps for preconditioner)
- precond_ns_coeffs: Muon coefficients (a,b,c) unless ASGO specifies different values
- precond_damping_eps
- precond_update_interval
- precond_normalization: trace | frob
- precond_ema_beta (optional smoothing of V)
- muon_ns_iters (existing Muon)
- muon_rms_scale / scaling_mode (existing Muon)
- mode: asgo vs muon_curv (enable or disable Muon orthogonalization)

Suggested API surface
- direction=asgo or direction=muon_curv
- Flags:
  - --precond-side, --precond-ns-iters, --precond-ns-coeffs
  - --precond-eps, --precond-interval, --precond-normalization
  - --muon-ns-iters, --muon-rms-scale
- Per-layer cap: optional max_dim to bypass preconditioning on very large layers.

Logging requirements
- Per-layer preconditioner stats: side, dim, eps, ns_iters, update_ms
- NS convergence proxy: ||I - V * P^2||_F (or a polynomial residual proxy)
- Muon orthogonality error: ||O^T O - I||_F or ||O O^T - I||_F
- Update RMS scale and preconditioner scaling factor
- Cache refresh count for preconditioner (interval hits)

Risks and constraints
- Newton-Schulz stability requires normalization (eigenvalues in [0,1]).
- One-sided preconditioning ignores cross-side curvature; may underperform on rectangular layers.
- Double normalization (whitening then Muon) can shrink updates; monitor RMS scaling.
- Convolution layers must be reshaped consistently with existing Muon handling.

Open questions
- Which NS coefficients to use for the preconditioner (Muon defaults vs ASGO paper)?
- Should Muon orthogonalization be optional or always applied after whitening?
- Preferred refresh cadence for V (interval vs EMA) to avoid instability?
