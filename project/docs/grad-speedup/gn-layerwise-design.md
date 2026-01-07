# Layerwise GN Design Note (Proxy v0)

Purpose
- Provide a lightweight stand-in for layerwise Gauss–Newton while the paper-accurate method is being developed.
- Keep the implementation safe and observable (preconditioner stats and timing).

Paper reference
- Full + Layerwise GN: `project/docs/grad-speedup/papers/arxiv-2510.09378.pdf`.
- The paper’s layerwise GN is a block-diagonal approximation of the full GN matrix (per-layer Taylor expansions).

Paper-accurate target (not implemented yet)
- GN curvature: G = J^T H_l J (Section 4.2, page 4).
- Update: Δθ = (G + λI)^{-1} g, θ_{t+1} = θ_t − η Δθ (Section 4.3, page 5 / Algorithm 1 page 3).
- Layerwise GN: replace G with block-diagonal per-layer GN (Section 6.3, pages 8–9).

Proxy implementation (current code)
- Direction name: `gn-layerwise`.
- Actual math: diagonal empirical Fisher / EMA(g^2) with damping:
  - diag_ggn ← β diag_ggn + (1−β) g^2
  - precondition: g ← g / (diag_ggn + damping)
- Large layers can fall back to scalar scaling using mean(g^2).
- This is **not** paper-accurate GN; treat as experimental / proxy only.

Config knobs
- `direction_beta`: EMA coefficient for diag_ggn (0–1).
- `direction_damping`: additive damping.
- `direction_eps`: clamp for denominators.
- `direction_update_every`: update cadence (every K steps).
- `direction_max_size`: if >0, switch to scalar scaling when param.numel() exceeds this threshold.

Guardrails
- 1D parameters are skipped.
- Non-finite denominators zero out gradients for safety.
- All preconditioner stats are logged in `precond_layer_stats`.

Next steps (paper-accurate)
- Implement per-layer GN blocks using layerwise Jacobians and loss Hessian (J^T H_l J).
- Add a new direction name (e.g., `gn-layerwise-exact`) and keep proxy separate.
- Validate against a small synthetic model before CIFAR-10 runs.
