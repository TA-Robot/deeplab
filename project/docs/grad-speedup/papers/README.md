# Grad-Speedup Paper Pack (Local PDFs)

Purpose
- Provide local, paper-accurate sources for implementation and conformance checks.
- All PDFs live alongside this file under `project/docs/grad-speedup/papers/`.

Downloaded: 2026-01-05

Core methods (implementation-critical)
- EoSS (diagnostic only): `arxiv-2412.20553.pdf`
  - Definition 3 (Batch Sharpness): Eq. (3), page 10.
  - Use for logging/diagnostics; no optimizer update rule specified.
- Adaptive Backtracking Line Search (ABLS): `arxiv-2408.13150.pdf`
  - Armijo case: violation v(α_k) Eq. (4a), adaptive factor ρ̂(v) Eq. (4b), Section 2.4 (page 3).
  - Use Algorithm 2 structure (adaptive backtracking) with the Armijo violation definition.
- (L0,L1)-GD and Polyak stepsizes: `arxiv-2409.14989.pdf`
  - Algorithm 1: (L0,L1)-GD update, page 7.
  - Algorithm 2: GD with Polyak stepsizes, page 9.
- GGNC (Generalized Gradient Norm Clipping): `arxiv-2506.01913.pdf`
  - Method definition: Eq. (GGNC), page 3.
  - Algorithm 1 (GGNC) and Algorithm 2 (stochastic short-step), page 4.
- SOAP (Shampoo + Adam in eigenbasis): `arxiv-2409.11321.pdf`
  - Algorithm 3 (SOAP, per-layer) and Algorithm 4 (Eigenvectors), page 6.
- Sophia: `arxiv-2305.14342.pdf`
  - Algorithm 3 (Sophia), page 3.
  - Update rule (Eq. 6) with clipping and Hessian EMA, page 6.
- Muon (scalable): `arxiv-2502.16982.pdf`
  - Base Muon update (Eq. 1) and Newton-Schulz iteration (Eq. 2), pages 2–3.
  - Scalable adjustments with weight decay and update RMS scaling (Eq. 3–7), pages 3–4.

Related / alternate
- `arxiv-2410.10800.pdf` (Optimizing (L0,L1)-Smooth Functions by Gradient Methods)
  - Complementary treatment of (L0,L1)-smooth methods; use as cross-check with the main (L0,L1)-GD paper.

Additional papers (unblocking pending methods)
- SPS / SPS+Momentum: `arxiv-2406.04142.pdf`
- Stochastic Adaptive GD Without Descent: `arxiv-2509.14969.pdf`
- Linearized Bregman: `arxiv-1405.2380.pdf`, `arxiv-1905.09449.pdf`
- Anderson acceleration: `arxiv-1809.02341.pdf`
- Shampoo (original): `arxiv-1802.09568.pdf`
  - Note: SAGD PDF corrected to arXiv:2509.14969; remove earlier misdownload.
