# Grad-Speedup Paper References

目的
- 実装・conformance チェックのために、参照すべき論文（arXiv）と「どこを読むべきか」を固定する。
- **PDF は Git にはコミットしない**（大きいバイナリを repo に持ち込まない）。

PDF をローカルに置きたい場合:
- `project/runs/grad-speedup/_papers/` 配下に保存（`project/runs/` はVCS対象外）
- ファイル名は `arxiv-<id>.pdf` を推奨

Core methods (implementation-critical)
- Full + Layerwise Gauss-Newton (GN): arXiv:2510.09378
  - Algorithm 1 (Gauss-Newton method), page 3.
  - Gauss-Newton matrix definition (Section 4.2), page 4: G := J^T H_l J.
  - Gauss-Newton update / preconditioner (Section 4.3), page 5.
  - Layerwise Gauss-Newton (Section 6.3), pages 8–9.
- EoSS (diagnostic only): arXiv:2412.20553
  - Definition 3 (Batch Sharpness): Eq. (3), page 10.
  - Use for logging/diagnostics; no optimizer update rule specified.
- Adaptive Backtracking Line Search (ABLS): arXiv:2408.13150
  - Armijo case: violation v(α_k) Eq. (4a), adaptive factor ρ̂(v) Eq. (4b), Section 2.4 (page 3).
  - Use Algorithm 2 structure (adaptive backtracking) with the Armijo violation definition.
- (L0,L1)-GD and Polyak stepsizes: arXiv:2409.14989
  - Algorithm 1: (L0,L1)-GD update, page 7.
  - Algorithm 2: GD with Polyak stepsizes, page 9.
- GGNC (Generalized Gradient Norm Clipping): arXiv:2506.01913
  - Method definition: Eq. (GGNC), page 3.
  - Algorithm 1 (GGNC) and Algorithm 2 (stochastic short-step), page 4.
- SOAP (Shampoo + Adam in eigenbasis): arXiv:2409.11321
  - Algorithm 3 (SOAP, per-layer) and Algorithm 4 (Eigenvectors), page 6.
- Sophia: arXiv:2305.14342
  - Algorithm 3 (Sophia), page 3.
  - Update rule (Eq. 6) with clipping and Hessian EMA, page 6.
- Muon (scalable): arXiv:2502.16982
  - Base Muon update (Eq. 1) and Newton-Schulz iteration (Eq. 2), pages 2–3.
  - Scalable adjustments with weight decay and update RMS scaling (Eq. 3–7), pages 3–4.

Related / alternate
- arXiv:2410.10800 (Optimizing (L0,L1)-Smooth Functions by Gradient Methods)
  - Complementary treatment of (L0,L1)-smooth methods; use as cross-check with the main (L0,L1)-GD paper.

Additional papers (unblocking pending methods)
- SPS / SPS+Momentum: arXiv:2406.04142
- Stochastic Adaptive GD Without Descent: arXiv:2509.14969
- Linearized Bregman: arXiv:1405.2380, arXiv:1905.09449
- Anderson acceleration: arXiv:1809.02341
- Shampoo (original): arXiv:1802.09568
  - Note: SAGD PDF corrected to arXiv:2509.14969; remove earlier misdownload.
