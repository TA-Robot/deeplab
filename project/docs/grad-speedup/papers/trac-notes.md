# TRAC (Tensor-Train LoRA with Across-layer shared Core) — Notes

Source
- OpenReview: tz5yPWZp9W (PDF provided by user)

Summary
- TRAC replaces the LoRA matrices A and B with Tensor-Train (TT) decompositions and
  shares one TT core across layers to reduce parameters while retaining capacity.

Core formulation (paper equations)
- LoRA form is W = W0 + ΔW with ΔW = B A. TRAC expresses A and B via TT cores.
- Input projection A (Eq. 5) is a 3-way TT:
  - A = C^A ×1 G1^A ×2 G2^A ×3 G3^A, with
    - C^A ∈ R^{1 × M × n1}
    - G1^A ∈ R^{1 × d1 × r1}
    - G2^A ∈ R^{r1 × n2 × r2}
    - G3^A ∈ R^{r2 × n3 × 1}
- Output projection B (Eq. 6) mirrors the same TT structure with C^B, G1^B, G2^B, G3^B.
  - m = m1 m2 m3 and n = n1 n2 n3 (factorization of in/out dims).
- Two TT cores are trained (G1, G3), while the middle core G2 is frozen.
- The final core G3 is shared across layers; per-layer controllers modulate it:
  - \bar{G3}^A = b1 ×1 G3^A ×2 d1, \bar{G3}^B = b2 ×1 G3^B ×2 d2 (Eq. 8).

Hyperparameters / guidance from paper
- TT ranks r1, r2 are main capacity knobs (separately for A and B).
- Asymmetric rank guidance: r^A >> r^B and n^A << n^B (A higher rank, B lower rank).
- Shared core is global across layers (within the chosen scope).

Initialization details
- TT-Norm initialization (Eq. 7) for TT cores.
- B-side last core O3^B is initialized to zero (keeps ΔW small at start).
- Controller embeddings (b, d) are initialized to zero.

Implementation mapping (ReLoRA integration)
- Treat TRAC as an alternative adapter to ReLoRA:
  - Replace LoRA A/B in ReLoRA layers with TRAC TT factorization.
  - Keep ReLoRA merge/reset schedule (merge every T, reset adapter params).
- Shared-core scope:
  - Default: share G3 across all layers in relora scope.
  - Option: allow per-block sharing (e.g., ResNet stage-level) for ablations.
- Controller params (b, d) are per-layer and trainable.
- G2 cores are frozen to match paper.

Open questions / risks
- Conv2d tensorization: need explicit mapping for m1 m2 m3 / n1 n2 n3.
- Choice of TT order: paper uses 3-way; keep 3-way for CIFAR-10 unless validated otherwise.
