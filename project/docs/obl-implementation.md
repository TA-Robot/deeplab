# Operator Basis Layer Implementation Notes

This document maps the ROS-ALTH operator library design to the current
implementation in `project/src/operator_basis.py` and how to configure it.

## Layer placement

- MLPWithOBL inserts an OBL block after every hidden layer.
- CNNWithOBL inserts an OBL block after every fully connected layer.

## Core mixing rule

Each operator output is normalized and mixed with a continuous coefficient:

```
h' = h + gamma * sum_k beta_k * Norm(f_k(h))
```

- `beta_k` is learnable per operator.
- `gamma` is a residual scale (fixed by default).
- Norm is LayerNorm or RMSNorm (configurable).

## Operator families covered

1) Frequency / RFF
   - Shared random linear projection + sin/cos over multiple scales

2) Polynomial / rational / gate
   - Polynomial degrees (2, 3, 4)
   - Rational x / (1 + alpha * x^2)
   - Gate: u * sigmoid(v) from shared linear projections

3) Diffusion / blur
   - 1D Laplacian diffusion along feature dimension
   - 1D Gaussian blur along feature dimension

4) Soft pooling / neighborhood
   - Log-sum-exp pooling over local 1D neighbors
   - Soft neighbor aggregation over local 1D neighbors

5) RBF prototypes
   - RBF features against random prototypes, projected back to input dim

6) Soft ranking / assignment
   - SoftSort (Sinkhorn-normalized proximity to a reference order)
   - Sinkhorn assignment against a random reference vector

7) Attention (lightweight)
   - Scalar attention over feature positions with softmax temperature

8) Random program composition
   - Random sequences of primitives (depth 2-4)
   - Optional residual skip per program

## Random program primitives

- linear, sin, cos, gelu, tanh, sigmoid
- poly2, poly3
- rational, diffusion
- softpool, softneighbor
- rbf, softsort, sinkhorn
- attention

## Configuration highlights

- `--obl-profile full|mini` selects operator coverage.
- `--obl-programs N` overrides number of random programs.
- `--obl-seed` fixes operator initialization and program sampling.
- `--obl-norm layernorm|rmsnorm` chooses normalization.

## Notes

- Operators are fixed by default; enable learning via config fields
  (`learnable_shared`, `learnable_gates`) if needed.
- The implementation uses 1D feature-neighborhood operators as a practical
  CPU-friendly approximation for diffusion and pooling primitives.
- Parameter count is reported as both trainable and total (including buffers).
