# Grad-Speedup Combination Experiment Plan

Purpose
- Move from single-module validation to staged combinations without exploding the search space.
- Use a gated funnel: smoke → short-run ranking → targeted combos → multi-seed validation.

Gates (must pass before entering combos)
- Paper-audited spec in `project/docs/grad-speedup/method-conformance.md`
- Paper-accurate implementation merged + smoke run logged in `project/docs/experiment-log.md`

Stage 1: Step-control sweep (done)
- Model: small-cnn, CIFAR-10, 1 epoch, 1 seed, GPU.
- Methods: {baseline, l0l1, sps, sps-momentum, adaptive-backtracking, sagd, silver}.
- Output: rank by cost-to-target at 0.85 (or lowest loss at 1 epoch if target not reached).

Stage 2: Direction sweep (next)
- Model: small-cnn, CIFAR-10, 1 epoch, 1 seed, GPU.
- Step control: none.
- Directions: {none, diag-precond, shampoo, soap, sophia, muon}.
- Output: rank by cost-to-target at 0.85 (or lowest loss at 1 epoch).

Stage 3: Pairwise combos (Step-control × Direction)
- Select top-2 step-control methods from Stage 1 that allow direction != none.
- Combine each with all directions (Stage 2 list).
- Note: adaptive-backtracking, sps-momentum, sagd require direction=none for paper-accurate updates.
- Budget: 1 epoch, 1 seed, small-cnn, GPU.
- Output: rank by cost-to-target; identify top-1 pair.

Stage 4: Add stability/outer
- Base: top-1 pair from Stage 3.
- Clip sweep: {none, ggnc-global, ggnc-layerwise}.
- Outer sweep: {none, anderson}.
- Budget: 1 epoch, 1 seed, small-cnn, GPU.
- Output: rank by cost-to-target; pick 1–2 winners.
- Runner: `project/grad-speedup/scripts/launch_stage4_sweep.sh` (templated CLI).

Stage 5: Promotion
- Model: resnet18.
- Budget: 5 epochs, 3 seeds, GPU.
- Output: final recommendation with cost-to-target and guardrail accuracy.

Run naming
- `YYYYMMDD-grad-speedup-<stage>-<method>-<model>-seed<seed>`
- Example: `20260106-grad-speedup-stage3-sps-soap-smallcnn-seed0`

Notes
- Avoid combining unverified modules.
- Keep direction/clip/sparsity disabled unless explicitly tested in the stage.
- Log all runs to `project/docs/experiment-log.md` and `project/runs/grad-speedup/`.
