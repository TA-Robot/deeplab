# Grad-Speedup (Isolated Codebase)

This directory is a clean-slate codebase for the grad-speedup track.
It is intentionally isolated from ROS-ALTH experiments and does not import
from project/src or run_mnist_experiment.py.

Layout
- src/: grad-speedup-only model/data/train code
- scripts/: launch and utility scripts
- configs/: JSON/YAML configs for runs
- reports/: generated summaries (JSON/CSV)

Runs and artifacts
- Runtime outputs belong under project/runs/grad-speedup/ (not committed).

Queue runner (sequential execution)
- Use the local queue to avoid manual start/stop waits between runs.
- Queue file (ignored by git): project/grad-speedup/queue/queue.txt (see queue.example.txt).
- Add a job:
  - project/grad-speedup/scripts/queue_add.sh "python -u grad-speedup/scripts/run_cifar10.py --run-id ..."
- Run the queue (watch mode):
  - project/grad-speedup/scripts/queue_run.sh --watch
- Dashboard can display queued jobs when the queue file path is set (default is prefilled).

Quickstart
```
cd project/grad-speedup
python scripts/run_cifar10.py --model resnet18 --optimizer sgd --epochs 1 --run-id 20260105-grad-speedup-cifar10-smoke
```

Defaults (important)
- SGD momentum defaults to 0.0 for this track’s fixed-LR/no-schedule comparisons.
- Test eval defaults to step-based cadence: `--eval-interval-steps 200` and `--eval-interval-epochs 0`.
- Target thresholds default to `0.80,0.85,0.90,0.92,0.94` (configure per budget).

Batch launch
```
cd project/grad-speedup
bash scripts/launch_cifar10_grad_speedup.sh
```

Config-driven runs
```
cd project/grad-speedup
python scripts/generate_configs.py --out-dir configs
python scripts/run_cifar10.py --config configs/baseline-sgd.json --run-id 20260105-grad-speedup-cifar10-config-smoke
```

Grid runner
```
cd project/grad-speedup
bash scripts/run_grid.sh configs
```

Report build
```
cd project/grad-speedup
python scripts/build_grad_speedup_report.py --runs-dir ../runs/grad-speedup --output reports/grad-speedup-report.json
```

Dashboard
```
cd project/grad-speedup
pip install -r dash/requirements.txt
python dash/app.py
```

Legacy Streamlit dashboard
```
cd project/grad-speedup
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

CSV columns (reports/grad-speedup-report.csv)
- run_id, seed, model, optimizer, mean_step_time_sec
- target, steps_to_target, time_to_target_sec, cost_to_target_sec

Notes
- Use --output-root runs/grad-speedup to keep artifacts isolated.
- For multiple seeds, pass --seeds 0,1,2 and omit --seed.
- Step-control methods follow the paper definitions; see project/docs/grad-speedup/method-conformance.md.
- For paper-accurate step-control runs, keep direction/clip/sparsity disabled unless the paper states otherwise.
- GD-based step-control rules (l0l1, sps, silver, adaptive-backtracking, sagd) force SGD momentum=0.
- sps-momentum uses a heavy-ball update with its own beta and ignores optimizer momentum.
- SPS and SAGD step sizes are absolute (optimizer LR ignored after λ0); l0l1/silver scale by optimizer LR (η).
- SAGD uses an extra gradient evaluation on the previous batch; optimizer LR is treated as λ0 (paper default 1e-3).
- Module A flags (optional):
  - --sparsity {none,linbreg}
  - --sparsity-lambda, --sparsity-update-interval
- Module C flags (optional):
  - --step-rule {none,l0l1,sps,sps-momentum,adaptive-backtracking,sagd,silver}
  - --step-l0, --step-l1 (l0l1 rule)
  - --step-fstar (sps/sps-momentum lower bound)
  - --step-sps-beta, --step-sps-c, --step-sps-max (sps-momentum rule; step-sps-max is optional cap)
  - --step-backtrack-c, --step-backtrack-max, --step-backtrack-rho (adaptive-backtracking rule)
  - --step-silver-rho (silver schedule)
  - --step-sagd-delta (SAGD Variant III exponent; default 1e-2)
- Module B flags (optional):
  - --direction {none,diag-precond,gn-layerwise,gn-layerwise-exact,shampoo,soap,sophia,muon}
  - diag-precond: --direction-beta, --direction-eps, --direction-update-every
  - gn-layerwise (proxy): diagonal empirical Fisher/EMA(g^2); not paper-accurate GN
    - --direction-beta, --direction-damping, --direction-eps, --direction-update-every
    - --direction-max-size (0 disables scalar fallback for large layers)
  - gn-layerwise-exact: per-layer GGN + CG solve + Armijo line search
    - --direction-damping (Tikhonov), --gn-cg-iters, --gn-cg-tol
    - --gn-layer-mode {all,topk,bottomk,randomk}, --gn-layer-k, --gn-layer-random-every-step/--no-gn-layer-random-every-step
    - --gn-update-interval (reuse last GN update for intermediate steps; experimental)
    - uses --step-backtrack-c/--step-backtrack-rho/--step-backtrack-max for line search
  - shampoo: --direction-beta, --direction-damping, --direction-update-every
  - soap: --direction-beta1, --direction-beta, --direction-eps, --direction-damping, --direction-update-every
  - direction-update-every sets the preconditioning cadence (f); SOAP refreshes eigenvectors, Shampoo refreshes inverse roots.
  - direction-damping adds diagonal damping before inverse-root/eigenvector updates.
  - sophia: --sophia-beta1, --sophia-beta2, --sophia-gamma, --sophia-eps, --sophia-hessian-every, --sophia-hutchinson-samples
  - muon: --muon-beta, --muon-eps, --muon-ns-iters, --muon-scale-mode {none,baseline,update-norm,adjusted-lr}, --muon-rms-scale, --muon-hidden-size
  - muon-hidden-size is required when using --muon-scale-mode baseline (sqrt(H) scaling).
- Module D flags (optional):
  - --clip-mode {none,ggnc,ggnc-global,ggnc-layerwise,global,layerwise}
  - --clip-rho (GGNC rho; tau = min(1, rho / ||d||_*))
  - --clip-alpha (GGNC momentum alpha; d_k = alpha * g_k + (1 - alpha) * d_{k-1})
  - ggnc aliases: ggnc -> ggnc-global, global/layerwise kept for compatibility
  - ggnc-global uses the L2 sharp operator (d^sharp = d); ggnc-layerwise uses per-tensor L2 with
    the product max norm (v_k is per-tensor normalized, ||d||_* is the sum of tensor norms)
- Module E flags (optional):
  - --anderson-memory, --anderson-interval, --anderson-damping, --anderson-lambda
  - anderson_damping is the Algorithm 1 mixing parameter beta_t; anderson_lambda is ridge regularization
- Parametrization flags (optional):
  - --param-mode {none,relora}
  - ReLoRA (arXiv:2307.05695): periodically merge/reset low-rank adapters to reduce the number of trained parameters (and optimizer state).
    - --relora-scope {linear,resnet-layer4,resnet-layer3-4,all}
    - --relora-rank, --relora-alpha, --relora-dropout
    - --relora-merge-interval, --relora-warmstart-steps
    - --relora-reset-optimizer/--no-relora-reset-optimizer, --relora-prune-optimizer-fraction
- Diagnostics:
  - --diagnostics (enables data-wait and max-memory stats)
- Method metrics:
  - step_size_* and line_search_* report step-control stats when enabled
  - clip_coef_* are GGNC tau stats; sophia_hessian_*, sophia_clip_frac_*, muon_ortho_iters_*, precond_* appear in metrics.jsonl when enabled
  - gn_update_time_s/gn_apply_time_s and gn_layer_stats appear for gn-layerwise-exact; step logs include gn_selected_count/gn_selected_layers and gn_update_time_ms/gn_apply_time_ms
  - precond_layer_stats includes per-layer update/apply counts plus timing fields (stat_update_time_s, precond_update_time_s, apply_time_s).
  - modules.step_control.name, modules.step_control.l0, modules.step_control.l1, modules.step_control.fstar
  - modules.step_control.sps_beta, modules.step_control.sps_c, modules.step_control.sps_max
  - modules.step_control.backtrack_c, modules.step_control.backtrack_max, modules.step_control.backtrack_rho
  - modules.step_control.silver_rho, modules.step_control.sagd_delta
  - modules.direction.name, modules.direction.beta, modules.direction.beta1, modules.direction.eps, modules.direction.damping, modules.direction.update_every, modules.direction.max_size
  - modules.direction.gn_cg_iters, modules.direction.gn_cg_tol
  - modules.direction.gn_layer_mode, modules.direction.gn_layer_k, modules.direction.gn_layer_random_every_step
  - modules.direction.gn_update_interval
  - modules.direction.sophia_beta1, modules.direction.sophia_beta2, modules.direction.sophia_gamma
  - modules.direction.sophia_eps, modules.direction.sophia_hessian_every, modules.direction.sophia_hutchinson_samples
  - modules.direction.muon_beta, modules.direction.muon_eps, modules.direction.muon_ns_iters
  - modules.direction.muon_scale_mode, modules.direction.muon_rms_scale, modules.direction.muon_hidden_size
  - modules.sparsity.name, modules.sparsity.lambda, modules.sparsity.update_interval
- Sparsity metrics (metrics.jsonl):
  - sparsity_fraction, dense_flops, effective_flops, sparsity_updates, sparsity_update_rate
- FLOPs note: dense_flops/effective_flops are forward conv/linear MACs per step, scaled by active weight fraction.

Related docs
- project/docs/grad-speedup/cifar10-implementation-spec.md
- project/docs/experiment-20260105-grad-speedup-cifar10.md
- project/docs/grad-speedup/temp-materials-summary.md
- project/docs/grad-speedup/method-conformance.md
