# Muon-curv short sweep plan (2026-01-08)

Goal: compare muon baseline vs muon-curv on CIFAR-10 (resnet18) with a 2k-step screen.

Fixed settings
- model: resnet18 (activation=relu)
- optimizer: sgd (lr=0.1, momentum=0.9, weight_decay=5e-4)
- batch_size=128, max_steps=2000
- eval_interval_steps=200, log_interval_steps=100
- warmup_steps=50, measure_steps=200
- targets: 0.60, 0.65, 0.70, 0.75, 0.85
- seed=0, device=cuda:0

Baseline reference
- 20260106-grad-speedup-screen2000-muon-sgd-seed0 (same config; use for comparison)

New runs
1. 20260108-grad-speedup-screen2000-muon-sgd-seed0
   - direction=muon (refresh baseline if needed)
2. 20260108-grad-speedup-screen2000-muon-curv-b0p99-ns5-i1-sgd-seed0
   - direction=muon-curv, beta2=0.99, ns_iters=5, update_interval=1, mode=auto
3. 20260108-grad-speedup-screen2000-muon-curv-b0p9-ns5-i1-sgd-seed0
   - direction=muon-curv, beta2=0.9, ns_iters=5, update_interval=1, mode=auto
4. 20260108-grad-speedup-screen2000-muon-curv-b0p99-ns3-i10-sgd-seed0
   - direction=muon-curv, beta2=0.99, ns_iters=3, update_interval=10, mode=auto

Metrics
- time-to-target for each threshold
- mean step time (ms)
- muon_curv_ns_iters_* for stability/cost monitoring
