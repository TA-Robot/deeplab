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

Quickstart
```
cd project/grad-speedup
python scripts/run_cifar10.py --model resnet18 --optimizer sgd --epochs 1 --run-id 20260105-grad-speedup-cifar10-smoke
```

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

CSV columns (reports/grad-speedup-report.csv)
- run_id, seed, model, optimizer, mean_step_time_sec
- target, steps_to_target, time_to_target_sec, cost_to_target_sec

Notes
- Use --output-root runs/grad-speedup to keep artifacts isolated.
- For multiple seeds, pass --seeds 0,1,2 and omit --seed.
- l0l1/eoss step rules are legacy placeholders until paper-accurate versions are implemented.
- Module C flags (optional):
  - --step-rule {none,l0l1,eoss,silver}
  - --step-l0, --step-l1 (l0l1 rule)
  - --step-curv-every, --step-eoss-beta (eoss rule)
  - --step-silver-rho (silver schedule)
- Module B flags (optional):
  - --direction {none,diag-precond}
  - --direction-beta, --direction-eps, --direction-update-every
- Module D flags (optional):
  - --clip-mode {none,global,layerwise}
  - --clip-rho
- Module E flags (optional):
  - --anderson-memory, --anderson-interval, --anderson-damping, --anderson-lambda
- Diagnostics:
  - --diagnostics (enables data-wait and max-memory stats)

Related docs
- project/docs/grad-speedup/cifar10-implementation-spec.md
- project/docs/experiment-20260105-grad-speedup-cifar10.md
- project/docs/grad-speedup/temp-materials-summary.md
