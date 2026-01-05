# Task Ticket: Grad-Speedup Config Schema + Grid Generator

## 1) Meta
- ticket_id: task-20260105-grad-speedup-configs
- role/agent: implementer-grad-speedup-configs
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 120 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Define a config schema (YAML or JSON) for grad-speedup runs and add a loader to
scripts/run_cifar10.py. Provide a grid generator to create experiment configs
for baseline and Module C variants without touching ROS-ALTH code.

## 3) Background / Context
We have a CLI-only runner. For repeatable experiment grids and later modules,
we need a config-driven run path.

## 4) Scope
In scope:
- Config schema for dataset, model, optimizer, logging, targets, and Module C.
- Loader that maps config -> CLI args internally.
- Grid generator to emit configs for baseline/Module C variants.

Out of scope:
- Implementing Module B/D/E algorithms.

## 5) Requirements
Must:
- Keep default CLI behavior unchanged if no config is passed.
- Config includes targets and early_stop mode.
- Generated configs write to project/grad-speedup/configs/.

Should:
- Support a simple override mechanism (CLI flag to override config values).

## 6) Acceptance Criteria
- [ ] New config file format documented in project/grad-speedup/README.md.
- [ ] run_cifar10.py accepts --config and runs from a config file.
- [ ] Grid generator script creates baseline + Module C configs.
- [ ] No changes to ROS-ALTH files.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/scripts/run_cifar10.py
- project/grad-speedup/scripts/generate_configs.py (new)
- project/grad-speedup/configs/ (new outputs)

Suggested config fields:
- dataset: {name, data_dir, val_size, batch_size, num_workers, seed, download}
- model: {name}
- optimizer: {type, lr, momentum, weight_decay}
- train: {epochs, deterministic, device}
- logging: {log_interval_steps, eval_interval_epochs, warmup_steps, measure_steps}
- targets: [0.85, 0.90, 0.92, 0.94]
- modules: {step_rule, step_l0, step_l1, step_curv_every, step_eoss_beta}

## 8) Commands
```
cd project/grad-speedup
python scripts/generate_configs.py --out-dir configs
python scripts/run_cifar10.py --config configs/baseline-sgd.json --run-id smoke-config
```

## 9) Deliverables
- Config schema docs and example config(s).
- Grid generator script.

## 10) Risks
- Schema drift; keep fields minimal and aligned with current runner flags.
