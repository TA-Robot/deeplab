# Task Ticket: Grad-Speedup Baseline Runs

## 1) Meta
- ticket_id: task-20260105-grad-speedup-baseline-run
- role/agent: implementer-grad-speedup-baseline-run
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 120 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Run baseline SGD and Adam on CIFAR-10 with the isolated runner and record
steps/time/cost to targets for seeds 0/1/2.

## 3) Background / Context
Baselines are needed before module comparisons.

## 4) Scope
In scope:
- Run baseline SGD and Adam with resnet18 or small-cnn (select one and note).
- Record run IDs and add to experiment-log.md.

Out of scope:
- Long hyperparameter sweeps.

## 5) Requirements
Must:
- Use output root ../runs/grad-speedup
- Run seeds {0,1,2}
- Capture summary.json

## 6) Acceptance Criteria
- [ ] Runs complete without errors.
- [ ] summary.json exists per run.
- [ ] experiment-log.md updated.

## 7) Commands
```
cd project/grad-speedup
bash scripts/launch_cifar10_grad_speedup.sh
```

## 8) Deliverables
- Baseline run IDs and log entry.

## 9) Risks
- CPU-only runs may be slow; consider GPU if available.
