# Task Ticket: Step-based Ablations (ResNet18, max_steps=14,000, seeds 0/1/2)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-step-ablations
- role/agent: implementer-step-ablations
- owner: PM
- created_at: 2026-01-06
- priority: P0
- timebox: GPU runs after baseline+winner
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Run step-based ablations with the same step budget as the baseline/winner to attribute gains.

Success looks like:
- All three ablation runs complete and produce per-seed summaries.
- Run IDs follow the fixed scheme below.

## 3) Background / Context
Epoch-based ablations are superseded. New step-based control is the standard:
- max_steps=14,000
- eval_interval_steps=1000

## 4) Scope
In scope:
- Run the following ablations (ResNet18, max_steps=14,000, seeds 0/1/2):
  1) l0l1-only
  2) soap-only
  3) l0l1 + soap (no anderson)

Out of scope:
- No parameter tuning; no dashboard/report updates.

## 5) Requirements
Must:
- Run IDs:
  - 20260106-grad-speedup-step-l0l1-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-soap-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012
- Use eval_interval_steps=1000, eval_interval_epochs=0.

## 6) Acceptance Criteria
- [ ] Each run has run root + per-seed summaries under project/runs/grad-speedup.
- [ ] Logs saved under project/runs/grad-speedup/_logs.

## 7) Commands
```bash
cd project

# l0l1-only
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-step-l0l1-only-resnet18-maxsteps14000-seeds012 \
  --model resnet18 --epochs 50 --max-steps 14000 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.0 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 0 --eval-interval-steps 1000 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule l0l1 --step-l0 1.0 --step-l1 0.1 --direction none --clip-mode none --sparsity none

# soap-only
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-step-soap-only-resnet18-maxsteps14000-seeds012 \
  --model resnet18 --epochs 50 --max-steps 14000 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.9 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 0 --eval-interval-steps 1000 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule none --direction soap --clip-mode none --sparsity none

# l0l1 + soap (no anderson)
python -u grad-speedup/scripts/run_cifar10.py \
  --run-id 20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012 \
  --model resnet18 --epochs 50 --max-steps 14000 --batch-size 128 --lr 0.1 --optimizer sgd \
  --momentum 0.0 --weight-decay 0.0005 --seeds 0,1,2 --data-seed 123 --val-size 5000 \
  --num-workers 4 --device cuda:0 --log-interval-steps 100 --eval-interval-epochs 0 --eval-interval-steps 1000 \
  --target-acc 0.85,0.9,0.92,0.94 --early-stop max --warmup-steps 50 --measure-steps 200 \
  --step-rule l0l1 --step-l0 1.0 --step-l1 0.1 --direction soap \
  --anderson-memory 0 --anderson-interval 0 --clip-mode none --sparsity none
```

## 8) Reporting
Report run IDs + any anomalies to PM.
