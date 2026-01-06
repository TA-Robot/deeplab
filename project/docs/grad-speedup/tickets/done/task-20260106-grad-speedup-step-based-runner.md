# Task Ticket: Step-based Training Control + Eval (Runner + Train)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-step-based-runner
- role/agent: implementer-step-based-runner
- owner: PM
- created_at: 2026-01-06
- priority: P0
- timebox: 2–3h
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Switch training control from epochs to steps, and enable evaluation on a step interval (default 1000). Keep backward compatibility with epoch-based configs.

Success looks like:
- `run_cifar10.py` supports `--max-steps` and `--eval-interval-steps`.
- Training stops at max_steps even mid-epoch.
- Test evals are logged at step intervals with `global_step` set.
- Existing epoch-based behavior continues to work when max_steps=0.

## 3) Background / Context
PM direction: prioritize step-based control; evaluation should be configurable, defaulting to 1000-step intervals. Experiments are paused until metrics are confirmed.

## 4) Scope
In scope:
- `project/grad-speedup/scripts/run_cifar10.py` CLI + config output updates.
- `project/grad-speedup/src/train.py` to honor max_steps and call a step callback.
- Metrics logging: step-based evals must be visible in metrics.jsonl (type=epoch, split=test, global_step populated).

Out of scope:
- Dashboard UI changes (separate ticket).
- Rerunning experiments.

## 5) Requirements
Must:
- Add CLI args:
  - `--max-steps` (int, default 0 = disabled)
  - `--eval-interval-steps` (int, default 0 = disabled)
- When `max_steps > 0`, training halts once global_step reaches it.
- When `eval_interval_steps > 0`, run evaluation at step intervals in addition to (or instead of) epoch eval.
- Preserve logging format; add no new dependencies.

Should:
- Avoid double eval on the same step if both epoch and step eval trigger.
- Record `max_steps` and `eval_interval_steps` in `config.json`.

## 6) Acceptance Criteria
- [ ] `run_cifar10.py --help` shows new flags.
- [ ] A short run with `--max-steps 10 --eval-interval-steps 5` logs two test evals in metrics.jsonl.
- [ ] No regression when running with epochs only.

## 7) Implementation Notes
- Add optional `on_step_end` callback to `train_one_epoch` that receives `(global_step, epoch, step_in_epoch)` and may request early stop.
- In `run_cifar10.py`, create a closure to trigger evaluation at `eval_interval_steps` and to stop if max_steps reached.
- When evaluation fires, log via `log_epoch(metrics_path, "test", epoch, global_step, eval_metrics)`.
- Track `last_eval_step` to avoid duplicate eval in the same step.

Files:
- `project/grad-speedup/src/train.py`
- `project/grad-speedup/scripts/run_cifar10.py`

## 8) Commands
Manual spot check:
```bash
cd project
python -u grad-speedup/scripts/run_cifar10.py --run-id step-test --model small-cnn --epochs 1 \
  --max-steps 10 --eval-interval-steps 5 --device cpu --batch-size 128 --download
```

## 9) Deliverables
- Updated code with new CLI flags and behavior.

## 10) Risks
- Ensure max_steps does not skip summary.json creation.

## 11) Reporting
Post summary: code changes + how to use new flags.
