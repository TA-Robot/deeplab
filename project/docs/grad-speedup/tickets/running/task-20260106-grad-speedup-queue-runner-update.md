# Task Ticket: Grad-Speedup Queue Runner Update (Grid Support)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-queue-runner-update
- role/agent: implementer-grad-speedup-queue
- owner: PM
- created_at: 2026-01-06
- priority: P2
- timebox: 2h
- workspace_scope: project/grad-speedup/
- related:
  - queue scripts: project/grad-speedup/scripts/queue_add.sh, queue_run.sh

## 2) Goal / Desired Outcome
Ensure the queue runner can execute a grid of config files sequentially and be appended without manual edits.

Success:
- `queue_add.sh` can add a directory of configs to the queue.
- `queue_run.sh` executes queued configs one by one and logs status.

## 3) Background / Context
We want to avoid manual waiting between runs. The grid sweep should be appendable and sequential.

## 4) Scope
In scope:
- Update queue_add.sh to accept a config directory and append all files.
- Ensure queue_run.sh skips completed runs and logs status to queue.txt.

Out of scope:
- Implementing a scheduler or parallel execution.

## 5) Requirements
Must:
- Config directory input supported (e.g., `queue_add.sh configs/grid-72`).
- Queue runner executes `run_cifar10.py --config <file>` for each entry.
- Status logging to queue.txt (queued/running/done/failed).

## 6) Acceptance Criteria
- [ ] A small sample queue runs to completion.
- [ ] Failed configs are marked and do not block the rest of the queue.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/scripts/queue_add.sh
- project/grad-speedup/scripts/queue_run.sh

## 8) Commands
```bash
cd project/grad-speedup
scripts/queue_add.sh configs/grid-72
scripts/queue_run.sh
```

## 9) Deliverables
- Updated queue scripts with directory support.

## 10) Risks / Edge Cases
- Duplicate entries; ensure idempotent behavior if possible.

## 11) Open Questions
- Should we allow limiting the number of queued configs per run?

## 12) Constraints / Guardrails
- No new dependencies.
