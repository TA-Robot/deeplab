# Task Ticket: Grad-Speedup Run Scripts

## 1) Meta
- ticket_id: task-20260105-grad-speedup-runner
- role/agent: implementer-grad-speedup-runner
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 60 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Provide reproducible run scripts for CIFAR-10 baselines and grad-speedup module variants.
Outputs should be routed to project/runs/grad-speedup/.

## 3) Background / Context
We need a consistent way to launch baseline and module combinations for the new track.

## 4) Scope
In scope:
- Add scripts under project/grad-speedup/scripts/
- Include baseline SGD/Adam and at least two module variants

Out of scope:
- Actual long-running execution (handled by PM/Infra when scheduled)

## 5) Requirements
Must:
- Use --output-root runs/grad-speedup
- Include run_id format YYYYMMDD-grad-speedup-cifar10-<variant>
- Set dataset=cifar10 and fixed seeds

## 6) Acceptance Criteria
- [ ] Scripts are checked in and documented in project/grad-speedup/README.md
- [ ] Scripts run without errors for a 1-epoch smoke test

## 7) Implementation Notes
- Use environment variables for device, batch size, and num_workers
- Keep defaults aligned with the CIFAR-10 spec

## 8) Commands
```
cd project/grad-speedup
bash scripts/launch_cifar10_grad_speedup.sh
```

## 9) Deliverables
- Shell script(s) for baselines and module variants

## 10) Risks
- Run IDs may collide; include timestamp or explicit IDs
