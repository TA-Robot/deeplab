# Task Ticket: Grad-Speedup Diagnostics (Memory + Data Wait)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-diagnostics
- role/agent: implementer-grad-speedup-diagnostics
- owner: PM
- created_at: 2026-01-05
- priority: P2
- timebox: 90 min
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Add optional diagnostics to the grad-speedup runner:
- Peak GPU memory per epoch (if CUDA)
- Data loader wait time (approx) per epoch

## 3) Background / Context
Diagnostics are recommended in the spec to interpret speed gains vs bottlenecks.

## 4) Scope
In scope:
- Track max_memory_allocated during each epoch (CUDA only).
- Estimate data loader wait time by timing batch fetch vs compute.
- Log diagnostics to metrics.jsonl and seed summary.

Out of scope:
- Any ROS-ALTH changes.

## 5) Requirements
Must:
- Work on CPU-only runs (no CUDA dependency for memory metrics).
- Add fields without breaking existing parsing.

## 6) Acceptance Criteria
- [ ] metrics.jsonl includes epoch-level memory and data_wait_time_sec (if enabled).
- [ ] seed summary includes aggregated diagnostics.
- [ ] Flags allow enabling/disabling diagnostics.

## 7) Implementation Notes
Files to touch:
- project/grad-speedup/src/train.py
- project/grad-speedup/scripts/run_cifar10.py

Suggested approach:
- For memory: torch.cuda.reset_peak_memory_stats() at epoch start, read max at epoch end.
- For data wait: time the loader iteration boundary (delta between batch fetch and step start).

## 8) Commands
```
cd project/grad-speedup
python scripts/run_cifar10.py --epochs 1 --device cpu --run-id smoke-diag
```

## 9) Deliverables
- Diagnostics fields added with documentation.

## 10) Risks
- Data wait time is approximate; document limitations.
