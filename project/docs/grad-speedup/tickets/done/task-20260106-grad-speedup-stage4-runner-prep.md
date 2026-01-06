# Task Ticket: Grad-Speedup Stage4 Runner Prep (Clip/Outer Sweep)

## 1) Meta
- ticket_id: task-20260106-grad-speedup-stage4-runner-prep
- role/agent: implementer-grad-speedup
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 120 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Prepare a repeatable runner (script or docs) to execute Stage4 clip/outer sweeps
once the top-1 pair (step-control + direction) is known.

## 3) Requirements
- Must accept inputs: step_rule, direction, run prefix, device, model.
- Must generate 6 runs for clip ∈ {none, ggnc-global, ggnc-layerwise} × outer ∈ {none, anderson}.
- Should default to small-cnn, 1 epoch, 1 seed, GPU.
- Write results under project/runs/grad-speedup.
- Keep code changes minimal (prefer a script under project/grad-speedup/scripts).

## 4) Acceptance Criteria
- [ ] Runner or documented commands exist and are reproducible.
- [ ] No new dependencies.

## 5) Suggested Approach
- Add a small script `scripts/run_stage4_clip_outer.py` or a shell script in `scripts/`.
- Ensure consistent run_id naming: `YYYYMMDD-grad-speedup-stage4-<step>-<dir>-<clip>-<outer>-smallcnn-gpu`.
