# Task Ticket: Grad-Speedup 72-Condition Grid Generator

## 1) Meta
- ticket_id: task-20260106-grad-speedup-grid-72-generator
- role/agent: implementer-grad-speedup-grid72
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 3h
- workspace_scope: project/grad-speedup/
- related:
  - spec: project/docs/grad-speedup/cifar10-implementation-spec.md
  - combination plan: project/docs/grad-speedup/combination-plan.md

## 2) Goal / Desired Outcome
Generate a full 72-condition config grid (SGD/AdamW × step-control × clip × anderson × sparsity)
so runs can be queued without manual editing.

Success:
- A script outputs 72 config files with deterministic names.
- Each config is valid for run_cifar10.py.

## 3) Background / Context
The base grid is the core of the spec. Existing generator only covers step-control variants.
We need a full grid generator and a predictable naming scheme.

## 4) Scope
In scope:
- Add a new script (or extend generate_configs.py) to emit 72 configs.
- Include EoSS, Backtracking, GGNC (global/layerwise), Anderson on/off, LinBreg on/off.
- Keep ResNet-18 CIFAR defaults.

Out of scope:
- Direction/preconditioning methods (SOAP/GN).

## 5) Requirements
Must:
- 72 configs with unique filenames and run_id fields.
- Base optimizers: SGD+Momentum and AdamW.
- Step-control: none, eoss, adaptive-backtracking.
- Clip: none, ggnc-global, ggnc-layerwise.
- Anderson: none, on.
- Sparsity: none, linbreg.

Should:
- Place outputs under project/grad-speedup/configs/grid-72/.
- Provide a README or index list of generated files.

## 6) Acceptance Criteria
- [ ] Script generates exactly 72 config files.
- [ ] `python scripts/run_cifar10.py --config <file>` succeeds for a sample config.

## 7) Implementation Notes
Suggested approach:
- Add `scripts/generate_grid_72.py` or extend existing generator with a `--grid-72` flag.
- Use deterministic file names like:
  `grid72-sgd-eoss-ggnc-layerwise-anderson-linbreg.json`.
- Include `run_id` in config for stable run naming.

Files to touch:
- project/grad-speedup/scripts/generate_configs.py (or new)
- project/grad-speedup/configs/ (output dir)
- project/grad-speedup/README.md (optional pointer)

## 8) Commands
```bash
cd project/grad-speedup
python scripts/generate_grid_72.py --out-dir configs/grid-72
```

## 9) Deliverables
- Grid generator script.
- 72 config files under configs/grid-72/ (not committed if auto-generated is undesired).

## 10) Risks / Edge Cases
- Filename length; keep concise.
- Config schema drift; ensure keys match run_cifar10.py expectations.

## 11) Open Questions
- Should configs be committed or generated on demand?

## 12) Constraints / Guardrails
- No new dependencies.
