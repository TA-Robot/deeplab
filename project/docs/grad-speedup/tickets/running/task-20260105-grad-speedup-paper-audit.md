# Task Ticket: Grad-Speedup Paper Audit (All Methods)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-paper-audit
- role/agent: triage-grad-speedup-paper-audit
- owner: PM
- created_at: 2026-01-05
- priority: P0
- timebox: 240 min
- workspace_scope: project/grad-speedup/ (docs only)
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Produce a paper-accurate method inventory with algorithm steps and required hyperparameters,
so implementation work can proceed without ambiguity.

## 3) Background / Context
We must align each implemented method with its primary paper. Existing code includes
heuristics (e.g., L0L1/EoSS) that may not match the literature. This audit gates all
implementation tasks.

Paper pack (local PDFs)
- project/docs/grad-speedup/papers/README.md (index)
- Core PDFs: arxiv-2412.20553, arxiv-2408.13150, arxiv-2506.01913, arxiv-2409.11321,
  arxiv-2305.14342, arxiv-2502.16982, arxiv-2409.14989, arxiv-2410.10800

## 4) Scope
In scope:
- For each method in project/docs/grad-speedup/method-conformance.md:
  - Identify primary paper(s) and venue/year.
  - Extract update rule / pseudocode / equations (section + equation numbers).
  - List hyperparameters + recommended defaults (if provided).
  - Note computational complexity and memory requirements.
- Update method-conformance.md with the extracted details and set status to "designed".

Out of scope:
- Writing or modifying training code.

## 5) Requirements
Must:
- Use primary sources (arXiv or proceedings).
- Avoid adding heuristic formulas that are not in the paper.
- Mark methods as "pending" if full algorithm details cannot be verified.

## 6) Acceptance Criteria
- [ ] method-conformance.md includes per-method algorithm notes and paper references.
- [ ] At least core methods (EoSS, L0L1, GGNC, Adaptive Backtracking, SOAP, Sophia, Muon)
      are updated with exact update rules and parameter definitions.

## 7) Implementation Notes
Primary file:
- project/docs/grad-speedup/method-conformance.md

## 8) Commands
```
cd project/grad-speedup
# no code execution required; documentation only
```

## 9) Deliverables
- Updated method-conformance.md with algorithm-specific notes.

## 10) Risks
- Papers may not specify full update rules; must mark as pending rather than guessing.
