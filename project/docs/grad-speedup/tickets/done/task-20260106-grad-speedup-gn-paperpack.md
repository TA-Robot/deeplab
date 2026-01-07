# Task Ticket: Layerwise GN paper pack (triage)

## 1) Meta

- ticket_id: task-20260106-grad-speedup-gn-paperpack
- role/agent: triage-gn-paperpack
- owner（manager）: codex-pm
- created_at: 2026-01-06
- priority: P1
- timebox: 60–90min for initial import
- workspace_scope: project/grad-speedup/
- related
  - issue: layerwise GN implementation requires paper-accurate spec

## 2) Goal / Desired Outcome

We need a paper-accurate source for “full Gauss–Newton” and “layerwise Gauss–Newton” claims referenced in temp materials.
This task brings the correct PDF(s) into the local paper pack and updates docs so implementers can code against exact equations.

### What success looks like
- The correct GN paper PDF(s) exist under `project/docs/grad-speedup/papers/`.
- `project/docs/grad-speedup/papers/README.md` is updated with the GN entry and key equations/pages.
- `project/docs/grad-speedup/method-conformance.md` references the GN paper and the exact update rule we will implement.
- Manager can hand off implementation without ambiguity.

## 3) Background / Context

Layerwise GN was requested as a priority method. Current local paper pack does not include GN/GGN references for this method.
Implementation must be paper-accurate before acceptance.

## 4) Scope

### In scope
- Identify the correct paper(s) for full GN and layerwise GN used in the temp discussion.
- Add PDF(s) to `project/docs/grad-speedup/papers/`.
- Update `papers/README.md` with the key algorithm/equations and page references.
- Update `method-conformance.md` with a short GN section and citations to the local PDFs.

### Out of scope
- Any code changes under `project/grad-speedup/src/`.
- Running experiments.

## 5) Requirements

### Must
- Provide paper-accurate equation references (page/section) for:
  - GN curvature (J^T H_l J or empirical Fisher / GGN).
  - Layerwise approximation (block-diagonal or per-layer factorization).
  - Update rule or preconditioner application.
- Add the PDF(s) locally and update README.

### Should
- Note any compute/overhead caveats explicitly stated by the paper.

## 6) Acceptance Criteria

- [ ] PDFs are saved under `project/docs/grad-speedup/papers/`.
- [ ] `papers/README.md` updated with exact paper IDs and equation references.
- [ ] `method-conformance.md` updated to cite the GN paper and summarize the method.

## 7) Implementation Notes

Suggested approach:
- Use local `project/docs/grad-speedup/temp-materials-summary.md` for the paper clues.
- If the paper is not in the repo, download the PDF and add it to the paper pack.
- Record exact page numbers for the GN / layerwise GN update.

## 8) Commands

```bash
cd project
ls docs/grad-speedup/papers
```

## 9) Deliverables

- PDFs in `project/docs/grad-speedup/papers/`
- Updated `project/docs/grad-speedup/papers/README.md`
- Updated `project/docs/grad-speedup/method-conformance.md`

## 10) Risks / Edge Cases

- Wrong paper selected (ensure claims match temp materials).
- If multiple candidate papers, list options and highlight uncertainty.

## 11) Open Questions

- Which exact GN paper is referenced by the temp discussion? (Confirm.)

## 12) Constraints / Guardrails

- Allowed paths: project/grad-speedup/ only
- Dependency changes: not needed
- Secrets: do not paste tokens

## 13) Reporting

- Provide short progress updates at key steps: paper found → PDF saved → README updated.
