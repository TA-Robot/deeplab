# Task Ticket: Review layerwise GN implementation

## 1) Meta

- ticket_id: task-20260106-grad-speedup-layerwise-gn-review
- role/agent: reviewer-layerwise-gn
- owner（manager）: codex-pm
- created_at: 2026-01-06
- priority: P1
- timebox: 60min
- workspace_scope: project/grad-speedup/
- related
  - issue: paper-accurate GN alignment required
  - depends_on: task-20260106-grad-speedup-layerwise-gn

## 2) Goal / Desired Outcome

Review the layerwise GN implementation changes in the implementer worktree and assess conformance to
`arxiv-2510.09378.pdf` (GN paper). Provide Must/Should/Nice findings.

## 3) Scope

### In scope
- Review GN layerwise code changes under `project/grad-speedup/` in the implementer worktree.
- Compare with the GN paper (arxiv-2510.09378) and method-conformance notes.
- Identify mismatches or risks.

### Out of scope
- Writing new code.

## 4) Requirements

### Must
- Identify any deviations from GN / layerwise GN definitions in the paper.
- Call out if implementation is actually a diagonal Fisher / EMA(g^2) proxy.

## 5) Acceptance Criteria

- [ ] Must/Should/Nice review delivered.

## 6) Commands

```bash
cd project
git status -sb
rg -n "gn-layerwise" grad-speedup docs/grad-speedup
```

## 7) Reporting

- Provide Must/Should/Nice with file references.
