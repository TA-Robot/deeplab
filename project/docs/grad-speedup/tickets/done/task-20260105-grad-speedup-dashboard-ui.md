# Task Ticket: Grad-Speedup Dashboard UI (Streamlit)

## 1) Meta
- ticket_id: task-20260105-grad-speedup-dashboard-ui
- role/agent: implementer-grad-speedup-dashboard-ui
- owner: PM
- created_at: 2026-01-05
- priority: P1
- timebox: 240 min
- workspace_scope: project/grad-speedup/
- scope_constraint: project/grad-speedup only (see project/AGENTS.md)

## 2) Goal / Desired Outcome
Build a rich local dashboard for run comparison and diagnostics using Streamlit.

## 3) Background / Context
The dashboard spec is in project/docs/grad-speedup/dashboard-spec.md. Data is
provided by the dashboard data layer (dashboard/data.py).

## 4) Scope
In scope:
- Create Streamlit app (project/grad-speedup/dashboard/app.py).
- Implement tabs: Overview, Run Detail, Compare, Diagnostics, System.
- Provide sidebar filters for run_id/model/optimizer/step_rule/direction/clip/sparsity.
- Use Plotly (preferred) or Altair for interactive plots.
- Add README with run instructions and dependencies.

Out of scope:
- Data ingestion (handled by dashboard-data ticket).

## 5) Requirements
Must:
- Load data via dashboard.data (no duplicated loaders).
- Handle missing metrics gracefully.
- Keep UI responsive with caching.

## 6) Acceptance Criteria
- [ ] Running `streamlit run project/grad-speedup/dashboard/app.py` works.
- [ ] Dashboard renders at least 8 plots spanning speed, quality, stability, efficiency.
- [ ] User can overlay multiple runs for comparison.

## 7) Implementation Notes
Suggested files:
- project/grad-speedup/dashboard/app.py
- project/grad-speedup/dashboard/README.md
- project/grad-speedup/dashboard/requirements.txt (streamlit, pandas, plotly)

## 8) Commands
```
cd project/grad-speedup
streamlit run dashboard/app.py
```

## 9) Deliverables
- Streamlit dashboard + README.

## 10) Risks
- Dependencies not installed; document install steps.
