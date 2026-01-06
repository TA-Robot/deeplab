# Task Ticket: Dashboard UI Improvements

## 1) Meta
- ticket_id: task-20260106-grad-speedup-dashboard-ui
- role/agent: implementer-dashboard-ui
- owner: PM
- created_at: 2026-01-06
- priority: P1
- timebox: 2–3h
- workspace_scope: project/grad-speedup/

## 2) Goal / Desired Outcome
Make the dashboard easier to interpret by improving labeling, coloring, ranking, and detailed views.

Success looks like:
- Scatter points show enough info to identify runs (hover and optional labels).
- Color/hue options are configurable.
- Bar charts can be sorted by metric.
- Learning curves can be plotted vs time as well as epoch.
- Users can jump from a plot to run detail via selection if supported.

## 3) Background / Context
User feedback:
- Scatter points only labeled by model; hard to know which run is which.
- Too many lines/points; labels truncated.
- Need per-time accuracy curves and time-to-date visualization.
- Bar charts should allow ranking sort.
- Click a plot to open run detail.

## 4) Scope
In scope:
- `project/grad-speedup/dashboard/app.py` UI updates:
  - Add sidebar controls for color by / label by (run_id, model, step_rule, direction, etc.).
  - Add optional label truncation or short labels.
  - Scatter hover data should include run_id, step_rule, direction, seed.
  - Add toggle to aggregate by run_id (mean across seeds) in overview plots.
  - Add sorting toggle + top-K for bar charts.
  - Add "learning curve" in Compare tab with x-axis selectable (epoch vs time).
  - Add "accuracy vs time" in Run Detail if elapsed time columns exist.
  - Add plot selection behavior: if Streamlit supports plotly selection, capture clicked run_id and route to Run Detail view (or preselect run).

Out of scope:
- Data layer changes (handled in separate ticket).
- New dependencies.

## 5) Requirements
Must:
- Keep existing visual theme.
- Do not break if selection events are unsupported (fallback to selectbox).
- Avoid adding heavy dependencies.

## 6) Acceptance Criteria
- [ ] Scatter plots show informative hover tooltips (run_id, seed, step_rule, direction).
- [ ] Bar charts can be sorted by metric (descending) and optionally top-K.
- [ ] Compare tab can plot accuracy vs time when elapsed time column exists.
- [ ] Run Detail supports selecting run from plot click if available; otherwise fallback to manual selectbox.

## 7) Implementation Notes
- Use `hover_data` and `customdata` in Plotly to keep full run_id even if labels are truncated.
- For aggregation, group by run_id and compute mean of numeric metrics; keep label columns as first non-null.
- For selection, check `st.plotly_chart` signature for `on_select` / `selection_mode` (guarded by try/except).

Files:
- project/grad-speedup/dashboard/app.py

## 8) Commands
Manual run:
```bash
cd project/grad-speedup
source .venv/bin/activate
streamlit run dashboard/app.py
```

## 9) Deliverables
- Updated `app.py` with the UI improvements.

## 10) Risks
- Streamlit version may not support plot selection; must gracefully degrade.

## 11) Reporting
Post a short summary: what UI changes were made and how to use new controls.
