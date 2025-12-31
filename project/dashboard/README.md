# Dashboard Report

This dashboard renders metrics from `project/runs/` in a single-page report.
The UI includes dataset and run-group selectors when multiple datasets are present.

## Build

```bash
cd project
python scripts/build_report.py --runs-dir runs --output dashboard/data/report.json
python -m http.server 8000 --directory dashboard
```

Open `http://localhost:8000` in a browser.

## Data sources

- `runs/<run-id>/config.json`
- `runs/<run-id>/env.json`
- `runs/<run-id>/metrics.jsonl`
- `runs/<run-id>/summary.json`

The report aggregates mean/std across seeds, compares baselines vs OBL variants,
and renders guardrail status, deltas, variance, and learning curves per dataset/run group.
