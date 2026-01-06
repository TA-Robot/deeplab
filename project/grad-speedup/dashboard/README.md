# Grad-Speedup Dashboard (Streamlit)

## Setup

Install dependencies from the dashboard folder:

```
pip install -r dashboard/requirements.txt
```

## Run

From the grad-speedup workspace:

```
cd project/grad-speedup
streamlit run dashboard/app.py
```

## Notes

- The app expects the data layer at `dashboard/data.py` with `load_all_runs`.
- Default runs path points to `project/runs/grad-speedup`. You can override it
  from the sidebar.
- Missing metrics are handled gracefully with placeholder notices.
