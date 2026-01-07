from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Grad-Speedup Dashboard", layout="wide")

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

.stApp {
  background: radial-gradient(circle at 20% 20%, #f7f1e3, #e7eef2 55%, #f5f8f6 100%);
}

h1, h2, h3, h4, h5, h6, .stMarkdown, .stTextInput, .stSelectbox, .stMultiSelect, .stSlider {
  font-family: 'Space Grotesk', sans-serif;
}

code, pre, .stCodeBlock, .stDataFrame {
  font-family: 'IBM Plex Mono', monospace;
}

section[data-testid="stSidebar"] > div {
  background: linear-gradient(180deg, #f7f1e3 0%, #eef4f7 100%);
}

div[data-testid="metric-container"] {
  background-color: #fef9f0;
  border: 1px solid #e7d9c5;
  padding: 10px 12px;
  border-radius: 8px;
}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)

COLORWAY = ["#1f6f8b", "#f2a541", "#d94f4f", "#3c8d73", "#2f3e46", "#c2a83e"]
px.defaults.color_discrete_sequence = COLORWAY

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_RUNS_DIR = PROJECT_DIR.parent / "runs" / "grad-speedup"
DEFAULT_QUEUE_FILE = PROJECT_DIR / "queue" / "queue.txt"

RUN_ID_CANDIDATES = ["run_id", "run"]
SEED_CANDIDATES = ["seed", "seed_id"]
TIME_COL_CANDIDATES = [
    "step_elapsed_time_sec",
    "epoch_elapsed_time_sec",
    "elapsed_time_s",
    "elapsed_time_sec",
    "time_s",
    "wall_time_s",
    "wall_time",
    "time_sec",
    "elapsed_time_ms",
    "time_ms",
    "wall_time_ms",
]


@st.cache_data(show_spinner=True)
def load_runs(
    data_root: str, queue_file: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_mod = importlib.import_module("dashboard.data")
    result = data_mod.load_all_runs(data_root, queue_file=queue_file, return_targets=True)
    if isinstance(result, tuple) and len(result) >= 4:
        return result[0], result[1], result[2], result[3]
    if isinstance(result, tuple) and len(result) >= 3:
        targets_df = pd.DataFrame()
        return result[0], result[1], result[2], targets_df
    if isinstance(result, dict):
        runs_df = result.get("runs") or result.get("runs_df")
        epochs_df = result.get("epochs") or result.get("epochs_df")
        steps_df = result.get("steps") or result.get("steps_df")
        targets_df = result.get("targets") or result.get("targets_df")
        if runs_df is not None and epochs_df is not None and steps_df is not None:
            if targets_df is None:
                targets_df = pd.DataFrame()
            return runs_df, epochs_df, steps_df, targets_df
    raise ValueError("dashboard.data.load_all_runs must return runs_df, epochs_df, steps_df, targets_df")


def resolve_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def dedupe_columns(cols: Sequence[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for col in cols:
        if col and col not in seen:
            seen.add(col)
            result.append(col)
    return result


def filter_existing_columns(df: pd.DataFrame, cols: Sequence[Optional[str]]) -> list[str]:
    return [col for col in cols if col and col in df.columns]


def build_option_map(cols: Sequence[str]) -> dict[str, Optional[str]]:
    option_map: dict[str, Optional[str]] = {"None": None}
    for col in cols:
        option_map[col.replace("_", " ").title()] = col
    return option_map


def first_non_null(series: pd.Series) -> Optional[object]:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def aggregate_by_run_id(df: pd.DataFrame, run_id_col: str, label_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty or run_id_col not in df.columns:
        return df
    label_cols = {col for col in label_cols if col and col in df.columns}
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in label_cols and col != run_id_col
    ]
    other_cols = [col for col in df.columns if col not in numeric_cols and col != run_id_col]
    for col in sorted(label_cols):
        if col not in other_cols and col != run_id_col:
            other_cols.append(col)
    agg: dict[str, object] = {col: "mean" for col in numeric_cols}
    for col in other_cols:
        agg[col] = first_non_null
    return df.groupby(run_id_col, as_index=False).agg(agg)


def normalize_targets_df(targets_df: pd.DataFrame) -> pd.DataFrame:
    if targets_df is None or targets_df.empty:
        return pd.DataFrame()
    df = targets_df.copy()
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def merge_target_metrics(
    runs_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    run_id_col: str,
    seed_col: Optional[str],
    target_value: Optional[float],
) -> pd.DataFrame:
    if runs_df is None or runs_df.empty:
        return runs_df
    if targets_df is None or targets_df.empty:
        return runs_df
    if run_id_col not in runs_df.columns or run_id_col not in targets_df.columns:
        return runs_df

    target_df = normalize_targets_df(targets_df)
    if target_value is not None and "target" in target_df.columns:
        target_df = target_df[target_df["target"].notna()]
        target_df = target_df[target_df["target"] == float(target_value)]

    key_cols = [run_id_col]
    if seed_col and seed_col in runs_df.columns and seed_col in target_df.columns:
        key_cols.append(seed_col)

    metric_cols = [
        "target",
        "steps_to_target",
        "time_to_target_sec",
        "cost_to_target_sec",
        "time_to_target",
        "cost_to_target",
        "target_accuracy",
        "target_epoch",
    ]
    metric_cols = [col for col in metric_cols if col in target_df.columns]
    if not metric_cols:
        return runs_df

    target_df = target_df[key_cols + metric_cols].copy()
    if target_df.duplicated(key_cols).any():
        target_df = target_df.groupby(key_cols, as_index=False).mean(numeric_only=True)

    runs_clean = runs_df.drop(columns=[col for col in metric_cols if col in runs_df.columns], errors="ignore")
    return runs_clean.merge(target_df, on=key_cols, how="left")


def choose_default_target(targets_df: pd.DataFrame) -> Optional[float]:
    if targets_df is None or targets_df.empty or "target" not in targets_df.columns:
        return None
    df = normalize_targets_df(targets_df)
    if df.empty:
        return None
    if "time_to_target_sec" in df.columns:
        counts = df.groupby("target")["time_to_target_sec"].apply(lambda s: s.notna().sum())
        if not counts.empty and counts.max() > 0:
            return float(counts.idxmax())
    return float(df["target"].dropna().max())


def reduce_status(values: pd.Series) -> Optional[str]:
    if values is None or values.empty:
        return None
    statuses = values.dropna().astype(str).str.lower().tolist()
    if "running" in statuses:
        return "running"
    if "created" in statuses:
        return "created"
    if "completed" in statuses:
        return "completed"
    return statuses[0] if statuses else None


def build_status_table(df: pd.DataFrame, run_id_col: str) -> pd.DataFrame:
    if df.empty or run_id_col not in df.columns:
        return df
    cols = [run_id_col]
    for col in ("status", "progress_pct", "progress_steps", "last_eval_step", "max_steps"):
        if col in df.columns:
            cols.append(col)
    status_df = df[cols].copy()
    agg: dict[str, object] = {}
    if "status" in status_df.columns:
        agg["status"] = reduce_status
    if "progress_pct" in status_df.columns:
        agg["progress_pct"] = "max"
    if "progress_steps" in status_df.columns:
        agg["progress_steps"] = "max"
    if "last_eval_step" in status_df.columns:
        agg["last_eval_step"] = "max"
    if "max_steps" in status_df.columns:
        agg["max_steps"] = "first"
    if not agg:
        return status_df.drop_duplicates(subset=[run_id_col])
    return status_df.groupby(run_id_col, as_index=False).agg(agg)


def truncate_label(value: str, max_len: Optional[int]) -> str:
    if max_len is None or max_len <= 0:
        return value
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def add_label_column(
    df: pd.DataFrame,
    label_col: Optional[str],
    max_len: Optional[int],
) -> Tuple[pd.DataFrame, Optional[str]]:
    if label_col is None or label_col not in df.columns:
        return df, None
    label_name = f"{label_col}_label"
    labels = df[label_col].astype(str).map(lambda value: truncate_label(value, max_len))
    plot_df = df.copy()
    plot_df[label_name] = labels
    return plot_df, label_name


def supports_plotly_selection() -> bool:
    try:
        params = inspect.signature(st.plotly_chart).parameters
    except (TypeError, ValueError):
        return False
    return "on_select" in params and "selection_mode" in params


def plotly_chart_with_selection(fig: go.Figure, selection_key: Optional[str]) -> Optional[object]:
    if supports_plotly_selection():
        try:
            return st.plotly_chart(
                fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=selection_key,
            )
        except TypeError:
            st.plotly_chart(fig, use_container_width=True)
            return None
    st.plotly_chart(fig, use_container_width=True)
    return None


def extract_selected_run_id(selection_event: Optional[object]) -> Optional[str]:
    if selection_event is None:
        return None
    selection = getattr(selection_event, "selection", None)
    if selection is None and isinstance(selection_event, dict):
        selection = selection_event.get("selection", selection_event)
    if selection is None:
        return None
    if isinstance(selection, dict):
        points = selection.get("points")
    else:
        points = getattr(selection, "points", None)
    if not points:
        return None
    point = points[0]
    if isinstance(point, dict):
        custom = point.get("customdata")
        if custom is not None:
            if isinstance(custom, (list, tuple)) and custom:
                return str(custom[0])
            return str(custom)
        if "x" in point:
            return str(point["x"])
        return None
    custom = getattr(point, "customdata", None)
    if custom is not None:
        if isinstance(custom, (list, tuple)) and custom:
            return str(custom[0])
        return str(custom)
    return None


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    legend_position = st.session_state.get("legend_position", "right")
    legend_cfg: dict[str, object] = {}
    margins = dict(l=40, r=20, t=50, b=35)
    if legend_position == "bottom":
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.25, xanchor="left", x=0)
        margins["b"] = 70
    elif legend_position == "hide":
        legend_cfg = dict()
        fig.update_layout(showlegend=False)
    else:
        legend_cfg = dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    fig.update_layout(
        font_family="Space Grotesk, sans-serif",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        legend=legend_cfg,
        margin=margins,
    )
    return fig


def show_empty(label: str) -> None:
    st.info(f"Missing or empty data for {label}.")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def prepare_bar_df(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sort_by: bool,
    top_k: Optional[int],
    descending: bool,
) -> pd.DataFrame:
    plot_df = df.copy()
    if sort_by and y_col in plot_df.columns:
        plot_df[y_col] = coerce_numeric(plot_df[y_col])
        plot_df = plot_df.sort_values(by=y_col, ascending=not descending, na_position="last")
    if top_k and top_k > 0:
        plot_df = plot_df.head(top_k)
    return plot_df


def build_multiselect(
    df: pd.DataFrame,
    label: str,
    col: Optional[str],
) -> Tuple[pd.DataFrame, Optional[Sequence[str]]]:
    if col is None or df.empty or col not in df.columns:
        return df, None
    values = sorted(df[col].dropna().astype(str).unique().tolist())
    if not values:
        return df, None
    selected = st.sidebar.multiselect(label, values, default=values)
    if selected and len(selected) < len(values):
        df = df[df[col].astype(str).isin(selected)]
    return df, selected


def build_range_filter(
    df: pd.DataFrame,
    label: str,
    col: Optional[str],
) -> Tuple[pd.DataFrame, Optional[Tuple[float, float]]]:
    if col is None or df.empty or col not in df.columns:
        return df, None
    if not pd.api.types.is_numeric_dtype(df[col]):
        return build_multiselect(df, label, col)
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    if min_val == max_val:
        return df, (min_val, max_val)
    selected = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    df = df[df[col].between(selected[0], selected[1])]
    return df, selected


def filter_by_run_ids(df: pd.DataFrame, run_ids: Iterable[str]) -> pd.DataFrame:
    run_col = resolve_col(df, RUN_ID_CANDIDATES)
    if run_col is None or df.empty:
        return df
    run_ids = list(run_ids)
    if not run_ids:
        return df.iloc[0:0]
    return df[df[run_col].astype(str).isin(run_ids)]


def pick_metric(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_line(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    title: str,
    color: Optional[str] = None,
    hover_data: Optional[Sequence[str]] = None,
    hover_name: Optional[str] = None,
    line_group: Optional[str] = None,
) -> None:
    if df.empty or x_col is None or y_col is None:
        show_empty(title)
        return
    if x_col not in df.columns or y_col not in df.columns:
        show_empty(title)
        return
    if color and color not in df.columns:
        color = None
    if hover_name and hover_name not in df.columns:
        hover_name = None
    if hover_data:
        hover_data = [col for col in hover_data if col in df.columns]
        if not hover_data:
            hover_data = None
    if line_group and line_group not in df.columns:
        line_group = None
    plot_df = df
    if line_group and x_col in df.columns:
        plot_df = df.sort_values(by=[line_group, x_col], na_position="last")
    fig = px.line(
        plot_df,
        x=x_col,
        y=y_col,
        color=color,
        title=title,
        hover_data=hover_data,
        hover_name=hover_name,
        line_group=line_group,
    )
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)


def plot_hist(
    df: pd.DataFrame,
    col: Optional[str],
    title: str,
    nbins: int = 50,
) -> None:
    if df.empty or col is None or col not in df.columns:
        show_empty(title)
        return
    fig = px.histogram(df, x=col, nbins=nbins, title=title)
    st.plotly_chart(apply_plot_theme(fig), use_container_width=True)


def plot_scatter(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    title: str,
    color: Optional[str] = None,
    hover_data: Optional[Sequence[str]] = None,
    hover_name: Optional[str] = None,
    text: Optional[str] = None,
    custom_data: Optional[Sequence[str]] = None,
    enable_selection: bool = False,
    selection_key: Optional[str] = None,
) -> Optional[object]:
    if df.empty or x_col is None or y_col is None:
        show_empty(title)
        return None
    if x_col not in df.columns or y_col not in df.columns:
        show_empty(title)
        return None
    if color and color not in df.columns:
        color = None
    if hover_name and hover_name not in df.columns:
        hover_name = None
    if hover_data:
        hover_data = [col for col in hover_data if col in df.columns]
        if not hover_data:
            hover_data = None
    if text and text not in df.columns:
        text = None
    if custom_data:
        custom_data = [col for col in custom_data if col in df.columns]
        if not custom_data:
            custom_data = None
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color,
        title=title,
        hover_data=hover_data,
        hover_name=hover_name,
        text=text,
        custom_data=custom_data,
    )
    if text:
        fig.update_traces(textposition="top center")
    fig = apply_plot_theme(fig)
    if enable_selection:
        return plotly_chart_with_selection(fig, selection_key)
    st.plotly_chart(fig, use_container_width=True)
    return None


def plot_bar(
    df: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    title: str,
    color: Optional[str] = None,
    sort_by: bool = False,
    top_k: Optional[int] = None,
    descending: bool = True,
    custom_data: Optional[Sequence[str]] = None,
    enable_selection: bool = False,
    selection_key: Optional[str] = None,
) -> Optional[object]:
    if df.empty or x_col is None or y_col is None:
        show_empty(title)
        return None
    if x_col not in df.columns or y_col not in df.columns:
        show_empty(title)
        return None
    plot_df = prepare_bar_df(df, x_col, y_col, sort_by, top_k, descending)
    if color and color not in plot_df.columns:
        color = None
    if custom_data:
        custom_data = [col for col in custom_data if col in plot_df.columns]
        if not custom_data:
            custom_data = None
    fig = px.bar(plot_df, x=x_col, y=y_col, color=color, title=title, custom_data=custom_data)
    fig = apply_plot_theme(fig)
    if enable_selection:
        return plotly_chart_with_selection(fig, selection_key)
    st.plotly_chart(fig, use_container_width=True)
    return None


def main() -> None:
    st.title("Grad-Speedup Dashboard")
    st.write("Compare runs across speed, quality, stability, and efficiency.")

    with st.sidebar:
        st.header("Data")
        data_root = st.text_input("Runs directory", str(DEFAULT_RUNS_DIR))
        queue_file = st.text_input("Queue file (optional)", str(DEFAULT_QUEUE_FILE))
        if st.button("Reload data"):
            load_runs.clear()
        st.divider()
        st.subheader("Filters")

    try:
        runs_df, epochs_df, steps_df, targets_df = load_runs(data_root, queue_file)
    except ModuleNotFoundError as exc:
        st.error("Missing dashboard.data module. Ensure the data layer is installed.")
        st.exception(exc)
        st.stop()
    except Exception as exc:
        st.error("Failed to load dashboard data.")
        st.exception(exc)
        st.stop()

    if runs_df is None or runs_df.empty:
        st.warning("No runs found.")
        st.stop()

    run_id_col = resolve_col(runs_df, RUN_ID_CANDIDATES)
    if run_id_col is None:
        st.error("runs_df must include a run_id column.")
        st.stop()

    seed_col = resolve_col(runs_df, SEED_CANDIDATES)

    targets_df = normalize_targets_df(targets_df)
    target_value = None
    if not targets_df.empty and "target" in targets_df.columns:
        target_values = sorted([v for v in targets_df["target"].dropna().unique()])
        if target_values:
            default_target = choose_default_target(targets_df)
            default_index = target_values.index(default_target) if default_target in target_values else 0
            with st.sidebar:
                st.subheader("Targets")
                target_value = st.selectbox(
                    "Target accuracy",
                    target_values,
                    index=default_index,
                    format_func=lambda v: f"{float(v):.2f}",
                )

    runs_df = merge_target_metrics(runs_df, targets_df, run_id_col, seed_col, target_value)
    filters_df = runs_df.copy()

    model_col = resolve_col(filters_df, ["model", "arch"])
    optimizer_col = resolve_col(filters_df, ["optimizer", "optim"])
    step_rule_col = resolve_col(filters_df, ["step_rule", "step_rule_name"])
    direction_col = resolve_col(filters_df, ["direction", "dir"])
    clip_col = resolve_col(filters_df, ["clip_mode", "clip"])
    sparsity_col = resolve_col(filters_df, ["sparsity", "sparsity_fraction"])
    anderson_col = resolve_col(filters_df, ["anderson", "anderson_enabled"])

    filters_df, _ = build_multiselect(filters_df, "Run ID", run_id_col)
    filters_df, _ = build_multiselect(filters_df, "Model", model_col)
    filters_df, _ = build_multiselect(filters_df, "Optimizer", optimizer_col)
    filters_df, _ = build_multiselect(filters_df, "Step rule", step_rule_col)
    filters_df, _ = build_multiselect(filters_df, "Direction", direction_col)
    filters_df, _ = build_multiselect(filters_df, "Clip", clip_col)
    filters_df, _ = build_range_filter(filters_df, "Sparsity", sparsity_col)
    if anderson_col:
        filters_df, _ = build_multiselect(filters_df, "Anderson", anderson_col)

    selected_run_ids = filters_df[run_id_col].astype(str).unique().tolist()

    if target_value is not None:
        time_col = resolve_col(filters_df, ["time_to_target", "time_to_target_sec"])
        if time_col is None or coerce_numeric(filters_df[time_col]).dropna().empty:
            st.warning("Selected target has no time-to-target data yet. Check eval settings or target accuracy.")

    run_seed_col = resolve_col(filters_df, SEED_CANDIDATES)
    label_cols = filter_existing_columns(
        filters_df,
        [
            run_id_col,
            model_col,
            optimizer_col,
            step_rule_col,
            direction_col,
            clip_col,
            sparsity_col,
            anderson_col,
            run_seed_col,
        ],
    )
    plot_option_cols = dedupe_columns(label_cols)

    with st.sidebar:
        st.subheader("Plot Controls")
        option_map = build_option_map(plot_option_cols)
        option_labels = list(option_map.keys())
        option_values = list(option_map.values())
        default_color_col = model_col or run_id_col
        color_index = option_values.index(default_color_col) if default_color_col in option_values else 0
        color_choice = st.selectbox("Color by", option_labels, index=color_index)
        color_by = option_map[color_choice]

        default_label_col = run_id_col
        label_index = option_values.index(default_label_col) if default_label_col in option_values else 0
        label_choice = st.selectbox("Label by", option_labels, index=label_index)
        label_by = option_map[label_choice]

        show_labels = st.toggle("Show point labels", value=False)
        truncate_labels = st.toggle("Truncate labels", value=True)
        if truncate_labels:
            label_max_chars = st.slider("Label max chars", 4, 40, 16)
        else:
            label_max_chars = None
        truncate_legend = st.toggle("Truncate legend labels", value=True)
        legend_position = st.selectbox("Legend position", ["Right", "Bottom", "Hide"], index=1)
        st.session_state["legend_position"] = legend_position.lower()
        aggregate_overview = st.toggle("Aggregate seeds by run_id", value=False)
        sort_bars = st.toggle("Sort bar charts by metric", value=True)
        sort_order = st.radio("Bar order", ["Descending", "Ascending"], horizontal=True)
        sort_desc = sort_order == "Descending"
        max_k = max(1, min(50, len(filters_df)))
        top_k = st.slider("Top-K bars (0=all)", 0, max_k, value=0)

    overview_plot_df = (
        aggregate_by_run_id(filters_df, run_id_col, label_cols) if aggregate_overview else filters_df
    )
    baseline_source_df = (
        aggregate_by_run_id(runs_df, run_id_col, label_cols) if aggregate_overview else runs_df
    )
    color_plot_df = overview_plot_df
    color_by_col = color_by
    if truncate_legend and color_by:
        color_plot_df, color_by_col = add_label_column(color_plot_df, color_by, label_max_chars)
    bar_color = color_by_col if color_by_col and color_by_col != run_id_col else None

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Runs", len(filters_df))
    col_b.metric("Steps", 0 if steps_df is None else len(filter_by_run_ids(steps_df, selected_run_ids)))
    col_c.metric("Epochs", 0 if epochs_df is None else len(filter_by_run_ids(epochs_df, selected_run_ids)))

    tabs = st.tabs(["Overview", "Run Detail", "Compare", "Diagnostics", "System"])

    with tabs[0]:
        st.subheader("Run Status")
        status_df = build_status_table(filters_df, run_id_col)
        if status_df.empty:
            show_empty("run status")
        else:
            st.dataframe(status_df, use_container_width=True)
            if "status" in status_df.columns and "progress_pct" in status_df.columns:
                running_rows = status_df[status_df["status"] == "running"]
                if not running_rows.empty:
                    st.caption("Running jobs")
                    for _, row in running_rows.iterrows():
                        label = str(row.get(run_id_col))
                        pct = row.get("progress_pct")
                        progress = 0.0
                        if pct is not None:
                            try:
                                progress = float(pct)
                            except (TypeError, ValueError):
                                progress = 0.0
                        progress = max(0.0, min(progress, 1.0))
                        detail = ""
                        steps_val = row.get("progress_steps")
                        max_steps_val = row.get("max_steps")
                        if steps_val is not None and max_steps_val is not None:
                            if not pd.isna(steps_val) and not pd.isna(max_steps_val):
                                detail = f" ({int(steps_val)}/{int(max_steps_val)} steps)"
                        st.write(f"{label}{detail}")
                        st.progress(progress)

        st.subheader("Run Table")
        overview_cols = [
            run_id_col,
            model_col,
            optimizer_col,
            step_rule_col,
            direction_col,
            clip_col,
            sparsity_col,
            "target" if "target" in filters_df.columns else None,
            pick_metric(filters_df, ["time_to_target", "time_to_target_sec"]),
            pick_metric(filters_df, ["steps_to_target"]),
            pick_metric(filters_df, ["cost_to_target", "cost_to_target_sec"]),
            pick_metric(filters_df, ["throughput", "steps_per_sec"]),
            pick_metric(filters_df, ["final_test_acc", "best_test_acc", "test_acc", "val_acc", "accuracy"]),
        ]
        overview_cols = [col for col in overview_cols if col]
        overview_df = filters_df[overview_cols].copy()
        st.dataframe(overview_df, use_container_width=True)
        st.download_button(
            "Download run table CSV",
            dataframe_to_csv(overview_df),
            file_name="runs_overview.csv",
            mime="text/csv",
        )

        st.subheader("Speed vs Quality")
        speed_metric = pick_metric(
            color_plot_df,
            ["time_to_target", "time_to_target_sec", "cost_to_target", "cost_to_target_sec"],
        )
        quality_metric = pick_metric(
            color_plot_df, ["final_test_acc", "best_test_acc", "test_acc", "val_acc", "accuracy"]
        )
        scatter_df = color_plot_df
        scatter_label_col = None
        if show_labels and label_by:
            scatter_df, scatter_label_col = add_label_column(scatter_df, label_by, label_max_chars)
        hover_cols = filter_existing_columns(
            scatter_df, [run_id_col, step_rule_col, direction_col, run_seed_col]
        )
        if label_by and label_by not in hover_cols:
            hover_cols.append(label_by)
        scatter_selection = plot_scatter(
            scatter_df,
            speed_metric,
            quality_metric,
            "Speed vs quality",
            color=color_by_col,
            hover_data=hover_cols,
            hover_name=run_id_col,
            text=scatter_label_col if show_labels else None,
            custom_data=[run_id_col],
            enable_selection=True,
            selection_key="overview_speed_quality",
        )

        st.subheader("Speed Summary")
        speed_bar_metric = pick_metric(
            color_plot_df,
            ["time_to_target", "time_to_target_sec", "cost_to_target", "cost_to_target_sec"],
        )
        bar_selection = plot_bar(
            color_plot_df,
            run_id_col,
            speed_bar_metric,
            "Speed metric by run",
            color=bar_color,
            sort_by=sort_bars,
            top_k=top_k,
            descending=sort_desc,
            custom_data=[run_id_col],
            enable_selection=True,
            selection_key="overview_speed_bar",
        )
        selected_run_from_plot = extract_selected_run_id(scatter_selection) or extract_selected_run_id(bar_selection)
        if selected_run_from_plot and selected_run_from_plot in selected_run_ids:
            last_selected = st.session_state.get("last_plot_selected_run_id")
            if last_selected != selected_run_from_plot:
                st.session_state["last_plot_selected_run_id"] = selected_run_from_plot
                st.session_state["run_detail_run_id"] = selected_run_from_plot
        if supports_plotly_selection():
            st.caption("Tip: click a point or bar to preselect the Run Detail tab.")

        st.subheader("Speedup vs Baseline")
        baseline_options = baseline_source_df[run_id_col].astype(str).unique().tolist()
        baseline_default = baseline_options[0] if baseline_options else None
        baseline_id = st.selectbox("Baseline run", baseline_options, index=0) if baseline_default else None
        if baseline_id and speed_bar_metric:
            baseline_value = baseline_source_df.loc[
                baseline_source_df[run_id_col].astype(str) == baseline_id, speed_bar_metric
            ]
            baseline_value = coerce_numeric(baseline_value).dropna()
            if baseline_value.empty:
                show_empty("baseline speed metric")
            else:
                base = baseline_value.mean()
                speedup_df = color_plot_df.copy()
                speedup_df["speedup"] = base / coerce_numeric(speedup_df[speed_bar_metric])
                plot_bar(
                    speedup_df,
                    run_id_col,
                    "speedup",
                    "Speedup vs baseline",
                    color=bar_color,
                    sort_by=sort_bars,
                    top_k=top_k,
                    descending=sort_desc,
                )
        else:
            show_empty("speedup vs baseline")

    with tabs[1]:
        st.subheader("Run Detail")
        run_options = selected_run_ids
        if not run_options:
            show_empty("run selection")
        else:
            run_select_key = "run_detail_run_id"
            if run_select_key in st.session_state and st.session_state[run_select_key] not in run_options:
                st.session_state.pop(run_select_key)
            selected_run = st.selectbox("Run", run_options, key=run_select_key)
            run_steps = filter_by_run_ids(steps_df, [selected_run]) if steps_df is not None else pd.DataFrame()
            run_epochs = filter_by_run_ids(epochs_df, [selected_run]) if epochs_df is not None else pd.DataFrame()

            seed_col = resolve_col(run_steps, SEED_CANDIDATES) or resolve_col(run_epochs, SEED_CANDIDATES)
            if seed_col and seed_col in run_steps.columns:
                seed_options = run_steps[seed_col].dropna().unique().tolist()
            elif seed_col and seed_col in run_epochs.columns:
                seed_options = run_epochs[seed_col].dropna().unique().tolist()
            else:
                seed_options = []

            if seed_options:
                selected_seed = st.selectbox("Seed", seed_options, index=0)
                if seed_col and seed_col in run_steps.columns:
                    run_steps = run_steps[run_steps[seed_col] == selected_seed]
                if seed_col and seed_col in run_epochs.columns:
                    run_epochs = run_epochs[run_epochs[seed_col] == selected_seed]

            x_axis = st.radio("X axis", ["step", "time"], horizontal=True)
            if x_axis == "step":
                step_x = resolve_col(run_steps, ["step", "global_step", "step_idx", "iteration"])
            else:
                step_x = resolve_col(run_steps, TIME_COL_CANDIDATES)
            epoch_x = resolve_col(run_epochs, ["epoch", "epoch_idx"])
            accuracy_metric = pick_metric(run_epochs, ["test_acc", "val_acc", "accuracy"])
            accuracy_time_col = resolve_col(run_epochs, TIME_COL_CANDIDATES)

            col1, col2 = st.columns(2)
            with col1:
                plot_line(
                    run_steps,
                    step_x,
                    pick_metric(run_steps, ["train_loss", "loss"]),
                    "Train loss",
                )
            with col2:
                plot_line(
                    run_epochs,
                    epoch_x,
                    accuracy_metric,
                    "Test accuracy",
                )

            if accuracy_time_col and accuracy_metric:
                plot_line(run_epochs, accuracy_time_col, accuracy_metric, "Accuracy vs time")

            st.subheader("Step Diagnostics")
            diag_cols = st.columns(3)
            with diag_cols[0]:
                plot_line(run_steps, step_x, pick_metric(run_steps, ["step_size", "lr"]), "Step size")
            with diag_cols[1]:
                plot_line(run_steps, step_x, pick_metric(run_steps, ["grad_norm", "grad_norm_clip"]), "Grad norm")
            with diag_cols[2]:
                plot_line(run_steps, step_x, pick_metric(run_steps, ["curvature", "hessian_trace"]), "Curvature")

            st.subheader("Step Time Distribution")
            step_time_col = pick_metric(run_steps, ["step_time_ms", "step_time"])
            plot_hist(run_steps, step_time_col, "Step time (ms)")

            if step_time_col and step_time_col in run_steps.columns and not run_steps.empty:
                step_time_values = coerce_numeric(run_steps[step_time_col]).dropna()
                if not step_time_values.empty:
                    p50 = step_time_values.quantile(0.5)
                    p90 = step_time_values.quantile(0.9)
                    st.caption(f"p50: {p50:.2f} ms | p90: {p90:.2f} ms")

            st.subheader("Export")
            st.download_button(
                "Download steps CSV",
                dataframe_to_csv(run_steps),
                file_name=f"steps_{selected_run}.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download epochs CSV",
                dataframe_to_csv(run_epochs),
                file_name=f"epochs_{selected_run}.csv",
                mime="text/csv",
            )

    with tabs[2]:
        st.subheader("Compare Runs")
        compare_options = selected_run_ids
        default_compare = compare_options[:3] if len(compare_options) > 3 else compare_options
        compare_runs = st.multiselect("Runs", compare_options, default=default_compare)
        compare_steps = filter_by_run_ids(steps_df, compare_runs) if steps_df is not None else pd.DataFrame()
        compare_epochs = filter_by_run_ids(epochs_df, compare_runs) if epochs_df is not None else pd.DataFrame()

        compare_color_col = run_id_col
        if truncate_legend and run_id_col:
            compare_epochs, compare_color_col = add_label_column(compare_epochs, run_id_col, label_max_chars)
            compare_steps, _ = add_label_column(compare_steps, run_id_col, label_max_chars)

        step_x = resolve_col(compare_steps, ["step", "global_step", "step_idx", "iteration"])
        epoch_x = resolve_col(compare_epochs, ["epoch", "epoch_idx"])
        time_x = resolve_col(compare_epochs, TIME_COL_CANDIDATES)
        accuracy_metric = pick_metric(compare_epochs, ["test_acc", "val_acc", "accuracy"])

        curve_axis = st.radio("Learning curve x-axis", ["epoch", "time"], horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            if curve_axis == "time":
                plot_line(
                    compare_epochs,
                    time_x,
                    accuracy_metric,
                    "Accuracy vs time",
                    color=compare_color_col,
                    hover_name=run_id_col,
                    line_group=run_id_col,
                )
            else:
                plot_line(
                    compare_epochs,
                    epoch_x,
                    accuracy_metric,
                    "Accuracy vs epoch",
                    color=compare_color_col,
                    hover_name=run_id_col,
                    line_group=run_id_col,
                )
        with col2:
            plot_line(
                compare_steps,
                step_x,
                pick_metric(compare_steps, ["train_loss", "loss"]),
                "Train loss overlay",
                color=compare_color_col,
                hover_name=run_id_col,
                line_group=run_id_col,
            )

        st.subheader("Speed and Quality Summary")
        compare_summary_cols = [
            run_id_col,
            pick_metric(
                filters_df, ["time_to_target", "time_to_target_sec", "cost_to_target", "cost_to_target_sec"]
            ),
            pick_metric(filters_df, ["final_test_acc", "best_test_acc", "test_acc", "val_acc", "accuracy"]),
            pick_metric(filters_df, ["line_search_accept_rate", "line_search_acceptance_rate"]),
            pick_metric(filters_df, ["effective_flops_ratio", "sparsity_fraction"]),
        ]
        compare_summary_cols = [col for col in compare_summary_cols if col]
        compare_summary = filters_df[filters_df[run_id_col].astype(str).isin(compare_runs)][compare_summary_cols]
        st.dataframe(compare_summary, use_container_width=True)

    with tabs[3]:
        st.subheader("Diagnostics")
        diag_runs = filters_df
        diag_steps = filter_by_run_ids(steps_df, selected_run_ids) if steps_df is not None else pd.DataFrame()
        diag_epochs = filter_by_run_ids(epochs_df, selected_run_ids) if epochs_df is not None else pd.DataFrame()

        col1, col2 = st.columns(2)
        with col1:
            plot_bar(
                diag_runs,
                run_id_col,
                pick_metric(diag_runs, ["line_search_accept_rate", "line_search_acceptance_rate"]),
                "Line search accept rate",
                color=bar_color,
                sort_by=sort_bars,
                top_k=top_k,
                descending=sort_desc,
            )
        with col2:
            plot_bar(
                diag_runs,
                run_id_col,
                pick_metric(diag_runs, ["precond_overhead", "precond_overhead_ms", "precond_overhead_pct"]),
                "Preconditioner overhead",
                color=bar_color,
                sort_by=sort_bars,
                top_k=top_k,
                descending=sort_desc,
            )

        col3, col4 = st.columns(2)
        with col3:
            plot_line(
                diag_epochs,
                resolve_col(diag_epochs, ["epoch", "epoch_idx"]),
                pick_metric(diag_epochs, ["sparsity_fraction", "sparsity"]),
                "Sparsity over epochs",
                color=run_id_col,
                line_group=run_id_col,
            )
        with col4:
            plot_line(
                diag_epochs,
                resolve_col(diag_epochs, ["epoch", "epoch_idx"]),
                pick_metric(diag_epochs, ["effective_flops_ratio", "effective_flops"]),
                "Effective FLOPs ratio",
                color=run_id_col,
                line_group=run_id_col,
            )

        st.subheader("Stability Signals")
        st_cols = st.columns(3)
        with st_cols[0]:
            plot_line(
                diag_steps,
                resolve_col(diag_steps, ["step", "global_step", "step_idx"]),
                pick_metric(diag_steps, ["grad_norm", "grad_norm_clip"]),
                "Grad norm",
                color=run_id_col,
                line_group=run_id_col,
            )
        with st_cols[1]:
            plot_line(
                diag_steps,
                resolve_col(diag_steps, ["step", "global_step", "step_idx"]),
                pick_metric(diag_steps, ["curvature", "hessian_trace"]),
                "Curvature",
                color=run_id_col,
                line_group=run_id_col,
            )
        with st_cols[2]:
            plot_line(
                diag_steps,
                resolve_col(diag_steps, ["step", "global_step", "step_idx"]),
                pick_metric(diag_steps, ["lr", "step_size"]),
                "Learning rate",
                color=run_id_col,
                line_group=run_id_col,
            )

    with tabs[4]:
        st.subheader("System")
        system_cols = [
            run_id_col,
            pick_metric(runs_df, ["device", "gpu", "accelerator"]),
            pick_metric(runs_df, ["torch_version", "pytorch_version"]),
            pick_metric(runs_df, ["cuda_version", "cuda"]),
            pick_metric(runs_df, ["cudnn_version", "cudnn"]),
            pick_metric(runs_df, ["deterministic", "determinism"]),
            pick_metric(runs_df, ["cpu", "cpu_model"]),
        ]
        system_cols = [col for col in system_cols if col]
        if system_cols:
            st.dataframe(runs_df[system_cols], use_container_width=True)
            st.download_button(
                "Download system CSV",
                dataframe_to_csv(runs_df[system_cols]),
                file_name="system.csv",
                mime="text/csv",
            )
        else:
            show_empty("system metadata")


if __name__ == "__main__":
    main()
