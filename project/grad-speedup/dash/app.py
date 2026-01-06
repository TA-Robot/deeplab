from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, callback, dcc, html, dash_table

DASH_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASH_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dashboard import data as data_mod  # noqa: E402

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

COLOR_CANDIDATES = [
    "run_id",
    "model",
    "optimizer",
    "step_rule",
    "direction",
    "clip_mode",
    "sparsity",
    "seed",
]

DATA_CACHE = {
    "runs": pd.DataFrame(),
    "epochs": pd.DataFrame(),
    "steps": pd.DataFrame(),
    "loaded_at": None,
    "runs_dir": None,
}


def resolve_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def pick_metric(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def truncate_label(value: str, max_len: Optional[int]) -> str:
    if max_len is None or max_len <= 0:
        return value
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def prepare_color_column(
    df: pd.DataFrame,
    color_by: Optional[str],
    truncate: bool,
    max_len: Optional[int],
) -> tuple[pd.DataFrame, Optional[str]]:
    if color_by is None or color_by not in df.columns:
        return df, None
    if not truncate:
        return df, color_by
    label_col = f"{color_by}_label"
    plot_df = df.copy()
    plot_df[label_col] = plot_df[color_by].astype(str).map(lambda v: truncate_label(v, max_len))
    return plot_df, label_col


def add_line_group(df: pd.DataFrame, run_id_col: Optional[str]) -> tuple[pd.DataFrame, Optional[str]]:
    if run_id_col is None or run_id_col not in df.columns:
        return df, None
    seed_col = resolve_col(df, SEED_CANDIDATES)
    if seed_col is None or seed_col not in df.columns:
        return df, run_id_col
    line_group = f"{run_id_col}_seed"
    plot_df = df.copy()
    plot_df[line_group] = (
        plot_df[run_id_col].astype(str) + "|seed=" + plot_df[seed_col].astype(str)
    )
    return plot_df, line_group


def apply_legend(fig, legend_position: str) -> None:
    legend = dict(title_text="")
    margins = dict(l=40, r=20, t=50, b=40)
    if legend_position == "bottom":
        legend.update(orientation="h", yanchor="top", y=-0.28, xanchor="left", x=0)
        margins["b"] = 90
    elif legend_position == "hide":
        fig.update_layout(showlegend=False)
    else:
        legend.update(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
        margins["r"] = 140
    fig.update_layout(
        legend=legend,
        margin=margins,
        font_family="Space Grotesk, sans-serif",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )


def filter_by_run_ids(df: pd.DataFrame, run_ids: Iterable[str]) -> pd.DataFrame:
    run_col = resolve_col(df, RUN_ID_CANDIDATES)
    if run_col is None or df.empty:
        return df
    run_ids = list(run_ids)
    if not run_ids:
        return df.iloc[0:0]
    return df[df[run_col].astype(str).isin(run_ids)]


def load_data(runs_dir: str, queue_file: Optional[str]) -> str:
    runs_df, epochs_df, steps_df = data_mod.load_all_runs(runs_dir, queue_file=queue_file)
    DATA_CACHE["runs"] = runs_df
    DATA_CACHE["epochs"] = epochs_df
    DATA_CACHE["steps"] = steps_df
    DATA_CACHE["loaded_at"] = datetime.utcnow().isoformat() + "Z"
    DATA_CACHE["runs_dir"] = runs_dir
    return f"Loaded {len(runs_df)} run rows | {len(epochs_df)} epochs | {len(steps_df)} steps"


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        DATA_CACHE.get("runs", pd.DataFrame()),
        DATA_CACHE.get("epochs", pd.DataFrame()),
        DATA_CACHE.get("steps", pd.DataFrame()),
    )


def apply_run_filters(
    runs_df: pd.DataFrame,
    model_values: Optional[Sequence[str]],
    optimizer_values: Optional[Sequence[str]],
    step_rule_values: Optional[Sequence[str]],
    direction_values: Optional[Sequence[str]],
    seed_values: Optional[Sequence[str]],
) -> pd.DataFrame:
    df = runs_df.copy()
    if model_values:
        df = df[df["model"].astype(str).isin(model_values)] if "model" in df.columns else df
    if optimizer_values:
        df = df[df["optimizer"].astype(str).isin(optimizer_values)] if "optimizer" in df.columns else df
    if step_rule_values:
        df = df[df["step_rule"].astype(str).isin(step_rule_values)] if "step_rule" in df.columns else df
    if direction_values:
        df = df[df["direction"].astype(str).isin(direction_values)] if "direction" in df.columns else df
    if seed_values:
        seed_col = resolve_col(df, SEED_CANDIDATES)
        if seed_col:
            df = df[df[seed_col].astype(str).isin(seed_values)]
    return df


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Grad-Speedup Dashboard"

load_data(str(DEFAULT_RUNS_DIR), str(DEFAULT_QUEUE_FILE))

app.layout = html.Div(
    className="app-container",
    children=[
        html.Div(
            className="sidebar",
            children=[
                html.H2("Controls"),
                html.Label("Runs directory"),
                dcc.Input(id="runs-dir", value=str(DEFAULT_RUNS_DIR), type="text"),
                html.Label("Queue file"),
                dcc.Input(id="queue-file", value=str(DEFAULT_QUEUE_FILE), type="text"),
                html.Button("Reload", id="reload-button", n_clicks=0),
                html.Div(id="load-status", className="status-text"),
                dcc.Store(id="data-version", data=0),
                html.Hr(),
                html.H3("Filters"),
                html.Label("Model"),
                dcc.Dropdown(id="filter-model", multi=True),
                html.Label("Optimizer"),
                dcc.Dropdown(id="filter-optimizer", multi=True),
                html.Label("Step rule"),
                dcc.Dropdown(id="filter-step-rule", multi=True),
                html.Label("Direction"),
                dcc.Dropdown(id="filter-direction", multi=True),
                html.Label("Seed"),
                dcc.Dropdown(id="filter-seed", multi=True),
                html.Hr(),
                html.H3("Plot Controls"),
                html.Label("Color by"),
                dcc.Dropdown(id="color-by"),
                html.Label("Legend position"),
                dcc.RadioItems(
                    id="legend-position",
                    options=[
                        {"label": "Bottom", "value": "bottom"},
                        {"label": "Right", "value": "right"},
                        {"label": "Hide", "value": "hide"},
                    ],
                    value="bottom",
                ),
                dcc.Checklist(
                    id="truncate-legend",
                    options=[{"label": "Truncate legend labels", "value": "on"}],
                    value=["on"],
                ),
                html.Label("Legend label max chars"),
                dcc.Slider(id="legend-max-chars", min=4, max=48, step=1, value=20),
                dcc.Checklist(
                    id="show-legend-table",
                    options=[{"label": "Show legend table (full labels)", "value": "on"}],
                    value=["on"],
                ),
            ],
        ),
        html.Div(
            className="content",
            children=[
                html.H1("Grad-Speedup Dashboard (Dash)"),
                html.Div(id="summary-metrics", className="summary-row"),
                dcc.Tabs(
                    id="tabs",
                    value="overview",
                    children=[
                        dcc.Tab(label="Overview", value="overview", children=[html.Div(id="overview-content")]),
                        dcc.Tab(
                            label="Compare",
                            value="compare",
                            children=[
                                html.Label("Runs"),
                                dcc.Dropdown(id="compare-runs", multi=True),
                                html.Label("Learning curve x-axis"),
                                dcc.RadioItems(
                                    id="compare-axis",
                                    options=[{"label": "Epoch", "value": "epoch"}, {"label": "Time", "value": "time"}],
                                    value="epoch",
                                ),
                                html.Div(id="compare-content"),
                            ],
                        ),
                        dcc.Tab(
                            label="Run Detail",
                            value="detail",
                            children=[
                                html.Label("Run"),
                                dcc.Dropdown(id="detail-run"),
                                html.Div(id="detail-content"),
                            ],
                        ),
                        dcc.Tab(label="Diagnostics", value="diagnostics", children=[html.Div(id="diagnostics-content")]),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("data-version", "data"),
    Output("load-status", "children"),
    Input("reload-button", "n_clicks"),
    State("runs-dir", "value"),
    State("queue-file", "value"),
    prevent_initial_call=False,
)
def reload_data(n_clicks: int, runs_dir: str, queue_file: str):
    status = load_data(runs_dir, queue_file)
    return n_clicks, status


@callback(
    Output("filter-model", "options"),
    Output("filter-model", "value"),
    Output("filter-optimizer", "options"),
    Output("filter-optimizer", "value"),
    Output("filter-step-rule", "options"),
    Output("filter-step-rule", "value"),
    Output("filter-direction", "options"),
    Output("filter-direction", "value"),
    Output("filter-seed", "options"),
    Output("filter-seed", "value"),
    Output("color-by", "options"),
    Output("color-by", "value"),
    Input("data-version", "data"),
)
def update_filter_options(_version: int):
    runs_df, _, _ = get_data()
    def options_for(col: str):
        if col not in runs_df.columns:
            return [], []
        values = sorted(runs_df[col].dropna().astype(str).unique().tolist())
        opts = [{"label": value, "value": value} for value in values]
        return opts, values

    model_opts, model_vals = options_for("model")
    opt_opts, opt_vals = options_for("optimizer")
    step_opts, step_vals = options_for("step_rule")
    dir_opts, dir_vals = options_for("direction")

    seed_col = resolve_col(runs_df, SEED_CANDIDATES) or "seed"
    seed_opts, seed_vals = options_for(seed_col) if seed_col in runs_df.columns else ([], [])

    color_opts = []
    for col in COLOR_CANDIDATES:
        if col in runs_df.columns:
            color_opts.append({"label": col.replace("_", " ").title(), "value": col})
    color_value = "run_id" if any(opt["value"] == "run_id" for opt in color_opts) else (color_opts[0]["value"] if color_opts else None)

    return (
        model_opts,
        model_vals,
        opt_opts,
        opt_vals,
        step_opts,
        step_vals,
        dir_opts,
        dir_vals,
        seed_opts,
        seed_vals,
        color_opts,
        color_value,
    )


@callback(
    Output("summary-metrics", "children"),
    Input("data-version", "data"),
    Input("filter-model", "value"),
    Input("filter-optimizer", "value"),
    Input("filter-step-rule", "value"),
    Input("filter-direction", "value"),
    Input("filter-seed", "value"),
)
def update_summary(_version, models, optimizers, step_rules, directions, seeds):
    runs_df, epochs_df, steps_df = get_data()
    filtered = apply_run_filters(runs_df, models, optimizers, step_rules, directions, seeds)
    run_count = len(filtered)
    run_id_col = resolve_col(filtered, RUN_ID_CANDIDATES)
    run_ids = (
        filtered[run_id_col].astype(str).unique().tolist()
        if run_id_col and not filtered.empty
        else []
    )
    steps_count = len(filter_by_run_ids(steps_df, run_ids)) if steps_df is not None else 0
    epochs_count = len(filter_by_run_ids(epochs_df, run_ids)) if epochs_df is not None else 0
    return [
        html.Div([html.Div("Runs", className="metric-label"), html.Div(str(run_count), className="metric-value")], className="metric"),
        html.Div([html.Div("Steps", className="metric-label"), html.Div(str(steps_count), className="metric-value")], className="metric"),
        html.Div([html.Div("Epochs", className="metric-label"), html.Div(str(epochs_count), className="metric-value")], className="metric"),
    ]


def build_overview_tab(
    runs_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    color_by: Optional[str],
    legend_position: str,
    truncate: bool,
    max_len: int,
    show_legend_table: bool,
) -> list:
    if runs_df.empty:
        return [html.Div("No runs available for the current filters.")]
    run_id_col = resolve_col(runs_df, RUN_ID_CANDIDATES)
    speed_metric = pick_metric(
        runs_df,
        ["mean_step_time_ms", "mean_step_time_sec", "step_time_ms_mean", "time_to_target", "cost_to_target"],
    )
    quality_metric = pick_metric(runs_df, ["final_test_acc", "best_test_acc", "test_acc", "val_acc", "accuracy"])
    if speed_metric is None or quality_metric is None or run_id_col is None:
        return [html.Div("Required metrics are missing for overview plots.")]

    plot_df, color_col = prepare_color_column(runs_df, color_by, truncate, max_len)
    hover_cols = [col for col in ["model", "optimizer", "step_rule", "direction", "seed"] if col in plot_df.columns]

    scatter_fig = px.scatter(
        plot_df,
        x=speed_metric,
        y=quality_metric,
        color=color_col,
        hover_name=run_id_col,
        hover_data=hover_cols,
        custom_data=[run_id_col] if run_id_col else None,
        title="Speed vs quality",
    )
    apply_legend(scatter_fig, legend_position)

    bar_fig = px.bar(
        plot_df,
        x=run_id_col,
        y=speed_metric,
        color=color_col if color_col != run_id_col else None,
        title="Speed metric by run",
    )
    apply_legend(bar_fig, legend_position)

    table_cols = [
        col
        for col in [
            run_id_col,
            "model",
            "optimizer",
            "step_rule",
            "direction",
            "seed",
            speed_metric,
            quality_metric,
            "status",
            "progress_pct",
        ]
        if col and col in runs_df.columns
    ]

    table = dash_table.DataTable(
        data=runs_df[table_cols].to_dict("records"),
        columns=[{"name": col, "id": col} for col in table_cols],
        page_size=12,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
    )

    content = [
        html.Div(
            className="grid-2",
            children=[
                dcc.Graph(id="speed-scatter", figure=scatter_fig),
                dcc.Graph(id="speed-bar", figure=bar_fig),
            ],
        ),
    ]
    if show_legend_table and color_by and color_col:
        legend_cols = [color_col]
        if color_col != color_by:
            legend_cols.append(color_by)
        legend_df = plot_df[legend_cols].dropna().drop_duplicates()
        legend_df = legend_df.rename(columns={color_col: "label", color_by: "full"})
        legend_columns = [{"name": "label", "id": "label"}]
        if "full" in legend_df.columns:
            legend_columns.append({"name": "full", "id": "full"})
        content.extend(
            [
                html.H3("Legend (full labels)"),
                dash_table.DataTable(
                    data=legend_df.to_dict("records"),
                    columns=legend_columns,
                    page_size=12,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                ),
            ]
        )

    content.extend(
        [
            html.H3("Run Table"),
            table,
        ]
    )
    return content


def build_compare_tab(
    runs_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    selected_runs: Sequence[str],
    color_by: Optional[str],
    legend_position: str,
    truncate: bool,
    max_len: int,
    axis_mode: str,
    show_legend_table: bool,
) -> list:
    if not selected_runs:
        return [html.Div("Select runs to compare.")]
    run_id_col = resolve_col(runs_df, RUN_ID_CANDIDATES)
    compare_epochs = filter_by_run_ids(epochs_df, selected_runs)
    compare_steps = filter_by_run_ids(steps_df, selected_runs)

    accuracy_metric = pick_metric(compare_epochs, ["test_acc", "val_acc", "accuracy"])
    epoch_x = resolve_col(compare_epochs, ["epoch", "epoch_idx"])
    time_x = resolve_col(compare_epochs, TIME_COL_CANDIDATES)
    step_x = resolve_col(compare_steps, ["step", "global_step", "step_idx", "iteration"])
    if run_id_col is None or accuracy_metric is None or (epoch_x is None and time_x is None):
        return [html.Div("Required metrics are missing for comparison plots.")]

    plot_epochs, color_col = prepare_color_column(compare_epochs, color_by, truncate, max_len)
    plot_steps, color_col_steps = prepare_color_column(compare_steps, color_by, truncate, max_len)
    if color_col_steps and color_col_steps != color_col:
        color_col = color_col_steps

    plot_epochs, line_group = add_line_group(plot_epochs, run_id_col)
    plot_steps, line_group_steps = add_line_group(plot_steps, run_id_col)
    if line_group and (epoch_x in plot_epochs.columns if epoch_x else False):
        plot_epochs = plot_epochs.sort_values(by=[line_group, epoch_x], na_position="last")
    if line_group_steps and (step_x in plot_steps.columns if step_x else False):
        plot_steps = plot_steps.sort_values(by=[line_group_steps, step_x], na_position="last")

    x_axis = time_x if axis_mode == "time" and time_x is not None else epoch_x
    acc_fig = px.line(
        plot_epochs,
        x=x_axis,
        y=accuracy_metric,
        color=color_col,
        line_group=line_group,
        hover_name=run_id_col,
        title=f"Accuracy vs {axis_mode}",
    )
    apply_legend(acc_fig, legend_position)

    loss_metric = pick_metric(plot_steps, ["train_loss", "loss"])
    if step_x is None or loss_metric is None or plot_steps.empty:
        loss_graph = html.Div("Loss data missing for selected runs.")
    else:
        loss_fig = px.line(
            plot_steps,
            x=step_x,
            y=loss_metric,
            color=color_col,
        line_group=line_group_steps,
            hover_name=run_id_col,
            title="Train loss overlay",
        )
        apply_legend(loss_fig, legend_position)
        loss_graph = dcc.Graph(figure=loss_fig)

    content = [html.Div(className="grid-2", children=[dcc.Graph(figure=acc_fig), loss_graph])]
    if show_legend_table and color_by and color_col:
        legend_cols = [color_col]
        if color_col != color_by:
            legend_cols.append(color_by)
        legend_df = plot_epochs[legend_cols].dropna().drop_duplicates()
        legend_df = legend_df.rename(columns={color_col: "label", color_by: "full"})
        legend_columns = [{"name": "label", "id": "label"}]
        if "full" in legend_df.columns:
            legend_columns.append({"name": "full", "id": "full"})
        content.extend(
            [
                html.H3("Legend (full labels)"),
                dash_table.DataTable(
                    data=legend_df.to_dict("records"),
                    columns=legend_columns,
                    page_size=12,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                ),
            ]
        )
    return content


def build_detail_tab(
    runs_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    run_id: Optional[str],
    legend_position: str,
) -> list:
    if not run_id:
        return [html.Div("Select a run to view details.")]
    run_id_col = resolve_col(runs_df, RUN_ID_CANDIDATES)
    if not run_id_col:
        return [html.Div("run_id column missing.")]

    run_row = runs_df[runs_df[run_id_col].astype(str) == str(run_id)]
    if run_row.empty:
        return [html.Div("Selected run not found in filtered data.")]
    detail_epochs = filter_by_run_ids(epochs_df, [run_id])
    detail_steps = filter_by_run_ids(steps_df, [run_id])
    seed_col = resolve_col(detail_epochs, SEED_CANDIDATES) or resolve_col(detail_steps, SEED_CANDIDATES)

    summary_cols = [
        col
        for col in [
            run_id_col,
            "model",
            "optimizer",
            "step_rule",
            "direction",
            "clip_mode",
            "sparsity",
            "batch_size",
            "max_steps",
            "lr",
            "momentum",
            "weight_decay",
            "device",
            "status",
            "progress_steps",
            "progress_pct",
        ]
        if col in run_row.columns
    ]
    summary_table = dash_table.DataTable(
        data=run_row[summary_cols].to_dict("records"),
        columns=[{"name": col, "id": col} for col in summary_cols],
        style_table={"overflowX": "auto"},
    )

    accuracy_metric = pick_metric(detail_epochs, ["test_acc", "val_acc", "accuracy"])
    epoch_x = resolve_col(detail_epochs, ["epoch", "epoch_idx"])
    plot_epochs, line_group_epochs = add_line_group(detail_epochs, run_id_col)
    if epoch_x and accuracy_metric and not plot_epochs.empty:
        acc_fig = px.line(
            plot_epochs,
            x=epoch_x,
            y=accuracy_metric,
            color=seed_col if seed_col in plot_epochs.columns else None,
            line_group=line_group_epochs,
            title="Accuracy vs epoch",
        )
        apply_legend(acc_fig, legend_position)
        acc_graph = dcc.Graph(figure=acc_fig)
    else:
        acc_graph = html.Div("Accuracy data missing for this run.")

    step_x = resolve_col(detail_steps, ["step", "global_step", "step_idx", "iteration"])
    loss_metric = pick_metric(detail_steps, ["train_loss", "loss"])
    plot_steps, line_group_steps = add_line_group(detail_steps, run_id_col)
    if step_x and loss_metric and not plot_steps.empty:
        loss_fig = px.line(
            plot_steps,
            x=step_x,
            y=loss_metric,
            color=seed_col if seed_col in plot_steps.columns else None,
            line_group=line_group_steps,
            title="Train loss",
        )
        apply_legend(loss_fig, legend_position)
        loss_graph = dcc.Graph(figure=loss_fig)
    else:
        loss_graph = html.Div("Loss data missing for this run.")

    return [
        html.H3("Run Summary"),
        summary_table,
        html.Div(className="grid-2", children=[acc_graph, loss_graph]),
    ]


def build_diagnostics_tab(
    runs_df: pd.DataFrame,
    epochs_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    color_by: Optional[str],
    legend_position: str,
    truncate: bool,
    max_len: int,
) -> list:
    if steps_df.empty and epochs_df.empty:
        return [html.Div("No diagnostic metrics available for the current filters.")]
    run_id_col = resolve_col(runs_df, RUN_ID_CANDIDATES)
    plot_steps, color_col = prepare_color_column(steps_df, color_by, truncate, max_len)
    plot_epochs, color_col_epochs = prepare_color_column(epochs_df, color_by, truncate, max_len)
    if color_col_epochs and color_col_epochs != color_col:
        color_col = color_col_epochs

    plot_steps, line_group_steps = add_line_group(plot_steps, run_id_col)
    plot_epochs, line_group_epochs = add_line_group(plot_epochs, run_id_col)

    step_x = resolve_col(plot_steps, ["step", "global_step", "step_idx", "iteration"])
    epoch_x = resolve_col(plot_epochs, ["epoch", "epoch_idx"])

    grad_metric = pick_metric(plot_steps, ["grad_norm", "grad_norm_clip"])
    if step_x and grad_metric and not plot_steps.empty:
        grad_fig = px.line(
            plot_steps,
            x=step_x,
            y=grad_metric,
            color=color_col,
            line_group=line_group_steps,
            title="Grad norm",
        )
        apply_legend(grad_fig, legend_position)
        grad_graph = dcc.Graph(figure=grad_fig)
    else:
        grad_graph = html.Div("Grad norm missing.")

    curv_metric = pick_metric(plot_steps, ["curvature", "hessian_trace"])
    if step_x and curv_metric and not plot_steps.empty:
        curv_fig = px.line(
            plot_steps,
            x=step_x,
            y=curv_metric,
            color=color_col,
            line_group=line_group_steps,
            title="Curvature",
        )
        apply_legend(curv_fig, legend_position)
        curv_graph = dcc.Graph(figure=curv_fig)
    else:
        curv_graph = html.Div("Curvature missing.")

    sparsity_metric = pick_metric(plot_epochs, ["sparsity_fraction", "sparsity"])
    if epoch_x and sparsity_metric and not plot_epochs.empty:
        sparsity_fig = px.line(
            plot_epochs,
            x=epoch_x,
            y=sparsity_metric,
            color=color_col,
            line_group=line_group_epochs,
            title="Sparsity",
        )
        apply_legend(sparsity_fig, legend_position)
        sparsity_graph = dcc.Graph(figure=sparsity_fig)
    else:
        sparsity_graph = html.Div("Sparsity missing.")

    return [html.Div(className="grid-3", children=[grad_graph, curv_graph, sparsity_graph])]


@callback(
    Output("compare-runs", "options"),
    Output("compare-runs", "value"),
    Output("detail-run", "options"),
    Output("detail-run", "value"),
    Input("data-version", "data"),
    Input("filter-model", "value"),
    Input("filter-optimizer", "value"),
    Input("filter-step-rule", "value"),
    Input("filter-direction", "value"),
    Input("filter-seed", "value"),
)
def update_run_dropdowns(_version, models, optimizers, step_rules, directions, seeds):
    runs_df, _, _ = get_data()
    filtered_runs = apply_run_filters(runs_df, models, optimizers, step_rules, directions, seeds)
    run_id_col = resolve_col(filtered_runs, RUN_ID_CANDIDATES)
    run_ids = (
        filtered_runs[run_id_col].astype(str).unique().tolist()
        if run_id_col and not filtered_runs.empty
        else []
    )
    options = [{"label": rid, "value": rid} for rid in run_ids]
    default_compare = run_ids[:3] if len(run_ids) > 3 else run_ids
    default_detail = run_ids[0] if run_ids else None
    return options, default_compare, options, default_detail


@callback(
    Output("overview-content", "children"),
    Input("data-version", "data"),
    Input("filter-model", "value"),
    Input("filter-optimizer", "value"),
    Input("filter-step-rule", "value"),
    Input("filter-direction", "value"),
    Input("filter-seed", "value"),
    Input("color-by", "value"),
    Input("legend-position", "value"),
    Input("truncate-legend", "value"),
    Input("legend-max-chars", "value"),
    Input("show-legend-table", "value"),
)
def update_overview(
    _version,
    models,
    optimizers,
    step_rules,
    directions,
    seeds,
    color_by,
    legend_position,
    truncate_vals,
    max_len,
    legend_table_vals,
):
    runs_df, epochs_df, steps_df = get_data()
    filtered_runs = apply_run_filters(runs_df, models, optimizers, step_rules, directions, seeds)
    run_id_col = resolve_col(filtered_runs, RUN_ID_CANDIDATES)
    run_ids = (
        filtered_runs[run_id_col].astype(str).unique().tolist()
        if run_id_col and not filtered_runs.empty
        else []
    )
    filtered_epochs = filter_by_run_ids(epochs_df, run_ids)
    filtered_steps = filter_by_run_ids(steps_df, run_ids)
    truncate = "on" in (truncate_vals or [])
    show_legend_table = "on" in (legend_table_vals or [])
    legend_position = legend_position or "bottom"
    return build_overview_tab(
        filtered_runs,
        filtered_epochs,
        filtered_steps,
        color_by,
        legend_position,
        truncate,
        max_len,
        show_legend_table,
    )


@callback(
    Output("diagnostics-content", "children"),
    Input("data-version", "data"),
    Input("filter-model", "value"),
    Input("filter-optimizer", "value"),
    Input("filter-step-rule", "value"),
    Input("filter-direction", "value"),
    Input("filter-seed", "value"),
    Input("color-by", "value"),
    Input("legend-position", "value"),
    Input("truncate-legend", "value"),
    Input("legend-max-chars", "value"),
)
def update_diagnostics(
    _version,
    models,
    optimizers,
    step_rules,
    directions,
    seeds,
    color_by,
    legend_position,
    truncate_vals,
    max_len,
):
    runs_df, epochs_df, steps_df = get_data()
    filtered_runs = apply_run_filters(runs_df, models, optimizers, step_rules, directions, seeds)
    run_id_col = resolve_col(filtered_runs, RUN_ID_CANDIDATES)
    run_ids = (
        filtered_runs[run_id_col].astype(str).unique().tolist()
        if run_id_col and not filtered_runs.empty
        else []
    )
    filtered_epochs = filter_by_run_ids(epochs_df, run_ids)
    filtered_steps = filter_by_run_ids(steps_df, run_ids)
    truncate = "on" in (truncate_vals or [])
    legend_position = legend_position or "bottom"
    return build_diagnostics_tab(
        filtered_runs,
        filtered_epochs,
        filtered_steps,
        color_by,
        legend_position,
        truncate,
        max_len,
    )


@callback(
    Output("compare-content", "children"),
    Input("compare-runs", "value"),
    Input("compare-axis", "value"),
    Input("data-version", "data"),
    Input("filter-model", "value"),
    Input("filter-optimizer", "value"),
    Input("filter-step-rule", "value"),
    Input("filter-direction", "value"),
    Input("filter-seed", "value"),
    Input("color-by", "value"),
    Input("legend-position", "value"),
    Input("truncate-legend", "value"),
    Input("legend-max-chars", "value"),
    Input("show-legend-table", "value"),
)
def update_compare(
    selected_runs,
    axis_mode,
    _version,
    models,
    optimizers,
    step_rules,
    directions,
    seeds,
    color_by,
    legend_position,
    truncate_vals,
    max_len,
    legend_table_vals,
):
    runs_df, epochs_df, steps_df = get_data()
    filtered_runs = apply_run_filters(runs_df, models, optimizers, step_rules, directions, seeds)
    run_id_col = resolve_col(filtered_runs, RUN_ID_CANDIDATES)
    filtered_ids = (
        filtered_runs[run_id_col].astype(str).unique().tolist()
        if run_id_col and not filtered_runs.empty
        else []
    )
    selected = [rid for rid in (selected_runs or []) if rid in filtered_ids]
    truncate = "on" in (truncate_vals or [])
    show_legend_table = "on" in (legend_table_vals or [])
    axis_mode = axis_mode or "epoch"
    return build_compare_tab(
        filtered_runs,
        epochs_df,
        steps_df,
        selected,
        color_by,
        legend_position or "bottom",
        truncate,
        max_len,
        axis_mode,
        show_legend_table,
    )


@callback(
    Output("detail-content", "children"),
    Input("detail-run", "value"),
    Input("data-version", "data"),
    State("legend-position", "value"),
)
def update_detail(detail_run, _version, legend_position):
    runs_df, epochs_df, steps_df = get_data()
    return build_detail_tab(runs_df, epochs_df, steps_df, detail_run, legend_position or "bottom")


if __name__ == "__main__":
    app.run(debug=True)
