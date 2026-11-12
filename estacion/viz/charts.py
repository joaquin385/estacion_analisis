"""Visualizaciones Plotly interactivas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


TEMPLATE = "plotly_white"
COLORWAY = ["#0F4C5C", "#E36414", "#5C4D7A", "#2A9D8F", "#BC4749", "#6D597A"]


def histogram(df: pd.DataFrame, column: str, color: str | None = None):
    fig = px.histogram(
        df, x=column, color=color, marginal="box", nbins=40,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Distribución de {column}",
    )
    fig.update_layout(bargap=0.05)
    return fig


def boxplot(df: pd.DataFrame, value: str, group: str | None = None):
    fig = px.box(
        df, y=value, x=group, points="outliers",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Boxplot de {value}" + (f" por {group}" if group else ""),
    )
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, trendline: bool = False):
    kwargs = dict(
        data_frame=df, x=x, y=y, color=color, template=TEMPLATE,
        color_discrete_sequence=COLORWAY, opacity=0.7,
        title=f"{y} vs {x}",
    )
    if trendline and color is None:
        kwargs["trendline"] = "ols"
    try:
        fig = px.scatter(**kwargs)
    except Exception:
        kwargs.pop("trendline", None)
        fig = px.scatter(**kwargs)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Matriz de correlación"):
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=TEMPLATE, title=title,
    )
    return fig


def time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, color: str | None = None):
    data = df[[date_col, value_col] + ([color] if color else [])].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col]).sort_values(date_col)
    fig = px.line(
        data, x=date_col, y=value_col, color=color,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Serie temporal: {value_col}",
    )
    return fig


def cluster_scatter(coords: np.ndarray, labels, title: str = "Segmentos (PCA 2D)"):
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": labels.astype(str) if hasattr(labels, "astype") else [str(x) for x in labels],
    })
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="cluster",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=title, opacity=0.75,
    )
    return fig


def anomaly_overview(df: pd.DataFrame, score_col: str, flag_col: str):
    fig = px.histogram(
        df, x=score_col, color=flag_col,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title="Distribución de scores de anomalía",
    )
    return fig


def group_means_bar(means: dict, title: str = "Medias por grupo"):
    plot_df = pd.DataFrame({"grupo": list(means.keys()), "media": list(means.values())})
    fig = px.bar(
        plot_df, x="grupo", y="media",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=title,
    )
    return fig
