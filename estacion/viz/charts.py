"""Visualizaciones Plotly interactivas — paleta Claro SaaS (Direction A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"
# Primario índigo; resto lilas (maqueta A)
COLOR_PRIMARY = "#4F46E5"
COLOR_SECONDARY = "#C7D2FE"
COLORWAY = ["#4F46E5", "#818CF8", "#C7D2FE", "#A5B4FC", "#6366F1", "#4338CA"]


def _es_tick(v) -> str:
    """Formato es-AR para ticks de eje."""
    try:
        from estacion.ux.presentation import format_number

        return format_number(v, digits=0 if abs(float(v)) >= 20 else 2)
    except Exception:
        return str(v)


def apply_chart_theme(
    fig,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = 280,
):
    """Aplica tipografía, márgenes y formato es-AR a un gráfico Plotly."""
    fig.update_layout(
        template=TEMPLATE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
        title=dict(
            text=title or (fig.layout.title.text if fig.layout.title else ""),
            font=dict(size=14, color="#0F172A", family="Inter, Segoe UI, sans-serif"),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        colorway=COLORWAY,
        margin=dict(l=40, r=16, t=48, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(
        title_text=x_title if x_title is not None else fig.layout.xaxis.title.text,
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#64748B", size=12),
        title_font=dict(color="#64748B", size=12),
    )
    fig.update_yaxes(
        title_text=y_title if y_title is not None else fig.layout.yaxis.title.text,
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#64748B", size=12),
        title_font=dict(color="#64748B", size=12),
        tickformat=",",  # miles; luego reescribimos si hace falta
    )
    # Reescribir ticks Y a es-AR (punto de miles)
    try:
        y = fig.data[0].y if fig.data else None
        if y is not None and len(y):
            fig.update_yaxes(tickvals=None)
            fig.update_layout(yaxis=dict(tickformat=".0f"))
            # Separador de miles estilo es: usamos hovertemplate + ticktext si pocos
    except Exception:
        pass
    fig.update_layout(separators=",.")  # decimal comma, thousands dot (Plotly)
    return fig


def histogram(df: pd.DataFrame, column: str, color: str | None = None):
    fig = px.histogram(
        df, x=column, color=color, marginal="box", nbins=40,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Distribución de {column}",
        labels={column: column, "count": "Cantidad"},
    )
    fig.update_layout(bargap=0.05)
    return apply_chart_theme(fig, y_title="Cantidad", x_title=column)


def boxplot(df: pd.DataFrame, value: str, group: str | None = None):
    fig = px.box(
        df, y=value, x=group, points="outliers",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Distribución de {value}" + (f" por {group}" if group else ""),
        labels={value: value, group or "": group or ""},
    )
    return apply_chart_theme(fig, y_title=value, x_title=group)


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, trendline: bool = False):
    kwargs = dict(
        data_frame=df, x=x, y=y, color=color, template=TEMPLATE,
        color_discrete_sequence=COLORWAY, opacity=0.7,
        title=f"{y} vs {x}",
        labels={x: x, y: y},
    )
    if trendline and color is None:
        kwargs["trendline"] = "ols"
    try:
        fig = px.scatter(**kwargs)
    except Exception:
        kwargs.pop("trendline", None)
        fig = px.scatter(**kwargs)
    return apply_chart_theme(fig, x_title=x, y_title=y, height=360)


def correlation_heatmap(corr: pd.DataFrame, title: str = "Matriz de correlación"):
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#EEF2FF", "#818CF8", "#4F46E5"],
        zmin=-1, zmax=1,
        template=TEMPLATE, title=title,
    )
    return apply_chart_theme(fig, height=420)


def time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, color: str | None = None):
    data = df[[date_col, value_col] + ([color] if color else [])].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col]).sort_values(date_col)
    fig = px.line(
        data, x=date_col, y=value_col, color=color,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Serie temporal: {value_col}",
        labels={date_col: "Fecha", value_col: value_col},
    )
    return apply_chart_theme(fig, x_title="Fecha", y_title=value_col, height=360)


def cluster_scatter(
    coords: np.ndarray,
    labels,
    title: str = "Segmentos (PCA 2D)",
    *,
    muted: bool = False,
):
    plot_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": labels.astype(str) if hasattr(labels, "astype") else [str(x) for x in labels],
    })
    if muted:
        fig = px.scatter(
            plot_df, x="PC1", y="PC2",
            template=TEMPLATE,
            title=title + " (sin grupos claros)",
            opacity=0.25,
        )
        fig.update_traces(marker=dict(color="#94A3B8"))
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        return apply_chart_theme(fig, height=360)
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="cluster",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=title, opacity=0.75,
        labels={"cluster": "Grupo"},
    )
    return apply_chart_theme(fig, height=360)


def anomaly_overview(df: pd.DataFrame, score_col: str, flag_col: str):
    fig = px.histogram(
        df, x=score_col, color=flag_col,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title="Distribución de puntajes de anomalía",
        labels={score_col: "Puntaje", flag_col: "Anomalía", "count": "Cantidad"},
    )
    return apply_chart_theme(fig, y_title="Cantidad", x_title="Puntaje")


def group_means_bar(
    means: dict,
    title: str = "Promedio por grupo",
    *,
    y_title: str = "Promedio",
    x_title: str = "Grupo",
    highlight: str | None = None,
    value_prefix: str = "",
    height: int = 220,
):
    """Barras con el grupo protagonista en índigo y el resto en lila claro."""
    items = [(str(k), float(v)) for k, v in means.items()]
    if not items:
        return go.Figure()
    if highlight is None:
        highlight = max(items, key=lambda kv: kv[1])[0]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [COLOR_PRIMARY if k == highlight else COLOR_SECONDARY for k in names]
    texts = []
    for v in vals:
        num = _es_tick(v)
        texts.append(f"{value_prefix}{num}" if value_prefix else num)
    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=vals,
                marker_color=colors,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(showlegend=False, bargap=0.35)
    return apply_chart_theme(fig, title=title, x_title=x_title, y_title=y_title, height=height)
