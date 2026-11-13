"""Visualizaciones Plotly interactivas — paleta Claro SaaS (Direction A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_white"
COLOR_PRIMARY = "#4F46E5"
COLOR_SECONDARY = "#C7D2FE"
COLORWAY = ["#4F46E5", "#818CF8", "#C7D2FE", "#A5B4FC", "#6366F1", "#4338CA"]


def _es_tick(v) -> str:
    """Formato es-AR para ticks/etiquetas (1.500 / 0,45)."""
    try:
        from estacion.ux.presentation import format_int_es, format_number

        fv = float(v)
        if abs(fv - round(fv)) < 1e-9:
            return format_int_es(int(round(fv)))
        return format_number(fv, digits=2)
    except Exception:
        return str(v)


def _set_es_axis_ticks(fig, axis: str = "y") -> None:
    """Fuerza ticktext es-AR (punto de miles) en un eje numérico."""
    try:
        from estacion.ux.presentation import format_int_es

        ax = fig.layout.yaxis if axis == "y" else fig.layout.xaxis
        # Tomar rango de datos
        vals = []
        for tr in fig.data:
            arr = getattr(tr, "y" if axis == "y" else "x", None)
            if arr is None:
                continue
            for v in arr:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return
        # ~5 ticks enteros
        span = hi - lo
        step = max(1.0, round(span / 4 / 100) * 100) if span >= 200 else max(1.0, span / 4)
        ticks = []
        t = 0.0
        while t <= hi * 1.05:
            ticks.append(t)
            t += step
            if len(ticks) > 8:
                break
        if not ticks:
            return
        texts = [format_int_es(int(round(x))) for x in ticks]
        if axis == "y":
            fig.update_yaxes(tickmode="array", tickvals=ticks, ticktext=texts, tickformat=None)
        else:
            fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=texts, tickformat=None)
    except Exception:
        pass


def apply_chart_theme(
    fig,
    *,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    height: int | None = 280,
    es_y_ticks: bool = True,
):
    """Aplica tipografía Inter, márgenes y formato es-AR."""
    fig.update_layout(
        template=TEMPLATE,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#334155"),
        title=dict(
            text=title if title is not None else (fig.layout.title.text if fig.layout.title else ""),
            font=dict(size=14, color="#0F172A", family="Inter, Segoe UI, sans-serif"),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#F8FAFC",
        colorway=COLORWAY,
        margin=dict(l=48, r=16, t=48, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        # decimal=coma, miles=punto (es-AR)
        separators=",.",
    )
    fig.update_xaxes(
        title_text=x_title if x_title is not None else (fig.layout.xaxis.title.text if fig.layout.xaxis.title else None),
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#64748B", size=12, family="Inter, Segoe UI, sans-serif"),
        title_font=dict(color="#64748B", size=12, family="Inter, Segoe UI, sans-serif"),
    )
    fig.update_yaxes(
        title_text=y_title if y_title is not None else (fig.layout.yaxis.title.text if fig.layout.yaxis.title else None),
        gridcolor="#EEF2F7",
        zeroline=False,
        tickfont=dict(color="#64748B", size=12, family="Inter, Segoe UI, sans-serif"),
        title_font=dict(color="#64748B", size=12, family="Inter, Segoe UI, sans-serif"),
    )
    if es_y_ticks:
        _set_es_axis_ticks(fig, "y")
        _set_es_axis_ticks(fig, "x")
    return fig


def histogram(df: pd.DataFrame, column: str, color: str | None = None, *, label: str | None = None):
    lab = label or column
    fig = px.histogram(
        df, x=column, color=color, marginal="box", nbins=40,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Distribución de {lab}",
        labels={column: lab, "count": "Cantidad"},
    )
    fig.update_layout(bargap=0.05)
    return apply_chart_theme(fig, y_title="Cantidad", x_title=lab)


def boxplot(df: pd.DataFrame, value: str, group: str | None = None, *, value_label: str | None = None, group_label: str | None = None):
    vl = value_label or value
    gl = group_label or group
    fig = px.box(
        df, y=value, x=group, points="outliers",
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Distribución de {vl}" + (f" por {gl}" if gl else ""),
        labels={value: vl, group or "": gl or ""},
    )
    return apply_chart_theme(fig, y_title=vl, x_title=gl)


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    trendline: bool = False,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
):
    xl, yl = x_label or x, y_label or y
    kwargs = dict(
        data_frame=df, x=x, y=y, color=color, template=TEMPLATE,
        color_discrete_sequence=COLORWAY, opacity=0.7,
        title=f"{yl} vs {xl}",
        labels={x: xl, y: yl},
    )
    if trendline and color is None:
        kwargs["trendline"] = "ols"
    try:
        fig = px.scatter(**kwargs)
    except Exception:
        kwargs.pop("trendline", None)
        fig = px.scatter(**kwargs)
    return apply_chart_theme(fig, x_title=xl, y_title=yl, height=360)


def correlation_heatmap(corr: pd.DataFrame, title: str = "Mapa de relaciones", labels: dict | None = None):
    mat = corr.copy()
    if labels:
        mat = mat.rename(index=labels, columns=labels)
    fig = px.imshow(
        mat, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#EEF2FF", "#818CF8", "#4F46E5"],
        zmin=-1, zmax=1,
        template=TEMPLATE, title=title,
    )
    # text_auto usa punto; reescribir a coma en hover
    fig.update_traces(texttemplate="%{z:.2f}", hovertemplate="%{x} · %{y}: %{z:.2f}<extra></extra>")
    return apply_chart_theme(fig, height=420, es_y_ticks=False)


def time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, color: str | None = None, *, value_label: str | None = None):
    vl = value_label or value_col
    data = df[[date_col, value_col] + ([color] if color else [])].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col]).sort_values(date_col)
    fig = px.line(
        data, x=date_col, y=value_col, color=color,
        template=TEMPLATE, color_discrete_sequence=COLORWAY,
        title=f"Serie temporal: {vl}",
        labels={date_col: "Fecha", value_col: vl},
    )
    return apply_chart_theme(fig, x_title="Fecha", y_title=vl, height=360)


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
    texts = [f"{value_prefix}{_es_tick(v)}" if value_prefix else _es_tick(v) for v in vals]
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


def relationship_means_line(
    means: dict,
    title: str = "Promedio por nivel",
    *,
    y_title: str = "Promedio",
    x_title: str = "Nivel",
    value_prefix: str = "",
    height: int = 210,
):
    """Línea + área (maqueta A) para relaciones tipo tamaño → ticket."""
    items = sorted(((str(k), float(v)) for k, v in means.items()), key=lambda kv: _sort_key(kv[0]))
    if not items:
        return go.Figure()
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    texts = [f"{value_prefix}{_es_tick(v)}" if value_prefix else _es_tick(v) for v in vals]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=names,
            y=vals,
            mode="lines+markers+text",
            line=dict(color=COLOR_PRIMARY, width=3),
            marker=dict(size=8, color=COLOR_PRIMARY),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.10)",
            text=texts,
            textposition="top center",
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    return apply_chart_theme(fig, title=title, x_title=x_title, y_title=y_title, height=height)


def _sort_key(k: str):
    try:
        return (0, float(k))
    except Exception:
        return (1, str(k))
