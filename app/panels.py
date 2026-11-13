"""Paneles de UI reutilizables (lenguaje simple + avanzado)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from estacion.core.types import Finding, FindingCategory
from estacion.ux.presentation import (
    build_plain_summary,
    humanize_finding,
    sort_findings_for_humans,
    format_p_value,
    format_number,
    finding_track,
)
from estacion.viz.charts import group_means_bar


CATEGORY_ICON = {
    FindingCategory.QUALITY: "🔧",
    FindingCategory.DISTRIBUTION: "📉",
    FindingCategory.RELATIONSHIP: "🔗",
    FindingCategory.TEMPORAL: "⏱",
    FindingCategory.SEGMENT: "🧩",
    FindingCategory.ANOMALY: "⚠",
    FindingCategory.CAUSAL: "🎯",
    FindingCategory.STATISTICAL: "📊",
}


def render_welcome(on_demo_click_key: str = "want_commercial") -> None:
    st.markdown("### Subí tu planilla y te contamos qué hay adentro")
    st.markdown(
        "Problemas de datos, patrones y oportunidades — con números calculados por código, "
        "no inventados por una IA."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Subí tu archivo**\n\nCSV o Excel desde la barra izquierda.")
    c2.markdown("**2. Mirá el resumen**\n\nHallazgos en lenguaje simple.")
    c3.markdown("**3. Descargá el informe**\n\nPDF o Markdown para compartir.")
    st.markdown("---")
    if st.button("Probar con un ejemplo de ventas", type="primary"):
        st.session_state[on_demo_click_key] = "commercial"
        st.rerun()
    with st.expander("¿Cómo se ve un hallazgo?"):
        st.markdown(
            """
            **Qué vimos:** la región Norte vende más que el resto (≈ 430 vs ≈ 280).

            **Por qué importa:** puede guiar foco comercial o stock.

            **Detalle técnico** (opcional): prueba entre grupos, p &lt; 0,001…
            """
        )


def render_load_preview(df: pd.DataFrame, meta: dict[str, Any] | None) -> None:
    meta = meta or {}
    warnings = meta.get("warnings") or []
    if meta.get("suspicious_parse") or (meta.get("n_cols") == 1):
        st.error(
            "La lectura del archivo parece incorrecta (¿1 sola columna?). "
            "Si exportaste desde Excel en español, el separador suele ser `;` "
            "y el decimal `,`. Revisá la vista previa."
        )
    for w in warnings:
        st.warning(w)
    cols = list(map(str, df.columns))
    st.success(
        f"Leímos **{len(df):,} filas** y **{len(cols)} columnas**: "
        + ", ".join(cols[:12])
        + ("…" if len(cols) > 12 else "")
        + ". ¿Se ve bien?"
    )
    st.dataframe(df.head(8), width="stretch")


def render_finding_cards(findings: list[Finding], narratives: dict[str, str]) -> None:
    ordered = sort_findings_for_humans(findings)
    track_filter = st.radio(
        "Mostrar",
        ["Todo", "Problemas de datos", "Descubrimientos"],
        horizontal=True,
        key="track_filter",
    )
    for f in ordered:
        track = finding_track(f)
        if track_filter == "Problemas de datos" and track != "problemas":
            continue
        if track_filter == "Descubrimientos" and track != "descubrimientos":
            continue
        h = humanize_finding(f)
        icon = CATEGORY_ICON.get(f.category, "•")
        border = {
            "red": "#BC4749",
            "orange": "#E36414",
            "teal": "#0F4C5C",
            "blue": "#3D5A80",
            "gray": "#6c757d",
        }.get(h["color"], "#0F4C5C")
        st.markdown(
            f'<div class="finding-card" style="border-left-color:{border}">'
            f"<strong>{icon} {h['title_simple']}</strong><br/>"
            f'<span class="muted">{h["label"]} · {track}</span><br/>'
            f"<b>Qué vimos:</b> {h['what']}<br/>"
            f"<b>Por qué importa:</b> {h['why']}<br/>"
            f"<b>Qué podés hacer:</b> {' · '.join(h['do'])}"
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Detalle técnico"):
            st.write(narratives.get(f.id, ""))
            if h["snippet"]:
                st.caption(h["snippet"])
            if f.p_value is not None:
                st.write(
                    f"p = {format_p_value(f.p_value)}"
                    + (
                        f" · p corregido = {format_p_value(f.p_value_fdr)}"
                        if f.p_value_fdr is not None
                        else ""
                    )
                )
            st.write(f"**Método:** {f.method}")
            st.write(f"**Magnitud:** {f.magnitude}")
            ev = f.evidence or {}
            if isinstance(ev.get("group_means"), dict) and ev["group_means"]:
                gm = {str(k): float(v) for k, v in ev["group_means"].items()}
                st.dataframe(
                    pd.DataFrame([{"grupo": k, "media": format_number(v)} for k, v in gm.items()]),
                    width="stretch",
                    hide_index=True,
                )
                try:
                    st.plotly_chart(group_means_bar(gm), width="stretch")
                except Exception:
                    pass
            else:
                pretty = {}
                for k, v in ev.items():
                    if isinstance(v, float):
                        pretty[k] = round(v, 4) if abs(v) < 1000 else round(v, 2)
                    elif not isinstance(v, (dict, list)):
                        pretty[k] = v
                if pretty:
                    st.dataframe(
                        pd.DataFrame([{"campo": k, "valor": v} for k, v in pretty.items()]),
                        width="stretch",
                        hide_index=True,
                    )
            st.write("**Otras explicaciones posibles:**", f.alternative_explanations)
            st.write("**Limitaciones:**", f.limitations)


def render_plain_summary(findings, profile, artifacts) -> None:
    bullets = build_plain_summary(findings, profile, artifacts)
    st.markdown("#### Resumen en lenguaje simple")
    for b in bullets:
        st.markdown(f"- {b}")
    fdr = (artifacts or {}).get("fdr_global") or {}
    if fdr.get("n_tested"):
        st.caption(
            f"Hallazgos confiables tras corregir por muchas pruebas: "
            f"{fdr.get('n_significant', 0)} de {fdr.get('n_tested')}."
        )


def render_segmentation_human(seg: dict[str, Any], df: pd.DataFrame | None = None) -> None:
    if not seg.get("ok"):
        st.warning(seg.get("reason") or "No se pudo segmentar")
        return
    natural = seg.get("natural_groups_likely")
    sil = (seg.get("kmeans") or {}).get("best", {}).get("silhouette")
    if natural:
        st.success(seg.get("caveat"))
    else:
        st.warning(seg.get("caveat"))
    if sil is not None:
        st.metric("Calidad del agrupamiento (Silhouette)", f"{float(sil):.2f}")
    best = (seg.get("kmeans") or {}).get("best") or {}
    st.write(f"Propuesta K-means: **k={best.get('k')}**")
    if df is not None and seg.get("kmeans", {}).get("labels") is not None:
        from estacion.ux.result_views import describe_segments

        phrases = describe_segments(
            df,
            seg["kmeans"].get("labels"),
            seg["kmeans"].get("index"),
            seg.get("numeric_cols") or [],
        )
        for p in phrases:
            st.markdown(f"- {p}")
    db = (seg.get("dbscan") or {}).get("best") or {}
    if db:
        noise = db.get("noise_pct")
        if noise is None and db.get("n_noise") is not None and db.get("n_samples"):
            noise = db["n_noise"] / max(1, db["n_samples"])
        if noise is not None and float(noise) > 0.5:
            st.info(
                f"DBSCAN deja ~{100 * float(noise):.0f}% de filas como 'ruido': "
                "suele indicar que no hay densidad de grupos clara."
            )
    grid = (seg.get("kmeans") or {}).get("grid") or []
    if grid:
        with st.expander("Ver métricas por k"):
            st.dataframe(pd.DataFrame(grid), width="stretch")
    st.caption(" · ".join(seg.get("limitations") or []))
