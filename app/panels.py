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
    dedupe_findings,
    format_p_value,
    format_number,
    format_magnitude,
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

DEMO_SELECT_KEY = "demo_dataset"
WELCOME_UPLOAD_KEY = "welcome_uploaded_files"

# Encabezados técnicos → español (modo avanzado / tablas)
COLUMN_LABELS_ES = {
    "n_missing": "Nº vacíos",
    "missing_pct": "% vacíos",
    "p_value": "p-valor",
    "p_fdr": "p corregido",
    "pvalue": "p-valor",
    "significant_fdr": "Significativo (corregido)",
    "method": "Método",
    "statistic": "Estadístico",
    "psi": "PSI (cambio)",
    "mean_a": "Media A",
    "mean_b": "Media B",
    "ks_2samp": "Comparación de distribuciones",
    "crude_coef": "Diferencia sin ajustar",
    "interpretation_guardrail": "Aviso de interpretación",
    "coef": "Coeficiente",
    "ate": "Efecto estimado",
    "estimate": "Estimación",
    "r_squared": "R²",
    "ci": "Intervalo de confianza",
    "n": "N",
    "treatment": "Tratamiento",
    "outcome": "Resultado",
    "confounders": "Confusores",
    "adjusted": "Ajuste",
    "ok": "Ok",
    "model": "Modelo",
    "odds_ratio": "Odds ratio",
    "column": "Columna",
    "n_anomalies": "Nº anomalías",
    "pct": "%",
    "group_means": "Medias por grupo",
    "effect_size": "Tamaño del efecto",
    "effect_size_cohens_d": "d de Cohen",
    "effect_size_eta_sq": "Eta²",
    "relative_mean_gap": "Brecha relativa",
    "campo": "Campo",
    "valor": "Valor",
    "grupo": "Grupo",
    "media": "Media",
}


METHOD_LABELS_ES = {
    "ks_2samp": "Comparación de distribuciones",
    "psi": "Índice de estabilidad poblacional (PSI)",
    "chi2": "Chi-cuadrado",
    "t_test": "Prueba t",
    "mannwhitneyu": "Mann–Whitney",
    "kruskal": "Kruskal–Wallis",
    "ols_adjusted": "Regresión ajustada",
    "logit_adjusted": "Regresión logística ajustada",
}


def rename_columns_es(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mapping = {c: COLUMN_LABELS_ES.get(str(c), COLUMN_LABELS_ES.get(str(c).lower(), str(c))) for c in df.columns}
    out = df.rename(columns=mapping)
    # traducir valores de método si existe columna Método/method
    for col in list(out.columns):
        if str(col).lower() in {"método", "method"} or col == "Método":
            out[col] = out[col].map(lambda x: METHOD_LABELS_ES.get(str(x), x))
    return out


def render_welcome(
    on_demo_click_key: str = "want_commercial",
    *,
    demo_select_key: str = DEMO_SELECT_KEY,
) -> None:
    st.markdown("### Subí tu planilla y te contamos qué hay adentro")
    st.markdown(
        "Problemas de datos, patrones y oportunidades — con números calculados por código, "
        "no inventados por una IA."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Subí tu archivo**\n\nAcá abajo o desde la barra izquierda.")
    c2.markdown("**2. Mirá el resumen**\n\nHallazgos en lenguaje simple.")
    c3.markdown("**3. Descargá el informe**\n\nPDF o Markdown para compartir.")
    st.markdown("---")
    st.markdown("**Subí tu CSV o Excel**")
    st.file_uploader(
        "Elegí un archivo de planilla",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key=WELCOME_UPLOAD_KEY,
        help="También acepta CSV de Excel en español (separador ; y decimal ,).",
        label_visibility="collapsed",
    )
    st.caption("Formatos: CSV o Excel (.xlsx). El botón del navegador puede decir “Browse files”.")

    def _load_commercial_demo() -> None:
        # on_click corre antes de instanciar widgets en el próximo run,
        # así podemos setear la key del selectbox sin error.
        st.session_state[demo_select_key] = "commercial"
        st.session_state[on_demo_click_key] = "commercial"

    col_demo, _ = st.columns([1, 1])
    with col_demo:
        st.button(
            "Probar con un ejemplo de ventas",
            type="primary",
            on_click=_load_commercial_demo,
        )
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
    bad = bool(meta.get("suspicious_parse") or (meta.get("n_cols") == 1))
    if bad:
        st.error(
            "La lectura del archivo parece incorrecta (¿1 sola columna?). "
            "Si exportaste desde Excel en español, el separador suele ser `;` "
            "y el decimal `,`. Revisá la vista previa."
        )
    for w in warnings:
        st.warning(w)

    n_rows = len(df)
    if n_rows < 30:
        st.warning(
            f"Tu archivo tiene solo **{n_rows} filas**. "
            "Con tan pocas filas es difícil encontrar patrones (recomendado: 30 o más)."
        )

    if not bad:
        cols = list(map(str, df.columns))
        st.success(
            f"Leímos **{n_rows:,} filas** y **{len(cols)} columnas**: "
            + ", ".join(cols[:12])
            + ("…" if len(cols) > 12 else "")
            + ". ¿Se ve bien?"
        )
    st.dataframe(df.head(8), width="stretch")


def _render_one_card(f: Finding, narratives: dict[str, str]) -> None:
    h = humanize_finding(f)
    icon = CATEGORY_ICON.get(f.category, "•")
    border = {
        "red": "#BC4749",
        "orange": "#E36414",
        "teal": "#0F4C5C",
        "blue": "#3D5A80",
        "gray": "#6c757d",
    }.get(h["color"], "#0F4C5C")
    track = h["track"]
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
        st.write(f"**Magnitud:** {format_magnitude(f.magnitude)}")
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
                label = COLUMN_LABELS_ES.get(str(k), str(k))
                if isinstance(v, float):
                    pretty[label] = format_number(v, digits=4 if abs(v) < 1 else 2)
                elif not isinstance(v, (dict, list)):
                    pretty[label] = v if v is not None else "—"
            if pretty:
                st.dataframe(
                    pd.DataFrame([{"campo": k, "valor": v} for k, v in pretty.items()]),
                    width="stretch",
                    hide_index=True,
                )
        st.write("**Otras explicaciones posibles:**", f.alternative_explanations)
        st.write("**Limitaciones:**", f.limitations)


def render_finding_cards(findings: list[Finding], narratives: dict[str, str]) -> None:
    findings = dedupe_findings(findings)
    discoveries = [f for f in findings if finding_track(f) == "descubrimientos"]
    problems = [f for f in findings if finding_track(f) == "problemas"]

    if not discoveries and not problems:
        st.info(
            "No encontramos hallazgos destacados. "
            "Puede ser que los datos estén muy parejos, o que haya pocas filas "
            "para detectar patrones. Probá Explorar o el modo avanzado."
        )
        return

    if discoveries:
        st.markdown("#### Lo más interesante")
        for f in discoveries[:3]:
            _render_one_card(f, narratives)
        rest = discoveries[3:]
        if rest:
            with st.expander(f"Más descubrimientos ({len(rest)})"):
                for f in rest:
                    _render_one_card(f, narratives)

    if problems:
        with st.expander(f"Problemas en tus datos ({len(problems)})", expanded=not discoveries):
            for f in problems:
                _render_one_card(f, narratives)


def render_plain_summary(findings, profile, artifacts) -> None:
    bullets = build_plain_summary(findings, profile, artifacts)
    st.markdown("#### Resumen en lenguaje simple")
    if not bullets:
        st.info("Todavía no hay un resumen para mostrar.")
        return
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
        st.metric("Calidad del agrupamiento", f"{float(sil):.2f}")
    best = (seg.get("kmeans") or {}).get("best") or {}
    st.write(f"Propuesta de agrupamiento: **k={best.get('k')}**")
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
                f"El agrupamiento por densidad deja ~{100 * float(noise):.0f}% de filas sin grupo: "
                "suele indicar que no hay densidad de grupos clara."
            )
    grid = (seg.get("kmeans") or {}).get("grid") or []
    if grid:
        with st.expander("Ver métricas por k"):
            st.dataframe(rename_columns_es(pd.DataFrame(grid)), width="stretch")
    st.caption(" · ".join(seg.get("limitations") or []))
