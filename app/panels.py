"""Paneles de UI reutilizables (lenguaje simple + avanzado) — Direction A."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from estacion.core.types import Finding, FindingCategory
from estacion.ux.presentation import (
    EMPTY_FINDINGS_MESSAGE,
    CopyContext,
    build_business_kpis,
    build_plain_summary,
    build_summary_headline,
    category_pill_es,
    dedupe_findings,
    format_bullet_list,
    format_p_value,
    format_number,
    format_magnitude,
    format_n_rows,
    finding_track,
    dataframe_for_display,
    group_vs_rest_gap,
    humanize_finding,
    sort_findings_for_humans,
    suggested_next_steps,
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
    for col in list(out.columns):
        if str(col).lower() in {"método", "method"} or col == "Método":
            out[col] = out[col].map(lambda x: METHOD_LABELS_ES.get(str(x), x))
    return out


def render_welcome(
    on_demo_click_key: str = "want_commercial",
    *,
    demo_select_key: str = DEMO_SELECT_KEY,
) -> None:
    st.markdown(
        '<span class="ea-eyebrow">Análisis automático de planillas</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="ea-hero-title">Subí tu planilla y en un minuto te decimos qué está pasando</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="ea-lead">Encontramos qué grupos venden más, qué columnas se mueven juntas '
        "y qué datos están rotos. Todo con números calculados, no inventados.</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ea-dropzone">', unsafe_allow_html=True)
    st.file_uploader(
        "Arrastrá tu archivo acá o elegilo de tu compu",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key=WELCOME_UPLOAD_KEY,
        help="CSV, XLSX · hasta 200 MB. También acepta CSV de Excel en español (separador ; y decimal ,).",
    )
    st.caption("CSV o Excel (.xlsx) · hasta 200 MB · compatible con Excel en español")
    st.markdown("</div>", unsafe_allow_html=True)

    def _load_demo(name: str):
        def _cb() -> None:
            st.session_state[demo_select_key] = name
            st.session_state[on_demo_click_key] = name

        return _cb

    st.markdown("**¿No tenés uno a mano?** Probá con un ejemplo:")
    c1, c2, c3 = st.columns(3)
    c1.button("Ventas", type="primary", on_click=_load_demo("commercial"), width="stretch")
    c2.button("Finanzas", on_click=_load_demo("financial"), width="stretch")
    c3.button("Operaciones", on_click=_load_demo("operational"), width="stretch")

    st.markdown("")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown("**1. Subí tu archivo**")
            st.caption("CSV o Excel. Si lo exportaste desde Excel en español, lo leemos igual.")
    with s2:
        with st.container(border=True):
            st.markdown("**2. Leé los hallazgos**")
            st.caption("Ordenados por importancia, con el número que los respalda.")
    with s3:
        with st.container(border=True):
            st.markdown("**3. Llevate el informe**")
            st.caption("PDF listo para mandar a tu equipo.")

    with st.expander("¿Cómo se ve un hallazgo?"):
        st.markdown(
            """
            **El Norte vende 52 % más por pedido**

            Un pedido promedio del Norte deja más que el resto. Además concentra buena parte
            de la facturación.

            **Por qué importa:** si es replicable, es la mejor pista para crecer.

            **Qué podés hacer:** compará clientes y productos del Norte vs. el resto.
            """
        )


def _is_parse_format_note(msg: str) -> bool:
    m = msg.lower()
    return (
        "separador" in m
        or "decimal" in m
        or "excel en español" in m
        or "coma decimal" in m
    )


def render_load_preview(
    df: pd.DataFrame,
    meta: dict[str, Any] | None,
    *,
    show_table: bool = True,
) -> None:
    meta = meta or {}
    warnings = [str(w) for w in (meta.get("warnings") or []) if w]
    bad = bool(meta.get("suspicious_parse") or (meta.get("n_cols") == 1))
    n_rows = len(df)

    if bad:
        bits = [
            "No pudimos leer este archivo como una planilla usable"
            + (" (parece 1 sola columna)." if meta.get("n_cols") == 1 else ".")
        ]
        bits.append(
            "Si exportaste desde Excel en español, el separador suele ser `;` y el decimal `,`."
        )
        for w in warnings:
            if w not in bits and not _is_parse_format_note(w):
                bits.append(w)
        st.error(" ".join(bits))
    else:
        format_notes = [w for w in warnings if _is_parse_format_note(w)]
        other = [w for w in warnings if not _is_parse_format_note(w)]
        if format_notes:
            st.info(format_notes[0])
        for w in other:
            st.warning(w)
        if n_rows < 30:
            st.warning(
                f"Tu archivo tiene solo **{format_n_rows(n_rows)}**. "
                "Con tan pocas filas es difícil encontrar patrones (recomendado: 30 o más)."
            )
        cols = list(map(str, df.columns))
        st.success(
            f"Leímos **{format_n_rows(n_rows)}** y **{len(cols)} columnas**: "
            + ", ".join(cols[:12])
            + ("…" if len(cols) > 12 else "")
            + ". ¿Se ve bien?"
        )
    if show_table:
        st.dataframe(dataframe_for_display(df.head(8)), width="stretch")


def _badge_color(label: str) -> str:
    if label in {"Muy confiable", "Confiable", "Muy relevante", "Relevante"}:
        return "green"
    if label in {"Revisar", "Atención"}:
        return "orange"
    if label in {"Para mirar"}:
        return "blue"
    return "gray"


def _render_technical_detail(f: Finding, narratives: dict[str, str], h: dict[str, Any]) -> None:
    from estacion.ux.presentation import format_p_clause, method_plain_es, evidence_pairs_plain

    narrative = narratives.get(f.id, "")
    if narrative:
        st.write(narrative)
    if h.get("snippet"):
        st.caption(h["snippet"])
    if f.p_value is not None:
        clause = format_p_clause(f.p_value)
        extra = (
            f" · p corregido = {format_p_value(f.p_value_fdr)}"
            if f.p_value_fdr is not None and not str(format_p_value(f.p_value_fdr)).startswith("<")
            else (
                f" · p corregido {format_p_value(f.p_value_fdr)}"
                if f.p_value_fdr is not None
                else ""
            )
        )
        st.write(clause + extra)
    st.write(f"**Método:** {method_plain_es(f.method)}")
    st.write(f"**Magnitud:** {format_magnitude(f.magnitude, finding=f)}")
    ev = f.evidence or {}
    if isinstance(ev.get("group_means"), dict) and ev["group_means"]:
        gm = {str(k): float(v) for k, v in ev["group_means"].items()}
        st.dataframe(
            pd.DataFrame([{"grupo": k, "media": format_number(v)} for k, v in gm.items()]),
            width="stretch",
            hide_index=True,
        )
        try:
            gap = group_vs_rest_gap(f)
            st.plotly_chart(
                group_means_bar(
                    gm,
                    title="Promedio por grupo",
                    y_title="Promedio",
                    x_title="Grupo",
                    highlight=gap["top_name"] if gap else None,
                ),
                width="stretch",
            )
        except Exception:
            pass
    else:
        pairs = evidence_pairs_plain(ev)
        if pairs:
            st.dataframe(
                pd.DataFrame([{"campo": k, "valor": v} for k, v in pairs]),
                width="stretch",
                hide_index=True,
            )

    alts = format_bullet_list(f.alternative_explanations)
    lims = format_bullet_list(f.limitations)
    if alts:
        st.markdown("**Otras explicaciones posibles:**")
        for a in alts:
            st.markdown(f"- {a}")
    if lims:
        st.markdown("**Limitaciones:**")
        for a in lims:
            st.markdown(f"- {a}")


def _mini_chart_for_finding(f: Finding, ctx: CopyContext | None = None):
    from estacion.ux.presentation import friendly_column, looks_like_money_column

    ev = f.evidence or {}
    means = ev.get("group_means")
    if isinstance(means, dict) and means:
        gap = group_vs_rest_gap(f)
        var = f.variables[0] if f.variables else "valor"
        var_l = friendly_column(str(var), ctx)
        prefix = ""
        if ctx and ctx.money_unit in {"$", "USD", "ARS"} and looks_like_money_column(str(var)):
            prefix = "$"
        return group_means_bar(
            {str(k): float(v) for k, v in means.items()},
            title=f"Promedio de {var_l} por grupo",
            y_title="Promedio",
            x_title="Grupo",
            highlight=gap["top_name"] if gap else None,
            value_prefix=prefix,
            height=210,
        )
    return None


def _render_one_card(
    f: Finding,
    narratives: dict[str, str],
    *,
    ctx: CopyContext | None = None,
    show_chart: bool = True,
) -> None:
    h = humanize_finding(f, ctx)
    with st.container(border=True):
        meta_cols = st.columns([1, 1.2])
        with meta_cols[0]:
            try:
                st.badge(h.get("pill") or category_pill_es(f), color="violet")
            except Exception:
                st.caption(h.get("pill") or "")
        with meta_cols[1]:
            try:
                st.badge(f"● {h['label']}", color=_badge_color(h["label"]))
            except Exception:
                st.caption(h["label"])

        chart = _mini_chart_for_finding(f, ctx) if show_chart else None
        if chart is not None:
            left, right = st.columns([1.35, 1])
            with left:
                st.markdown(f"### {h['title_simple']}")
                st.markdown(f'<p class="ea-lbl">Qué vimos</p>', unsafe_allow_html=True)
                st.markdown(h["what"])
                st.markdown(f'<p class="ea-lbl">Por qué importa</p>', unsafe_allow_html=True)
                st.markdown(h["why"])
                st.markdown(f'<p class="ea-lbl">Qué podés hacer</p>', unsafe_allow_html=True)
                for a in h["do"]:
                    st.markdown(f"- {a}")
            with right:
                st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(f"### {h['title_simple']}")
            st.markdown(f'<p class="ea-lbl">Qué vimos</p>', unsafe_allow_html=True)
            st.markdown(h["what"])
            st.markdown(f'<p class="ea-lbl">Por qué importa</p>', unsafe_allow_html=True)
            st.markdown(h["why"])
            st.markdown(f'<p class="ea-lbl">Qué podés hacer</p>', unsafe_allow_html=True)
            for a in h["do"]:
                st.markdown(f"- {a}")

        with st.expander("Ver detalle técnico"):
            _render_technical_detail(f, narratives, h)


def render_problem_sidebar(
    problems: list[Finding],
    *,
    ctx: CopyContext | None = None,
    all_findings: list[Finding] | None = None,
) -> None:
    st.markdown('<p class="ea-sec">Antes de confiar en los totales</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"**{len(problems)} problema{'s' if len(problems) != 1 else ''} en tus datos**")
        if not problems:
            st.caption("No hay alertas de calidad fuertes.")
        for f in problems[:5]:
            h = humanize_finding(f, ctx)
            st.markdown(f"**!** {h['title_simple']}")
            st.caption(h["what"][:180] + ("…" if len(h["what"]) > 180 else ""))
        if problems:
            with st.expander("Ver cómo arreglarlos"):
                for f in problems:
                    h = humanize_finding(f, ctx)
                    st.markdown(f"**{h['title_simple']}**")
                    for a in h["do"]:
                        st.markdown(f"- {a}")

    steps = suggested_next_steps(all_findings or problems, ctx)
    with st.container(border=True):
        st.markdown("**Próximos pasos sugeridos**")
        for i, s in enumerate(steps, 1):
            st.markdown(f"{i}. {s}")


def render_finding_cards(
    findings: list[Finding],
    narratives: dict[str, str],
    *,
    show_empty: bool = True,
    ctx: CopyContext | None = None,
    layout: str = "stacked",
) -> None:
    findings = dedupe_findings(findings)
    discoveries = [f for f in findings if finding_track(f) == "descubrimientos"]
    problems = [f for f in findings if finding_track(f) == "problemas"]

    if not discoveries and not problems:
        if show_empty:
            st.info(EMPTY_FINDINGS_MESSAGE)
        return

    if layout == "summary":
        main, side = st.columns([1.55, 0.9], gap="large")
        with main:
            st.markdown('<p class="ea-sec">Lo que encontramos</p>', unsafe_allow_html=True)
            for f in discoveries[:3]:
                _render_one_card(f, narratives, ctx=ctx, show_chart=True)
            rest = discoveries[3:]
            if rest:
                with st.expander(f"Más descubrimientos ({len(rest)})"):
                    for f in rest:
                        _render_one_card(f, narratives, ctx=ctx, show_chart=False)
        with side:
            render_problem_sidebar(problems, ctx=ctx, all_findings=findings)
        return

    if discoveries:
        st.markdown("#### Lo más interesante")
        for f in discoveries[:3]:
            _render_one_card(f, narratives, ctx=ctx)
        rest = discoveries[3:]
        if rest:
            with st.expander(f"Más descubrimientos ({len(rest)})"):
                for f in rest:
                    _render_one_card(f, narratives, ctx=ctx, show_chart=False)

    if problems:
        with st.expander(f"Problemas en tus datos ({len(problems)})", expanded=not discoveries):
            for f in problems:
                _render_one_card(f, narratives, ctx=ctx, show_chart=False)


def render_summary_header(
    findings: list[Finding],
    profile,
    artifacts: dict[str, Any],
    df: pd.DataFrame | None = None,
    *,
    ctx: CopyContext | None = None,
) -> None:
    from estacion.ux.presentation import friendly_dataset_name

    headline, sub = build_summary_headline(findings, profile, artifacts, ctx)
    label = (ctx.dataset_label if ctx and ctx.dataset_label else None) or friendly_dataset_name(profile.name)
    st.markdown(f'<p class="ea-sec">{label} · análisis listo</p>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="ea-headline">{headline}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="ea-subhead">{sub}</p>', unsafe_allow_html=True)
    kpis = build_business_kpis(findings, profile, df, ctx)
    cols = st.columns(4)
    for col, kpi in zip(cols, kpis):
        with col:
            # delta_arrow off: el subtítulo no es una variación (+/-)
            st.metric(
                kpi["label"],
                kpi["value"],
                delta=kpi["delta"] or None,
                delta_arrow="off",
                border=True,
            )


def render_plain_summary(findings, profile, artifacts, ctx: CopyContext | None = None) -> None:
    bullets = build_plain_summary(findings, profile, artifacts, ctx=ctx)
    st.markdown("#### Resumen en lenguaje simple")
    if not bullets and not findings:
        return
    if not bullets:
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
        st.metric("Calidad del agrupamiento", f"{float(sil):.2f}", border=True)
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
    lims = format_bullet_list(seg.get("limitations") or [])
    if lims:
        st.caption(" · ".join(lims))
