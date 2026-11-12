"""
Estación Universal de Análisis de Datos — Interfaz local Streamlit.

Ejecutar:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estacion.core.loader import load_table, load_multiple, suggest_join_keys
from estacion.core.profiler import profile_dataframe
from estacion.core.drift import compare_datasets
from estacion.core.session import SessionStore
from estacion.core.quality_rules import (
    DOMAIN_PRESETS,
    QualityRuleSet,
    load_rules,
    evaluate_quality_rules,
    save_rules_example,
)
from estacion.core.multitable import build_relationship_graph, suggest_analysis_paths
from estacion.discovery.engine import run_discovery
from estacion.stats.descriptive import descriptive_report
from estacion.stats.correlation import correlation_analysis
from estacion.stats.hypothesis import group_comparison
from estacion.stats.regression import linear_regression, logistic_regression
from estacion.clustering.segmenter import run_segmentation
from estacion.causal.explorer import exploratory_causal_screen
from estacion.causal.did import difference_in_differences_multipperiod
from estacion.causal.staggered_did import staggered_att, prepare_treat_time_from_binary
from estacion.causal.aipw_staggered import aipw_staggered_att
from estacion.causal.dowhy_bridge import dowhy_available, estimate_ate_dowhy
from estacion.causal.dag_editor import (
    build_dag,
    validate_dag,
    suggest_edges_from_roles,
    dag_to_plotly,
    ALLOWED_ROLES,
)
from estacion.patterns.temporal import temporal_analysis
from estacion.patterns.forecast import forecast_series
from estacion.patterns.missingness import analyze_missingness
from estacion.plugins import (
    list_plugins,
    run_plugin_checks,
    load_dictionary,
    validate_against_dictionary,
    infer_dictionary_stub,
    save_dictionary,
)
from estacion.causal.dag_interactive import render_interactive_dag, parse_exported_dag_json
from estacion.viz.charts import (
    histogram,
    boxplot,
    scatter,
    correlation_heatmap,
    time_series_plot,
    cluster_scatter,
    group_means_bar,
)
from estacion.reports.generator import generate_markdown_report, export_findings_json
from estacion.reports.pdf_export import generate_pdf_report
from estacion.ai.interpreter import interpret_findings
import json

st.set_page_config(
    page_title="Estación Universal de Análisis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Serif', serif !important; color: #0F4C5C; }
.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #d8eee8 0%, transparent 55%),
    radial-gradient(900px 400px at 100% 0%, #f7e6d4 0%, transparent 50%),
    #f7f4ef;
}
div[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0F4C5C 0%, #163a43 100%);
}
div[data-testid="stSidebar"] * { color: #f3f0ea !important; }
.finding-card {
  background: rgba(255,255,255,0.72);
  border-left: 4px solid #E36414;
  padding: 0.9rem 1rem;
  margin-bottom: 0.75rem;
  border-radius: 0 10px 10px 0;
}
.brand {
  font-family: 'IBM Plex Serif', serif;
  font-size: 1.8rem;
  letter-spacing: 0.02em;
  color: #0F4C5C;
}
.muted { color: #5c6670; font-size: 0.95rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _load_uploaded(files) -> dict[str, pd.DataFrame]:
    tables = {}
    for f in files:
        data = f.read()
        df, meta = load_table(data, name=Path(f.name).stem)
        tables[meta["name"]] = df
    return tables


@st.cache_data(show_spinner=False)
def _cached_discovery(df: pd.DataFrame, name: str, quality_domain: str = "default"):
    return run_discovery(df, name=name, quality_domain=quality_domain)


def main():
    st.markdown('<div class="brand">Estación Universal de Análisis de Datos</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Exploración, estadística y descubrimiento automático — '
        "los números salen de código, no de un LLM.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Datos")
        uploaded = st.file_uploader(
            "CSV o Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True
        )
        demo = st.selectbox(
            "O usar dataset sintético",
            [
                "—",
                "commercial",
                "financial",
                "operational",
                "causal_confounded",
                "nonlinear_null",
                "segments",
            ],
        )
        quality_domain = st.selectbox("Dominio de reglas de calidad", list(DOMAIN_PRESETS.keys()))
        use_ollama = st.checkbox(
            "Narrativas vía Ollama local",
            value=False,
            help="Opcional. Requiere Ollama en localhost. Sin API paga; si falla, usa plantillas.",
        )
        run_btn = st.button("Analizar", type="primary", use_container_width=True)
        st.divider()
        st.caption("Sesiones guardadas")
        store = SessionStore(ROOT / "outputs" / "sessions")
        sessions = store.list_sessions()
        session_labels = ["—"] + [f"{s['name']} ({s['id']})" for s in sessions]
        pick_session = st.selectbox("Cargar sesión", session_labels)

    tables: dict[str, pd.DataFrame] = {}
    if pick_session and pick_session != "—":
        sid = pick_session.rsplit("(", 1)[-1].rstrip(")")
        loaded = store.load(sid)
        if loaded.get("df") is not None:
            tables[loaded["meta"]["dataset_name"]] = loaded["df"]
            from estacion.core.types import Finding, FindingCategory, FindingSeverity

            findings_loaded = []
            for raw in loaded["findings"]:
                findings_loaded.append(
                    Finding(
                        id=raw["id"],
                        title=raw["title"],
                        observation=raw["observation"],
                        evidence=raw.get("evidence", {}),
                        magnitude=raw.get("magnitude", ""),
                        method=raw.get("method", ""),
                        alternative_explanations=raw.get("alternative_explanations", []),
                        follow_up=raw.get("follow_up", []),
                        category=FindingCategory(raw.get("category", "quality")),
                        severity=FindingSeverity(raw.get("severity", "info")),
                        score=float(raw.get("score", 0)),
                        variables=raw.get("variables", []),
                        limitations=raw.get("limitations", []),
                        statistically_significant=raw.get("statistically_significant"),
                        practically_important=raw.get("practically_important"),
                        p_value=raw.get("p_value"),
                        p_value_fdr=raw.get("p_value_fdr"),
                        significant_fdr_global=raw.get("significant_fdr_global"),
                    )
                )
            st.session_state["findings"] = findings_loaded
            st.session_state["profile"] = profile_dataframe(loaded["df"], loaded["meta"]["dataset_name"])
            st.session_state["artifacts"] = loaded.get("artifacts", {})
            st.session_state["active_table"] = loaded["meta"]["dataset_name"]
            st.session_state["quality_domain"] = quality_domain
            st.session_state["narratives"] = interpret_findings(
                findings_loaded, try_ollama=use_ollama
            )
            st.sidebar.success(f"Sesión {sid} cargada")
    elif uploaded:
        tables = _load_uploaded(uploaded)
    elif demo and demo != "—":
        path = ROOT / "data" / "synthetic" / f"{demo}.csv"
        if path.exists():
            df, meta = load_table(path)
            tables[meta["name"]] = df
        else:
            st.warning("Dataset sintético no generado. Ejecutá: python -m data.synthetic.generate_datasets")

    if not tables:
        st.info("Cargá un archivo o elegí un dataset sintético para comenzar.")
        st.markdown(
            """
            **Qué hace esta estación**
            - Perfila tipos, calidad y granularidad
            - Descubre correlaciones, segmentos, anomalías y patrones temporales
            - Corre pruebas estadísticas con tamaños de efecto e IC
            - Explora asociaciones causales **sin afirmar causalidad**
            - Compara datasets (drift), exporta PDF y guarda sesiones
            """
        )
        return

    names = list(tables.keys())
    selected = st.sidebar.selectbox("Tabla activa", names)
    df = tables[selected]

    if len(tables) > 1:
        with st.expander("Grafo de relaciones entre tablas", expanded=True):
            graph = build_relationship_graph(tables)
            st.write(graph.get("note"))
            st.write("**Nodos**", pd.DataFrame([{k: n[k] for k in ("id", "n_rows", "n_cols") if k in n} for n in graph["nodes"]]))
            st.write("**Aristas (joins candidatos)**", pd.DataFrame(graph["edges"]) if graph["edges"] else "—")
            st.write("**Rutas sugeridas**")
            for p in suggest_analysis_paths(graph):
                st.markdown(f"- {p}")

    tabs = st.tabs(
        [
            "Descubrimiento",
            "Perfil / Reglas",
            "Missingness",
            "Explorar",
            "Estadística",
            "Segmentación",
            "Causal exploratorio",
            "Comparar / Drift",
            "Informe / Sesiones",
        ]
    )

    # --- Descubrimiento ---
    with tabs[0]:
        st.subheader("Motor de descubrimiento automático")
        need_run = (
            run_btn
            or "findings" not in st.session_state
            or st.session_state.get("active_table") != selected
            or st.session_state.get("quality_domain") != quality_domain
        )
        if need_run and not (pick_session and pick_session != "—" and not run_btn):
            with st.spinner("Investigando el dataset…"):
                findings, profile, artifacts = _cached_discovery(df, selected, quality_domain)
            st.session_state["findings"] = findings
            st.session_state["profile"] = profile
            st.session_state["artifacts"] = artifacts
            st.session_state["active_table"] = selected
            st.session_state["quality_domain"] = quality_domain
            st.session_state["narratives"] = interpret_findings(
                findings, try_ollama=use_ollama
            )
        else:
            findings = st.session_state["findings"]
            profile = st.session_state["profile"]
            artifacts = st.session_state["artifacts"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", f"{profile.n_rows:,}")
        c2.metric("Columnas", profile.n_cols)
        c3.metric("Hallazgos", len(findings))
        fdr = artifacts.get("fdr_global") or {}
        c4.metric("FDR global sig.", f"{fdr.get('n_significant', '—')}/{fdr.get('n_tested', '—')}")

        sev_filter = st.multiselect(
            "Severidad",
            ["critical", "notable", "interesting", "info"],
            default=["critical", "notable", "interesting"],
        )
        only_fdr = st.checkbox("Solo hallazgos que sobreviven FDR global (si tienen p)", value=False)
        narratives = {n["id"]: n["narrative"] for n in st.session_state.get("narratives", [])}
        for f in findings:
            if f.severity.value not in sev_filter:
                continue
            if only_fdr and f.p_value is not None and f.significant_fdr_global is False:
                continue
            fdr_txt = ""
            if f.p_value_fdr is not None:
                fdr_txt = f" · p_FDR={f.p_value_fdr:.4g}"
            st.markdown(
                f'<div class="finding-card"><strong>{f.title}</strong><br/>'
                f'<span class="muted">{f.category.value} · {f.severity.value} · score {f.score:.0f}{fdr_txt}</span>'
                f"<br/>{f.observation}<br/>"
                f"<em>Magnitud:</em> {f.magnitude} · <em>Método:</em> {f.method}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Evidencia, alternativas y seguimiento"):
                st.write(narratives.get(f.id, ""))
                st.json(f.evidence)
                st.write("**Alternativas:**", f.alternative_explanations)
                st.write("**Seguimiento:**", f.follow_up)
                st.write("**Limitaciones:**", f.limitations)

    # --- Perfil / Reglas ---
    with tabs[1]:
        profile = st.session_state.get("profile") or profile_dataframe(df, selected)
        st.subheader("Perfil del dataset")
        rows = []
        for v in profile.variables:
            rows.append(
                {
                    "variable": v.name,
                    "dtype": v.dtype,
                    "role": v.role.value,
                    "missing_%": round(v.missing_pct * 100, 2),
                    "unique": v.n_unique,
                    "warnings": "; ".join(v.warnings),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write("**Granularidad:**", profile.granularity_notes or "—")
        if profile.numeric_columns:
            st.write("**Descriptivos**")
            st.dataframe(pd.DataFrame(descriptive_report(df, profile.numeric_columns)).T)

        st.subheader("Editor de reglas de calidad")
        base = load_rules(domain=quality_domain)
        c1, c2, c3 = st.columns(3)
        max_miss = c1.slider("max_missing_pct", 0.0, 1.0, float(base.max_missing_pct), 0.01)
        max_dup = c2.slider("max_duplicate_pct", 0.0, 0.5, float(base.max_duplicate_pct), 0.001)
        max_out = c3.slider("max_iqr_outlier_pct", 0.0, 0.5, float(base.max_iqr_outlier_pct), 0.01)
        nonneg = st.multiselect(
            "Columnas no negativas",
            list(df.columns),
            default=[c for c in base.non_negative_columns if c in df.columns],
        )
        ids_u = st.multiselect(
            "IDs que deben ser únicos",
            list(df.columns),
            default=[c for c in base.id_columns_must_be_unique if c in df.columns],
        )
        sentinels = st.text_input(
            "Sentinelas (coma-separados)",
            value=",".join(map(str, base.forbidden_sentinel_values)),
        )
        sent_list: list = []
        for tok in sentinels.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                sent_list.append(int(tok))
            except ValueError:
                try:
                    sent_list.append(float(tok))
                except ValueError:
                    sent_list.append(tok)
        custom_rules = QualityRuleSet(
            name=f"{quality_domain}_custom",
            max_missing_pct=max_miss,
            max_duplicate_pct=max_dup,
            max_iqr_outlier_pct=max_out,
            max_dominant_category_pct=base.max_dominant_category_pct,
            forbidden_sentinel_values=sent_list,
            non_negative_columns=nonneg,
            id_columns_must_be_unique=ids_u,
            date_columns_must_parse=base.date_columns_must_parse,
            custom_notes=base.custom_notes + ["Editado en UI"],
        )
        qres = evaluate_quality_rules(df, custom_rules)
        st.write(f"**Violaciones:** {qres['n_violations']} (críticas: {qres['n_critical']})")
        if qres.get("violations"):
            st.dataframe(pd.DataFrame(qres["violations"]), use_container_width=True)
        if st.button("Guardar reglas custom (YAML)"):
            outp = ROOT / "config" / "quality_rules_custom.yaml"
            outp.write_text(
                __import__("yaml").dump(custom_rules.to_dict(), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            st.success(f"Guardado en {outp}")

        st.subheader("Plugins de dominio")
        plugs = list_plugins()
        plug_name = st.selectbox(
            "Plugin",
            ["—"] + [p["name"] for p in plugs],
            format_func=lambda n: n if n == "—" else next(p["title"] for p in plugs if p["name"] == n),
        )
        if plug_name != "—" and st.button("Ejecutar checks de dominio"):
            st.session_state["plugin_result"] = run_plugin_checks(plug_name, df)
        pr = st.session_state.get("plugin_result")
        if pr and pr.get("ok"):
            st.write(f"**{pr.get('title')}** — hallazgos: {pr.get('n_findings')} (críticos: {pr.get('n_critical')})")
            st.dataframe(pd.DataFrame(pr.get("findings") or []), use_container_width=True)
            st.caption(" · ".join(pr.get("notes") or []))

        st.markdown("##### Diccionario de datos versionado")
        dict_dir = ROOT / "config" / "dictionaries"
        dict_files = sorted(
            [p.name for p in dict_dir.glob("*.yaml")] + [p.name for p in dict_dir.glob("*.yml")]
            + [p.name for p in dict_dir.glob("*.json")]
        ) if dict_dir.exists() else []
        dict_choice = st.selectbox("Diccionario", ["—"] + dict_files, key="dict_file")
        cda, cdb = st.columns(2)
        with cda:
            if dict_choice != "—" and st.button("Validar contra diccionario"):
                dd = load_dictionary(dict_dir / dict_choice)
                st.session_state["dict_val"] = validate_against_dictionary(df, dd)
            if st.button("Inferir stub de diccionario"):
                stub = infer_dictionary_stub(df, name=selected, domain=quality_domain)
                outp = dict_dir / f"{selected}_draft.json"
                save_dictionary(stub, outp)
                st.success(f"Stub guardado en {outp}")
        with cdb:
            up_dict = st.file_uploader("Subir diccionario YAML/JSON", type=["yaml", "yml", "json"], key="dict_up")
            if up_dict is not None and st.button("Validar diccionario subido"):
                tmp = dict_dir / f"_upload_{up_dict.name}"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(up_dict.getvalue())
                dd = load_dictionary(tmp)
                st.session_state["dict_val"] = validate_against_dictionary(df, dd)
        dv = st.session_state.get("dict_val")
        if dv:
            st.write(f"Diccionario `{dv.get('dictionary')}` v{dv.get('version')}: {dv.get('n_findings')} hallazgos")
            st.dataframe(pd.DataFrame(dv.get("findings") or []), use_container_width=True)

    # --- Missingness ---
    with tabs[2]:
        st.subheader("Análisis de missingness")
        if st.button("Analizar faltantes", key="miss_btn"):
            st.session_state["missingness"] = analyze_missingness(df)
        miss = st.session_state.get("missingness") or (st.session_state.get("artifacts") or {}).get("missingness")
        if miss:
            # artifacts store a compact version; full analysis if button pressed
            if "summary" in miss:
                st.dataframe(miss["summary"], use_container_width=True)
                st.write("**Little MCAR**", miss.get("little_mcar"))
                st.write("**Pantalla MAR**", {k: v for k, v in miss.get("mar_screen", {}).items() if k != "tests"})
                tests = (miss.get("mar_screen") or {}).get("tests") or []
                if tests:
                    st.dataframe(pd.DataFrame(tests).head(30), use_container_width=True)
                pw = miss.get("pairwise_both_missing")
                if isinstance(pw, pd.DataFrame) and not pw.empty:
                    st.plotly_chart(correlation_heatmap(pw, "Ambos faltantes (proporción)"), use_container_width=True)
            else:
                st.json(miss)
                st.info("Pulsá 'Analizar faltantes' para el análisis completo con tablas.")
        else:
            st.info("Ejecutá el descubrimiento o pulsá el botón de análisis.")

    # --- Explorar ---
    with tabs[3]:
        st.subheader("Exploración interactiva")
        profile = st.session_state.get("profile") or profile_dataframe(df, selected)
        cols = list(df.columns)
        left, right = st.columns(2)
        with left:
            col = st.selectbox("Histograma", cols, key="hist_col")
            color = st.selectbox("Color (opcional)", ["—"] + cols, key="hist_color")
            st.plotly_chart(
                histogram(df, col, None if color == "—" else color),
                use_container_width=True,
            )
        with right:
            x = st.selectbox("Scatter X", profile.numeric_columns or cols, key="sx")
            y = st.selectbox(
                "Scatter Y",
                [c for c in (profile.numeric_columns or cols) if c != x] or cols,
                key="sy",
            )
            st.plotly_chart(scatter(df, x, y, trendline=True), use_container_width=True)

        if len(profile.numeric_columns) >= 2:
            corr = correlation_analysis(df, profile.numeric_columns[:20])
            st.plotly_chart(
                correlation_heatmap(corr["spearman_matrix"], "Spearman"),
                use_container_width=True,
            )

        if profile.datetime_columns and profile.numeric_columns:
            dcol = st.selectbox("Fecha", profile.datetime_columns)
            vcol = st.selectbox("Valor temporal", profile.numeric_columns, key="ts_val")
            st.plotly_chart(time_series_plot(df, dcol, vcol), use_container_width=True)
            tres = temporal_analysis(df, dcol, vcol)
            st.json({k: tres[k] for k in tres if k not in {"limitations", "series_tail"}})
            st.markdown("##### Pronóstico exploratorio (grid AIC + conformal)")
            horizon = st.slider("Horizonte", 4, 36, 12)
            fmethod = st.selectbox("Método", ["aic_grid", "auto", "ets", "arima"])
            use_conf = st.checkbox("Intervalos conformal", value=True)
            if st.button("Pronosticar"):
                st.session_state["forecast"] = forecast_series(
                    df, dcol, vcol, horizon=horizon, method=fmethod, conformal=use_conf
                )
            fc = st.session_state.get("forecast")
            if fc:
                if fc.get("ok"):
                    st.write(f"Método: `{fc['method']}` · backtest: {fc.get('backtest')}")
                    if fc.get("aic_selection"):
                        st.write("**Selección AIC (top)**")
                        st.dataframe(pd.DataFrame(fc["aic_selection"]), use_container_width=True)
                    if fc.get("conformal_multihorizon"):
                        st.write("**Conformal multi-horizon**", fc["conformal_multihorizon"])
                    elif fc.get("conformal"):
                        st.write("**Conformal 1-step**", fc["conformal"])
                    st.dataframe(pd.DataFrame(fc["forecast"]), use_container_width=True)
                    st.caption("; ".join(fc.get("limitations") or []))
                else:
                    st.warning(fc.get("reason"))

        with st.expander("Filtros"):
            filt_col = st.selectbox("Filtrar por", ["—"] + cols)
            if filt_col != "—":
                if pd.api.types.is_numeric_dtype(df[filt_col]):
                    lo, hi = float(df[filt_col].min()), float(df[filt_col].max())
                    rng = st.slider("Rango", lo, hi, (lo, hi))
                    view = df[df[filt_col].between(*rng)]
                else:
                    vals = st.multiselect("Valores", sorted(df[filt_col].dropna().astype(str).unique())[:100])
                    view = df[df[filt_col].astype(str).isin(vals)] if vals else df
                st.write(f"Filas filtradas: {len(view):,}")
                st.dataframe(view.head(100))

    # --- Estadística ---
    with tabs[4]:
        st.subheader("Estadística dirigida")
        profile = st.session_state.get("profile") or profile_dataframe(df, selected)
        mode = st.radio("Análisis", ["Comparación de grupos", "Regresión lineal", "Regresión logística"], horizontal=True)
        if mode == "Comparación de grupos":
            g = st.selectbox("Grupo", profile.categorical_columns or list(df.columns))
            v = st.selectbox("Valor", profile.numeric_columns or list(df.columns))
            if st.button("Ejecutar comparación"):
                res = group_comparison(df, v, g)
                st.json(res)
                if res.get("ok") and res.get("group_means"):
                    st.plotly_chart(group_means_bar(res["group_means"]), use_container_width=True)
                    st.plotly_chart(boxplot(df, v, g), use_container_width=True)
        elif mode == "Regresión lineal":
            y = st.selectbox("y", profile.numeric_columns or list(df.columns), key="lin_y")
            xs = st.multiselect("x", [c for c in df.columns if c != y], key="lin_x")
            if xs and st.button("Ajustar OLS"):
                res = linear_regression(df, y, xs)
                st.text(res.get("summary_text") or res)
                st.write(res.get("limitations"))
        else:
            y = st.selectbox("y binaria", profile.possible_targets or list(df.columns), key="log_y")
            xs = st.multiselect("x", [c for c in df.columns if c != y], key="log_x")
            if xs and st.button("Ajustar Logit"):
                res = logistic_regression(df, y, xs)
                st.text(res.get("summary_text") or res)
                st.write(res.get("limitations"))

    # --- Segmentación ---
    with tabs[5]:
        st.subheader("Segmentación")
        profile = st.session_state.get("profile") or profile_dataframe(df, selected)
        nums = st.multiselect("Numéricas", profile.numeric_columns, default=profile.numeric_columns[:5])
        cats = st.multiselect("Categóricas (mixtas)", profile.categorical_columns, default=[])
        if st.button("Correr segmentación") and len(nums) >= 2:
            with st.spinner("Evaluando K-means / jerárquico / DBSCAN…"):
                seg = run_segmentation(df, nums, cats)
            st.session_state["seg"] = seg
        seg = st.session_state.get("seg")
        if seg and seg.get("ok"):
            st.write(seg.get("caveat"))
            st.write("**K-means grid**", seg["kmeans"]["grid"])
            st.write("**Estabilidad**", seg.get("stability"))
            st.write("**DBSCAN best**", seg["dbscan"]["best"])
            if seg.get("mixed"):
                st.write("**Mixtos (Gower / k-prototypes)**", seg["mixed"])
            labels = seg["kmeans"].get("labels")
            coords = seg.get("pca_coords")
            if labels is not None and coords is not None:
                st.plotly_chart(cluster_scatter(coords, labels), use_container_width=True)
            st.write("**Limitaciones:**", seg.get("limitations"))

    # --- Causal ---
    with tabs[6]:
        st.subheader("Análisis causal exploratorio")
        st.warning(
            "Ningún resultado demuestra causalidad. "
            "Incluye PSM, DiD, adopción escalonada y editor de DAG."
        )
        causal_mode = st.radio(
            "Modo",
            ["Ajuste / PSM", "DiD multi-período", "Adopción escalonada", "Editor DAG"],
            horizontal=True,
        )
        cols = list(df.columns)
        if causal_mode == "Ajuste / PSM":
            treatment = st.selectbox("Treatment", cols, key="c_t")
            outcome = st.selectbox("Outcome", [c for c in cols if c != treatment], key="c_o")
            confounders = st.multiselect(
                "Confusores hipotéticos",
                [c for c in cols if c not in {treatment, outcome}],
                key="c_c",
            )
            if st.button("Explorar causalmente"):
                res = exploratory_causal_screen(df, treatment, outcome, confounders)
                st.session_state["causal"] = res
            res = st.session_state.get("causal")
            if res:
                st.write(res["disclaimer"])
                st.write("**Asociación ajustada**", res["adjusted_association"])
                if res.get("sensitivity_adjusted"):
                    st.write("**E-value / sensibilidad (ajuste)**", res["sensitivity_adjusted"])
                psm = res.get("propensity_matching")
                if psm and psm.get("ok"):
                    st.write("**PSM**", {k: psm[k] for k in psm if k not in {"matched_outcomes", "sensitivity"}})
                    if psm.get("covariate_balance"):
                        st.dataframe(pd.DataFrame(psm["covariate_balance"]), use_container_width=True)
                    if psm.get("sensitivity"):
                        st.write("**Sensibilidad PSM**", psm["sensitivity"])
            if st.button("Estimar ATE con DoWhy (opcional)"):
                st.session_state["dowhy"] = estimate_ate_dowhy(df, treatment, outcome, confounders)
            dw = st.session_state.get("dowhy")
            if dw:
                if dw.get("ok"):
                    st.write("**DoWhy**", dw)
                else:
                    st.info(dw.get("reason") or "DoWhy no disponible")
                st.caption(f"DoWhy instalado: {dowhy_available()}")
        elif causal_mode == "DiD multi-período":
            outcome = st.selectbox("Outcome", cols, key="did_o")
            gcol = st.selectbox("Grupo tratado/control", [c for c in cols if c != outcome], key="did_g")
            tcol = st.selectbox("Tiempo / período", [c for c in cols if c not in {outcome, gcol}], key="did_t")
            levels = sorted(df[gcol].dropna().unique().tolist(), key=lambda x: str(x))
            treated = st.selectbox("Nivel tratado", levels, key="did_tl")
            periods = sorted(df[tcol].dropna().unique().tolist(), key=lambda x: str(x))
            post = st.multiselect("Períodos post", periods, default=periods[-1:] if periods else [])
            if post and st.button("Correr DiD"):
                st.session_state["did"] = difference_in_differences_multipperiod(
                    df, outcome, gcol, tcol, treated, post
                )
            did = st.session_state.get("did")
            if did:
                st.write(did)
                pre = did.get("pretrends") or {}
                if pre.get("ok"):
                    st.info(pre.get("interpretation"))
        elif causal_mode == "Adopción escalonada":
            st.caption(
                "ATT(g,t) simplificado estilo Callaway–Sant'Anna + AIPW agregado. "
                "Requiere panel unit×time y columna de primer período tratado (NaN = never-treated)."
            )
            outcome = st.selectbox("Outcome", cols, key="cs_o")
            unit = st.selectbox("Unidad", [c for c in cols if c != outcome], key="cs_u")
            time_c = st.selectbox("Tiempo", [c for c in cols if c not in {outcome, unit}], key="cs_t")
            mode_tt = st.radio(
                "Origen del tiempo de tratamiento",
                ["Columna de primer tratamiento", "Derivar de indicador 0/1"],
                horizontal=True,
            )
            work = df.copy()
            if mode_tt.startswith("Columna"):
                tt = st.selectbox(
                    "Primer período tratado",
                    [c for c in cols if c not in {outcome, unit, time_c}],
                    key="cs_tt",
                )
            else:
                tbin = st.selectbox(
                    "Indicador de tratamiento",
                    [c for c in cols if c not in {outcome, unit, time_c}],
                    key="cs_tb",
                )
                work = work.copy()
                work["_treat_time"] = prepare_treat_time_from_binary(work, unit, time_c, tbin)
                tt = "_treat_time"
            aipw_covars = st.multiselect(
                "Covariables AIPW (opcional)",
                [c for c in cols if c not in {outcome, unit, time_c, tt}],
                key="cs_aipw_x",
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Estimar ATT escalonado"):
                    st.session_state["stag"] = staggered_att(work, outcome, unit, time_c, tt)
            with b2:
                if st.button("Estimar AIPW agregado"):
                    st.session_state["aipw"] = aipw_staggered_att(
                        work,
                        outcome,
                        unit,
                        time_c,
                        tt,
                        covariate_cols=aipw_covars or None,
                        n_boot=200,
                    )
            stag = st.session_state.get("stag")
            if stag:
                st.write(stag.get("disclaimer"))
                m1, m2, m3 = st.columns(3)
                m1.metric("ATT simple", stag.get("overall_att_simple_avg"))
                m2.metric("ATT ponderado", stag.get("overall_att_weighted"))
                dr = stag.get("doubly_robust_lite") or {}
                m3.metric("ATT DR-lite", dr.get("att") if dr.get("ok") else None)
                st.write("Control:", stag.get("control_mode"), "· celdas:", stag.get("n_att_cells"))
                if stag.get("pretrend_check"):
                    st.write("**Placebos / pre-trends**", stag["pretrend_check"])
                if stag.get("att_by_event_time_weighted"):
                    st.write("**Event-time ponderado**")
                    st.dataframe(pd.DataFrame(stag["att_by_event_time_weighted"]), use_container_width=True)
                if stag.get("ok"):
                    st.dataframe(pd.DataFrame(stag["att_by_cohort_time"]), use_container_width=True)
                else:
                    st.warning(stag.get("reason"))
                if dr:
                    st.write("**DR-lite**", dr)
                st.caption(" · ".join(stag.get("limitations") or []))
            aipw = st.session_state.get("aipw")
            if aipw:
                st.markdown("##### AIPW agregado (pre→post)")
                if aipw.get("ok"):
                    a1, a2, a3 = st.columns(3)
                    a1.metric("ATT AIPW", aipw.get("att_aipw"))
                    a2.metric("ATE AIPW", aipw.get("ate_aipw"))
                    ci = aipw.get("ci95_att_bootstrap") or [None, None]
                    a3.metric("IC95 ATT", f"{ci[0]:.3f} … {ci[1]:.3f}" if ci[0] is not None and np.isfinite(ci[0]) else "—")
                    st.write(aipw)
                    st.caption(" · ".join(aipw.get("limitations") or []))
                else:
                    st.warning(aipw.get("reason"))
        else:
            st.caption("Editor DAG: roles + aristas + canvas interactivo (vis-network).")
            nodes = st.multiselect("Nodos", cols, default=cols[: min(6, len(cols))], key="dag_nodes")
            roles = {}
            if nodes:
                rc1, rc2 = st.columns(2)
                for i, n in enumerate(nodes):
                    with (rc1 if i % 2 == 0 else rc2):
                        roles[n] = st.selectbox(
                            f"Rol · {n}",
                            list(ALLOWED_ROLES),
                            index=list(ALLOWED_ROLES).index("other"),
                            key=f"dag_role_{n}",
                        )
            if st.button("Sugerir aristas por roles"):
                st.session_state["dag_edges"] = suggest_edges_from_roles(roles)
            edge_src = st.selectbox("Arista desde", ["—"] + nodes, key="e_src")
            edge_dst = st.selectbox("Arista hacia", ["—"] + [n for n in nodes if n != edge_src], key="e_dst")
            if "dag_edges" not in st.session_state:
                st.session_state["dag_edges"] = []
            if st.button("Agregar arista") and edge_src != "—" and edge_dst != "—":
                e = [edge_src, edge_dst]
                if e not in st.session_state["dag_edges"]:
                    st.session_state["dag_edges"].append(e)
            if st.button("Limpiar aristas"):
                st.session_state["dag_edges"] = []
            st.write("Aristas:", st.session_state["dag_edges"])
            dag = build_dag(nodes, st.session_state["dag_edges"], roles)
            val = validate_dag(dag)
            st.write("**Validación**", val)
            dag_payload = {
                "nodes": nodes,
                "edges": st.session_state["dag_edges"],
                "roles": roles,
            }
            st.download_button(
                "Descargar DAG JSON",
                data=json.dumps(dag_payload, ensure_ascii=False, indent=2),
                file_name="dag.json",
                mime="application/json",
                key="dag_dl",
            )
            dag_file = st.file_uploader("Subir DAG JSON", type=["json"], key="dag_upload")
            if dag_file is not None and st.button("Importar archivo DAG"):
                try:
                    parsed = parse_exported_dag_json(dag_file.getvalue().decode("utf-8"))
                    st.session_state["dag_edges"] = parsed["dag"]["edges"]
                    st.write("**Validación importada**", parsed["validation"])
                    st.success("DAG cargado desde archivo")
                except Exception as exc:
                    st.error(f"JSON inválido: {exc}")
            if nodes:
                st.plotly_chart(dag_to_plotly(dag), use_container_width=True)
                st.markdown("##### Canvas interactivo (arrastrar nodos)")
                render_interactive_dag(nodes, st.session_state["dag_edges"], roles)
                exported = st.text_area("Pegá aquí el JSON exportado del canvas")
                if exported.strip() and st.button("Importar JSON del canvas"):
                    try:
                        parsed = parse_exported_dag_json(exported)
                        st.session_state["dag_edges"] = parsed["dag"]["edges"]
                        st.write("**Validación importada**", parsed["validation"])
                        st.success("Aristas actualizadas desde el canvas")
                    except Exception as exc:
                        st.error(f"JSON inválido: {exc}")

    # --- Drift ---
    with tabs[7]:
        st.subheader("Comparación entre datasets (drift)")
        if len(tables) < 2:
            st.info("Cargá al menos dos archivos para comparar, o usá un split manual abajo.")
            frac = st.slider("Split aleatorio de la tabla activa (para demo)", 0.1, 0.9, 0.5)
            if st.button("Comparar split aleatorio"):
                a = df.sample(frac=frac, random_state=42)
                b = df.drop(a.index)
                st.session_state["drift"] = compare_datasets(a, b, "split_a", "split_b")
        else:
            a_name = st.selectbox("Dataset A", names, key="drift_a")
            b_name = st.selectbox("Dataset B", [n for n in names if n != a_name] or names, key="drift_b")
            if st.button("Comparar datasets"):
                st.session_state["drift"] = compare_datasets(tables[a_name], tables[b_name], a_name, b_name)
        drift = st.session_state.get("drift")
        if drift:
            st.write(drift["interpretation"])
            st.write(f"Driftados: {drift['n_drifted']}")
            st.dataframe(pd.DataFrame(drift["column_tests"]), use_container_width=True)
            if drift.get("only_in_a") or drift.get("only_in_b"):
                st.write("Solo en A:", drift["only_in_a"])
                st.write("Solo en B:", drift["only_in_b"])

    # --- Informe / Sesiones ---
    with tabs[8]:
        st.subheader("Informes y persistencia")
        profile = st.session_state.get("profile")
        findings = st.session_state.get("findings")
        artifacts = st.session_state.get("artifacts", {})
        if not profile or findings is None:
            st.info("Ejecutá el descubrimiento primero.")
        else:
            out_dir = ROOT / "outputs"
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Markdown + JSON"):
                    md = generate_markdown_report(
                        profile, findings, artifacts, out_dir / f"{selected}_informe.md"
                    )
                    export_findings_json(findings, out_dir / f"{selected}_hallazgos.json")
                    st.session_state["last_md"] = md
                    st.success(f"Exportado a {out_dir}")
            with c2:
                if st.button("Exportar PDF"):
                    pdf_path = out_dir / f"{selected}_informe.pdf"
                    with st.spinner("Generando PDF…"):
                        generate_pdf_report(df, profile, findings, pdf_path, artifacts)
                    st.session_state["last_pdf"] = str(pdf_path)
                    st.success(f"PDF: {pdf_path}")
                    with open(pdf_path, "rb") as fh:
                        st.download_button("Descargar PDF", fh, file_name=pdf_path.name)
            with c3:
                sess_name = st.text_input("Nombre de sesión", value=selected)
                if st.button("Guardar sesión"):
                    meta = store.save(
                        df, profile, findings, artifacts, name=sess_name, notes="UI save"
                    )
                    st.success(f"Sesión guardada: {meta.id}")

            if st.session_state.get("last_md"):
                st.download_button(
                    "Descargar Markdown",
                    st.session_state["last_md"],
                    file_name=f"{selected}_informe.md",
                )
            st.markdown(generate_markdown_report(profile, findings, artifacts)[:4000])

        st.subheader("Comparar dos sesiones guardadas")
        sessions = store.list_sessions()
        if len(sessions) < 2:
            st.caption("Guardá al menos dos sesiones para comparar hallazgos.")
        else:
            labels = {f"{s['name']} ({s['id']})": s["id"] for s in sessions}
            a = st.selectbox("Sesión A", list(labels), key="cmp_a")
            b = st.selectbox("Sesión B", [x for x in labels if x != a] or list(labels), key="cmp_b")
            if st.button("Comparar sesiones"):
                la = store.load(labels[a])
                lb = store.load(labels[b])
                fa = {f["id"]: f for f in la["findings"]}
                fb = {f["id"]: f for f in lb["findings"]}
                titles_a = {f["title"] for f in la["findings"]}
                titles_b = {f["title"] for f in lb["findings"]}
                only_a = sorted(titles_a - titles_b)
                only_b = sorted(titles_b - titles_a)
                common = sorted(titles_a & titles_b)
                st.write(f"Comunes: {len(common)} | Solo A: {len(only_a)} | Solo B: {len(only_b)}")
                st.write("**Solo en A**", only_a[:30])
                st.write("**Solo en B**", only_b[:30])
                # drift de datos si ambos tienen df
                if la.get("df") is not None and lb.get("df") is not None:
                    common_cols = [c for c in la["df"].columns if c in lb["df"].columns]
                    if common_cols:
                        drift = compare_datasets(
                            la["df"][common_cols],
                            lb["df"][common_cols],
                            labels[a],
                            labels[b],
                        )
                        st.write("**Drift entre datasets de sesión**", drift["interpretation"])
                        st.dataframe(pd.DataFrame(drift["column_tests"]).head(40), use_container_width=True)


if __name__ == "__main__":
    main()
