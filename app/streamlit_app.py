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

from estacion.core.loader import load_table, suggest_join_keys
from estacion.core.profiler import profile_dataframe
from estacion.core.drift import compare_datasets
from estacion.core.session import SessionStore
from estacion.core.quality_rules import (
    DOMAIN_PRESETS,
    QualityRuleSet,
    load_rules,
    evaluate_quality_rules,
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
from estacion.ux.presentation import (
    default_explore_columns,
    default_causal_columns,
    format_p_value,
    format_number,
)
try:
    from app.panels import (
        render_welcome,
        render_load_preview,
        render_finding_cards,
        render_plain_summary,
        render_summary_header,
        render_segmentation_human,
        DEMO_SELECT_KEY,
        WELCOME_UPLOAD_KEY,
    )
except ImportError:  # streamlit a veces pone app/ en sys.path
    from panels import (  # type: ignore
        render_welcome,
        render_load_preview,
        render_finding_cards,
        render_plain_summary,
        render_summary_header,
        render_segmentation_human,
        DEMO_SELECT_KEY,
        WELCOME_UPLOAD_KEY,
    )
import json

st.set_page_config(
    page_title="Estación de Análisis",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea {
  font-family: Inter, "Segoe UI", sans-serif !important;
}
.stApp { background: #F6F8FB; color: #0F172A; }
h1, h2, h3 { color: #0F172A !important; letter-spacing: -0.02em; font-weight: 700 !important; }
.brand {
  font-size: 1.15rem;
  letter-spacing: -0.01em;
  color: #0F172A;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.25rem;
}
.brand:before {
  content: "";
  width: 26px; height: 26px; border-radius: 8px;
  background: linear-gradient(135deg, #4F46E5, #06B6D4);
  flex: none;
}
.ea-eyebrow {
  display: inline-block;
  background: #EEF2FF; color: #4F46E5;
  font-weight: 600; font-size: 0.8rem;
  padding: 0.35rem 0.75rem; border-radius: 999px;
  margin-bottom: 0.75rem;
}
.ea-hero-title {
  font-size: clamp(1.7rem, 4vw, 2.6rem) !important;
  line-height: 1.12 !important;
  letter-spacing: -0.03em !important;
  color: #0F172A !important;
  margin: 0.2rem 0 0.6rem 0 !important;
}
.ea-lead { color: #475569; font-size: 1.05rem; line-height: 1.5; max-width: 40rem; }
.ea-headline {
  font-size: clamp(1.45rem, 3vw, 1.95rem) !important;
  line-height: 1.2 !important;
  letter-spacing: -0.025em !important;
  color: #0F172A !important;
  margin: 0.15rem 0 0.4rem 0 !important;
}
.ea-subhead { color: #475569; font-size: 1.02rem; line-height: 1.5; max-width: 48rem; }
.ea-sec {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
  color: #64748B; font-weight: 600; margin: 0.4rem 0 0.6rem 0;
}
.ea-lbl { color: #94A3B8; font-size: 0.8rem; font-weight: 600; margin: 0.55rem 0 0.1rem 0; }
.ea-dropzone {
  background: #fff; border: 2px dashed #C7D2FE; border-radius: 18px;
  padding: 0.4rem 0.6rem 0.2rem; margin: 0.6rem 0 0.8rem;
}
/* Maquillaje frágil del uploader (Streamlit no traduce el chrome en inglés) */
[data-testid="stFileUploaderDropzone"] small {
  visibility: hidden; position: relative;
}
[data-testid="stFileUploaderDropzone"] small::after {
  content: "CSV, XLSX · hasta 200 MB";
  visibility: visible; position: absolute; left: 0; top: 0;
  color: #64748B; font-size: 0.85rem;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
[data-testid="stFileUploaderDropzone"] span[data-testid="stFileUploaderDropzoneInstructions"] span {
  font-size: 0 !important;
}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p::after {
  content: "Arrastrá tu archivo acá";
  font-size: 0.95rem !important; color: #0F172A; font-weight: 600;
}
div[data-testid="stMetric"] {
  background: #fff; border: 1px solid #E5E9F0; border-radius: 16px;
  padding: 0.85rem 1rem;
}
@media (max-width: 768px) {
  div[data-testid="stMetric"] label { font-size: 0.75rem !important; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.15rem !important; }
  .brand { font-size: 1.05rem; }
  .ea-hero-title { font-size: 1.55rem !important; }
  .ea-headline { font-size: 1.35rem !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _clear_loaded_data() -> None:
    """Quitar archivo / volver al inicio (limpia tablas y uploaders)."""
    for k in (
        "loaded_tables",
        "load_metas",
        "load_source",
        "findings",
        "profile",
        "artifacts",
        "narratives",
        "active_table",
        "quality_domain",
        WELCOME_UPLOAD_KEY,
        "sidebar_uploaded_files",
        "repair_uploaded_files",
        "_prefer_loaded_upload",
        "_run_analyze",
        "_pdf_bytes_keys",
    ):
        st.session_state.pop(k, None)
    # Limpiar caches de PDF
    for k in list(st.session_state.keys()):
        if str(k).startswith("_pdf_bytes::"):
            st.session_state.pop(k, None)
    st.session_state[DEMO_SELECT_KEY] = "—"

DEMO_LABELS = {
    "—": "—",
    "commercial": "Ventas (ejemplo)",
    "financial": "Finanzas (ejemplo)",
    "operational": "Operaciones (ejemplo)",
    "segments": "Segmentos (ejemplo)",
    "causal_confounded": "Causal con confusor (prueba)",
    "nonlinear_null": "Relación nula no lineal (prueba)",
}
DOMAIN_LABELS = {k: ("General" if k == "default" else k.capitalize()) for k in DOMAIN_PRESETS}


def _read_upload_bytes(f) -> bytes:
    """Leer bytes de un UploadedFile rebobinando el buffer (Streamlit no siempre resetea)."""
    if hasattr(f, "seek"):
        try:
            f.seek(0)
        except Exception:
            pass
    data = f.read() if hasattr(f, "read") else b""
    if not data and hasattr(f, "getvalue"):
        try:
            data = f.getvalue()
        except Exception:
            data = b""
    return data or b""


def _load_uploaded(files) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    tables = {}
    metas = {}
    for f in files:
        data = _read_upload_bytes(f)
        df, meta = load_table(data, name=Path(f.name).stem)
        tables[meta["name"]] = df
        metas[meta["name"]] = meta
    return tables, metas


def _apply_uploaded_tables(tables: dict, metas: dict) -> None:
    st.session_state["loaded_tables"] = tables
    st.session_state["load_metas"] = metas
    st.session_state["load_source"] = "upload"


@st.cache_data(show_spinner=False)
def _cached_discovery(df: pd.DataFrame, name: str, quality_domain: str = "default"):
    return run_discovery(df, name=name, quality_domain=quality_domain)


def _analysis_ready(selected, quality_domain) -> bool:
    return (
        "findings" in st.session_state
        and st.session_state.get("active_table") == selected
        and st.session_state.get("quality_domain") == quality_domain
    )


def _run_discovery(df, selected, quality_domain, use_ollama):
    with st.spinner("1/3 Revisando calidad… 2/3 Buscando relaciones… 3/3 Detectando anomalías…"):
        # invalidar cache si cambió dominio: clave ya incluye quality_domain
        findings, profile, artifacts = _cached_discovery(df, selected, quality_domain)
    st.session_state["findings"] = findings
    st.session_state["profile"] = profile
    st.session_state["artifacts"] = artifacts
    st.session_state["active_table"] = selected
    st.session_state["quality_domain"] = quality_domain
    st.session_state["narratives"] = interpret_findings(findings, try_ollama=use_ollama)
    return findings, profile, artifacts


def _render_explore(df, profile):
    st.subheader("Explorar")
    st.caption("Mirálos vos: histograma, cajas y relaciones. Los ejes usan formato es-AR.")
    defaults = default_explore_columns(profile)
    cols = list(df.columns)
    plot_cols = [c for c in cols if c not in set(profile.identifier_columns)] or cols
    left, right = st.columns(2)
    with left:
        hist_default = defaults["hist"] if defaults["hist"] in plot_cols else plot_cols[0]
        hist_idx = plot_cols.index(hist_default) if hist_default in plot_cols else 0
        col = st.selectbox("¿Cómo se reparten los valores?", plot_cols, index=hist_idx, key="hist_col")
        color_opts = ["—"] + [c for c in profile.categorical_columns if c in plot_cols]
        color = st.selectbox("Color (opcional)", color_opts, key="hist_color")
        st.plotly_chart(histogram(df, col, None if color == "—" else color), width="stretch")
    with right:
        nums = [c for c in (profile.numeric_columns or plot_cols) if c not in set(profile.identifier_columns)]
        nums = nums or plot_cols
        x_def = defaults["scatter_x"] if defaults["scatter_x"] in nums else nums[0]
        y_cands = [c for c in nums if c != x_def] or nums
        y_def = defaults["scatter_y"] if defaults["scatter_y"] in y_cands else y_cands[0]
        x = st.selectbox("¿Se mueven juntas? Eje X", nums, index=nums.index(x_def), key="sx")
        y = st.selectbox(
            "Eje Y",
            [c for c in nums if c != x] or nums,
            index=max(0, ([c for c in nums if c != x] or nums).index(y_def)
                      if y_def in ([c for c in nums if c != x] or nums) else 0),
            key="sy",
        )
        st.plotly_chart(scatter(df, x, y, trendline=True), width="stretch")

    if len(profile.numeric_columns) >= 2:
        corr = correlation_analysis(df, profile.numeric_columns[:20])
        st.plotly_chart(
            correlation_heatmap(corr["spearman_matrix"], "Mapa de relaciones (más oscuro = más relacionadas)"),
            width="stretch",
        )

    if profile.datetime_columns and profile.numeric_columns:
        dcol = st.selectbox("Fecha", profile.datetime_columns)
        vcol = st.selectbox("Valor temporal", profile.numeric_columns, key="ts_val")
        st.plotly_chart(time_series_plot(df, dcol, vcol), width="stretch")
        with st.expander("Detalle temporal y pronóstico", expanded=False):
            tres = temporal_analysis(df, dcol, vcol)
            st.write(
                tres.get("interpretation")
                or tres.get("summary")
                or "Análisis temporal calculado."
            )
            with st.expander("Ver datos crudos del temporal"):
                st.write({k: tres[k] for k in tres if k not in {"limitations", "series_tail"}})
            horizon = st.slider(
                "Horizonte",
                4,
                36,
                12,
                help="Cuántos períodos a futuro querés proyectar.",
            )
            fmethod = st.selectbox(
                "Método de pronóstico",
                ["aic_grid", "auto", "ets", "arima"],
                format_func=lambda m: {
                    "aic_grid": "Elegir el mejor (AIC)",
                    "auto": "Automático",
                    "ets": "Suavizado exponencial (ETS)",
                    "arima": "ARIMA",
                }.get(m, m),
                help="AIC compara modelos; conformal calibra intervalos con errores recientes.",
            )
            use_conf = st.checkbox(
                "Intervalos conformal",
                value=True,
                help="Bandas de incertidumbre calibradas con errores de predicción pasados.",
            )
            if st.button("Pronosticar"):
                st.session_state["forecast"] = forecast_series(
                    df, dcol, vcol, horizon=horizon, method=fmethod, conformal=use_conf
                )
            fc = st.session_state.get("forecast")
            if fc:
                if fc.get("ok"):
                    st.write(f"Método: `{fc['method']}`")
                    st.dataframe(pd.DataFrame(fc["forecast"]), width="stretch")
                    st.caption("; ".join(fc.get("limitations") or []))
                else:
                    st.warning(fc.get("reason"))


def _render_informe(df, selected, store):
    st.subheader("Informe")
    profile = st.session_state.get("profile")
    findings = st.session_state.get("findings")
    artifacts = st.session_state.get("artifacts", {})
    if not profile or findings is None:
        st.info("Ejecutá el análisis primero.")
        return
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # PDF en un solo clic: generar (o reutilizar cache de sesión) y ofrecer download directo.
    cache_key = f"_pdf_bytes::{selected}::{len(findings)}::{profile.n_rows}"
    if cache_key not in st.session_state:
        pdf_path = out_dir / f"{selected}_informe.pdf"
        with st.spinner("Preparando PDF…"):
            generate_pdf_report(df, profile, findings, pdf_path, artifacts)
            st.session_state[cache_key] = pdf_path.read_bytes()
    st.download_button(
        "Descargar informe (PDF)",
        data=st.session_state[cache_key],
        file_name=f"{selected}_informe.pdf",
        mime="application/pdf",
        type="primary",
        key="pdf_dl_direct",
    )

    with st.expander("Otros formatos"):
        md = generate_markdown_report(profile, findings, artifacts)
        st.download_button("Descargar Markdown", md, file_name=f"{selected}_informe.md")
        js = json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2, default=str)
        st.download_button("Descargar hallazgos (JSON)", js, file_name=f"{selected}_hallazgos.json")
        if st.button("También guardar en carpeta outputs/"):
            generate_markdown_report(profile, findings, artifacts, out_dir / f"{selected}_informe.md")
            export_findings_json(findings, out_dir / f"{selected}_hallazgos.json")
            st.success("Guardado en outputs/")

    with st.expander("Guardar este análisis para verlo después"):
        st.caption(
            "Guardá una copia local del análisis (datos + hallazgos) para reabrirla "
            "más adelante desde “Cargar sesión” en la barra lateral."
        )
        sess_name = st.text_input("Nombre para este análisis", value=selected)
        if st.button("Guardar análisis", help="Queda disponible en esta computadora, en outputs/sessions."):
            meta = store.save(df, profile, findings, artifacts, name=sess_name, notes="UI save")
            st.success(f"Análisis guardado: {meta.id}")

    st.markdown("#### Vista previa del informe")
    # Una sola vez: el markdown ya incluye el resumen en lenguaje simple.
    st.markdown(generate_markdown_report(profile, findings, artifacts)[:3500])


def _build_copy_context(df, profile, selected: str):
    from estacion.ux.presentation import (
        CopyContext,
        friendly_dataset_name,
        load_dictionary_units,
        resolve_money_unit,
    )

    units = load_dictionary_units(name_hint=selected or getattr(profile, "name", None))
    override = st.session_state.get("money_unit_override")
    if override in (None, "Auto"):
        override = None
    target = None
    if profile is not None:
        target = (profile.possible_targets or profile.numeric_columns or [None])[0]
    money = resolve_money_unit(column=target, dictionary_units=units, override=override)
    # Si el diccionario trae $ en revenue, usarlo aunque el target no esté
    if money == "unidades" and units:
        for col, u in units.items():
            if u and str(u).strip() in {"$", "USD", "ARS"}:
                money = str(u).strip()
                break
    return CopyContext(
        df=df,
        profile=profile,
        money_unit=money,
        dataset_label=friendly_dataset_name(selected or getattr(profile, "name", None)),
    )


def main():
    # Reset de uploads antes de instanciar widgets (on_click del botón "Quitar…").
    if st.session_state.pop("_reset_uploads", False):
        _clear_loaded_data()

    st.markdown('<div class="brand">Estación de Análisis</div>', unsafe_allow_html=True)

    store = SessionStore(ROOT / "outputs" / "sessions")

    with st.sidebar:
        st.header("Tus datos")
        uploaded = st.file_uploader(
            "Subí CSV o Excel",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="También acepta CSV de Excel en español (separador ; y decimal ,).",
            key="sidebar_uploaded_files",
        )
        st.selectbox(
            "Unidad monetaria",
            ["Auto", "$", "unidades"],
            key="money_unit_override",
            help="Auto usa el diccionario de datos si tiene unidad; si no, 'unidades'.",
        )
        demo_keys = list(DEMO_LABELS.keys())
        # El botón de la portada (on_click) setea DEMO_SELECT_KEY antes de este
        # punto en el rerun. Key fija: el valor persiste entre interacciones.
        if st.session_state.pop("want_commercial", None):
            st.session_state["_auto_analyze_once"] = True
            # want_commercial se setea en el mismo on_click que DEMO_SELECT_KEY;
            # no volvemos a asignar DEMO_SELECT_KEY acá (el selectbox ya puede existir).
        if DEMO_SELECT_KEY not in st.session_state:
            st.session_state[DEMO_SELECT_KEY] = "—"
        elif st.session_state[DEMO_SELECT_KEY] not in demo_keys:
            st.session_state[DEMO_SELECT_KEY] = "—"
        demo = st.selectbox(
            "O probá un ejemplo",
            demo_keys,
            key=DEMO_SELECT_KEY,
            format_func=lambda k: DEMO_LABELS[k],
        )

        advanced = st.toggle("Modo avanzado", value=False, help="Activa estadística, causal, plugins, etc.")
        run_btn = st.button(
            "Analizar",
            type="primary",
            width="stretch",
            help="Elegir un archivo no analiza solo: hay que pulsar Analizar.",
        )

        with st.expander("Opciones avanzadas"):
            quality_domain = st.selectbox(
                "Dominio de reglas de calidad",
                list(DOMAIN_PRESETS.keys()),
                format_func=lambda k: DOMAIN_LABELS.get(k, k),
            )
            use_ollama = st.checkbox(
                "Narrativas vía Ollama local",
                value=False,
                help="Opcional. Requiere Ollama en localhost. Sin API paga.",
            )
            st.caption("Sesiones guardadas")
            sessions = store.list_sessions()
            session_labels = ["—"] + [f"{s['name']} ({s['id']})" for s in sessions]
            pick_session = st.selectbox("Cargar sesión", session_labels)

    tables: dict[str, pd.DataFrame] = {}
    load_metas: dict[str, dict] = {}

    # Prioridad de carga:
    # 1) reparación ("Subir otro archivo") — gana sobre el inválido de la portada
    # 2) sidebar
    # 3) portada
    # 4) tablas ya parseadas en session_state
    # Siempre seek(0) al leer: si no, un re-read tras rerun devuelve vacío.
    repair_files = st.session_state.get("repair_uploaded_files") or []
    welcome_uploaded = st.session_state.get(WELCOME_UPLOAD_KEY) or []

    if pick_session and pick_session != "—":
        sid = pick_session.rsplit("(", 1)[-1].rstrip(")")
        loaded = store.load(sid)
        if loaded.get("df") is not None:
            tables = {loaded["meta"]["dataset_name"]: loaded["df"]}
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
            st.session_state["narratives"] = interpret_findings(findings_loaded, try_ollama=use_ollama)
            st.session_state["loaded_tables"] = tables
            st.session_state["load_source"] = "session"
            st.sidebar.success(f"Sesión {sid} cargada")
    elif repair_files:
        tables, load_metas = _load_uploaded(repair_files)
        _apply_uploaded_tables(tables, load_metas)
        # El inválido de la portada no debe volver a pisar este resultado.
        st.session_state.pop(WELCOME_UPLOAD_KEY, None)
        st.session_state["_prefer_loaded_upload"] = True
    elif uploaded:
        tables, load_metas = _load_uploaded(uploaded)
        _apply_uploaded_tables(tables, load_metas)
        st.session_state.pop("repair_uploaded_files", None)
        st.session_state.pop("_prefer_loaded_upload", None)
    elif welcome_uploaded and not st.session_state.get("_prefer_loaded_upload"):
        tables, load_metas = _load_uploaded(welcome_uploaded)
        _apply_uploaded_tables(tables, load_metas)
    elif demo and demo != "—":
        path = ROOT / "data" / "synthetic" / f"{demo}.csv"
        if path.exists():
            df, meta = load_table(path)
            tables = {meta["name"]: df}
            load_metas = {meta["name"]: meta}
            st.session_state["loaded_tables"] = tables
            st.session_state["load_metas"] = load_metas
            st.session_state["load_source"] = "demo"
        else:
            st.warning("Dataset de ejemplo no generado. Ejecutá: python -m data.synthetic.generate_datasets")
    elif st.session_state.get("loaded_tables") and st.session_state.get("load_source") == "upload":
        # Recuperar tablas ya parseadas (p.ej. tras salir de la portada o tras reparar).
        tables = st.session_state["loaded_tables"]
        load_metas = st.session_state.get("load_metas") or {}

    if not tables:
        render_welcome(demo_select_key=DEMO_SELECT_KEY)
        return

    names = list(tables.keys())
    selected = st.sidebar.selectbox("Tabla activa", names) if len(names) > 1 else names[0]
    df = tables[selected]
    meta = (st.session_state.get("load_metas") or load_metas or {}).get(selected) or {}
    from_upload = st.session_state.get("load_source") == "upload"
    bad_parse = bool(meta.get("suspicious_parse") and meta.get("n_cols") == 1)

    def _on_clear_upload() -> None:
        st.session_state["_reset_uploads"] = True
        st.session_state.pop("repair_uploaded_files", None)
        st.session_state.pop("_prefer_loaded_upload", None)

    auto_once = st.session_state.pop("_auto_analyze_once", False)
    run_analyze_now = bool(
        run_btn or auto_once or st.session_state.pop("_run_analyze", False)
    )
    show_analyze_prompt = not _analysis_ready(selected, quality_domain) and not run_analyze_now
    # Evitar tabla duplicada: expander (mensajes) + preview bajo Analizar.
    preview_table_in_expander = not (from_upload and show_analyze_prompt and not bad_parse)

    if from_upload or meta.get("warnings") or meta.get("suspicious_parse") or bad_parse:
        with st.expander(
            "Vista previa de la carga",
            expanded=bool(meta.get("suspicious_parse")) or meta.get("n_cols") == 1,
        ):
            render_load_preview(df, meta, show_table=preview_table_in_expander)

    if from_upload or bad_parse:
        c_clear, c_up = st.columns([1, 2])
        with c_clear:
            st.button(
                "Quitar archivo / Volver al inicio",
                on_click=_on_clear_upload,
                key="btn_clear_upload",
            )
        if bad_parse:
            with c_up:
                st.file_uploader(
                    "Subir otro archivo",
                    type=["csv", "xlsx", "xls"],
                    accept_multiple_files=True,
                    key="repair_uploaded_files",
                    help="Elegí otro CSV o Excel con separador correcto.",
                )
            # El próximo run prioriza repair_uploaded_files arriba y descarta el inválido.
            st.stop()

    if len(tables) > 1 and advanced:
        with st.expander("Relaciones entre tablas"):
            graph = build_relationship_graph(tables)
            st.write(graph.get("note"))
            st.dataframe(pd.DataFrame(graph["edges"]) if graph["edges"] else pd.DataFrame(), width="stretch")

    def _on_main_analyze() -> None:
        st.session_state["_run_analyze"] = True

    if show_analyze_prompt:
        st.info("Datos listos. Pulsá **Analizar** para generar el resumen.")
        st.button(
            "Analizar",
            type="primary",
            key="main_analyze_btn",
            help="Corre el análisis automático sobre la planilla cargada.",
            on_click=_on_main_analyze,
        )
        from estacion.ux.presentation import dataframe_for_display

        st.dataframe(dataframe_for_display(df.head(12)), width="stretch")
        return

    if run_analyze_now:
        _run_discovery(df, selected, quality_domain, use_ollama)

    if not _analysis_ready(selected, quality_domain):
        return

    findings = st.session_state["findings"]
    profile = st.session_state["profile"]
    artifacts = st.session_state["artifacts"]
    narratives = {n["id"]: n["narrative"] for n in st.session_state.get("narratives", [])}

    if profile.n_rows < 30 and not bad_parse:
        from estacion.ux.presentation import format_n_rows

        st.warning(
            f"Tu archivo tiene solo **{format_n_rows(profile.n_rows)}**. "
            "Con tan pocas filas es difícil encontrar patrones (recomendado: 30 o más)."
        )

    copy_ctx = _build_copy_context(df, profile, selected)

    if not advanced:
        tabs = st.tabs(["Resumen", "Explorar", "Informe"])
        with tabs[0]:
            from estacion.ux.presentation import EMPTY_FINDINGS_MESSAGE

            empty = not findings
            if empty:
                st.info(EMPTY_FINDINGS_MESSAGE)
            else:
                render_summary_header(findings, profile, artifacts, df, ctx=copy_ctx)
                st.markdown("")
                render_finding_cards(
                    findings, narratives, show_empty=False, ctx=copy_ctx, layout="summary"
                )
        with tabs[1]:
            _render_explore(df, profile)
            with st.expander("Calidad rápida"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Columna": v.name,
                                "Tipo": v.role.value,
                                "% vacíos": round(v.missing_pct * 100, 1),
                                "Únicos": v.n_unique,
                            }
                            for v in profile.variables
                        ]
                    ),
                    width="stretch",
                )
        with tabs[2]:
            _render_informe(df, selected, store)
        return

    # --- Modo avanzado ---
    tabs = st.tabs(
        [
            "Resumen",
            "Perfil",
            "Faltantes",
            "Explorar",
            "Estadística",
            "Segmentos",
            "Causa",
            "Comparar",
            "Informe",
        ]
    )

    with tabs[0]:
        from estacion.ux.presentation import EMPTY_FINDINGS_MESSAGE, format_int_es

        st.caption("Resumen de hallazgos en lenguaje simple, con números que ya calculó el motor.")
        if not findings:
            st.info(EMPTY_FINDINGS_MESSAGE)
        else:
            render_summary_header(findings, profile, artifacts, df, ctx=copy_ctx)
            fdr = artifacts.get("fdr_global") or {}
            n_sig, n_tested = fdr.get("n_significant"), fdr.get("n_tested")
            if n_sig is not None and n_tested is not None:
                st.caption(
                    f"Hallazgos confiables tras corregir por muchas pruebas: "
                    f"{format_int_es(n_sig)} de {format_int_es(n_tested)}."
                )
            render_finding_cards(
                findings, narratives, show_empty=False, ctx=copy_ctx, layout="summary"
            )

    with tabs[1]:
        from estacion.ux.presentation import format_int_es, format_pct
        from estacion.ux.result_views import (
            ROLE_LABELS_ES,
            descriptive_table_es,
            dtype_label_es,
            humanize_granularity_notes,
        )

        st.caption(
            "La ficha de cada columna: qué tipo de dato es, cuántos vacíos tiene y si algo se ve raro. "
            "Ej.: te avisa si 'order_id' tiene repetidos cuando debería ser único."
        )
        st.subheader("Perfil del dataset")
        perfil_df = pd.DataFrame(
            [
                {
                    "Columna": v.name,
                    "Tipo": dtype_label_es(v.dtype),
                    "Rol": ROLE_LABELS_ES.get(v.role.value, v.role.value),
                    "% vacíos": format_pct(v.missing_pct, digits=1),
                    "Únicos": format_int_es(v.n_unique),
                    "Avisos": "; ".join(v.warnings) or "—",
                }
                for v in profile.variables
            ]
        )
        st.dataframe(
            perfil_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Columna": st.column_config.TextColumn("Columna", width="medium"),
                "Tipo": st.column_config.TextColumn("Tipo", width="small"),
                "Rol": st.column_config.TextColumn("Rol", width="small"),
                "% vacíos": st.column_config.TextColumn("% vacíos", width="small"),
                "Únicos": st.column_config.TextColumn("Únicos", width="small"),
                "Avisos": st.column_config.TextColumn("Avisos", width="large"),
            },
        )
        long_avisos = [(r["Columna"], r["Avisos"]) for _, r in perfil_df.iterrows() if r["Avisos"] not in {"—", ""}]
        if long_avisos:
            with st.expander("Avisos completos por columna", expanded=False):
                for col_name, avisos in long_avisos:
                    st.markdown(f"**{col_name}:** {avisos}")
        notes = humanize_granularity_notes(profile.granularity_notes)
        if notes:
            st.markdown("**Granularidad:**")
            for n in notes:
                st.markdown(f"- {n}")
        else:
            st.write("**Granularidad:** —")
        if profile.numeric_columns:
            st.write("**Descriptivos**")
            desc_df = descriptive_table_es(descriptive_report(df, profile.numeric_columns))
            method_cfg = {}
            if "Método" in desc_df.columns:
                method_cfg["Método"] = st.column_config.TextColumn("Método", width="large")
            st.dataframe(
                desc_df,
                width="stretch",
                hide_index=True,
                column_config=method_cfg or None,
            )
            if "Método" in desc_df.columns:
                with st.expander("Métodos de los descriptivos (texto completo)", expanded=False):
                    for _, row in desc_df.iterrows():
                        var = row.get("Variable", row.get("variable", "—"))
                        st.markdown(f"**{var}:** {row.get('Método', '—')}")

        st.subheader("Editor de reglas de calidad")
        base = load_rules(domain=quality_domain)
        c1, c2, c3 = st.columns(3)
        max_miss = c1.slider("Máx. % vacíos", 0.0, 1.0, float(base.max_missing_pct), 0.01)
        max_dup = c2.slider("Máx. % duplicados", 0.0, 0.5, float(base.max_duplicate_pct), 0.001)
        max_out = c3.slider("Máx. % outliers IQR", 0.0, 0.5, float(base.max_iqr_outlier_pct), 0.01)
        nonneg = st.multiselect(
            "Columnas no negativas",
            list(df.columns),
            default=[c for c in base.non_negative_columns if c in df.columns],
            placeholder="Elegí opciones",
        )
        ids_u = st.multiselect(
            "IDs que deben ser únicos",
            list(df.columns),
            default=[c for c in base.id_columns_must_be_unique if c in df.columns],
            placeholder="Elegí opciones",
        )
        sentinels = st.text_input(
            "Valores trampa (coma-separados)",
            value=",".join(map(str, base.forbidden_sentinel_values)),
            help="Ej: -999, NA, n/a — códigos que parecen número pero son faltantes.",
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
            from estacion.ux.result_views import rename_df_columns_es

            st.dataframe(
                rename_df_columns_es(pd.DataFrame(qres["violations"])),
                width="stretch",
                hide_index=True,
            )

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
            st.write(f"**{pr.get('title')}** — hallazgos: {pr.get('n_findings')}")
            st.dataframe(pd.DataFrame(pr.get("findings") or []), width="stretch")

        st.markdown("##### Diccionario de datos")
        dict_dir = ROOT / "config" / "dictionaries"
        dict_files = sorted(
            [p.name for p in dict_dir.glob("*.yaml")]
            + [p.name for p in dict_dir.glob("*.yml")]
            + [p.name for p in dict_dir.glob("*.json")]
        ) if dict_dir.exists() else []
        dict_choice = st.selectbox("Diccionario", ["—"] + dict_files, key="dict_file")
        if dict_choice != "—" and st.button("Validar contra diccionario"):
            dd = load_dictionary(dict_dir / dict_choice)
            st.session_state["dict_val"] = validate_against_dictionary(df, dd)
        if st.button("Inferir stub de diccionario"):
            stub = infer_dictionary_stub(df, name=selected, domain=quality_domain)
            outp = dict_dir / f"{selected}_draft.json"
            save_dictionary(stub, outp)
            st.success(f"Stub guardado en {outp}")
        dv = st.session_state.get("dict_val")
        if dv:
            st.write(f"Diccionario `{dv.get('dictionary')}` v{dv.get('version')}: {dv.get('n_findings')} hallazgos")
            st.dataframe(pd.DataFrame(dv.get("findings") or []), width="stretch")

    with tabs[2]:
        st.caption("¿Para qué sirve? Entender si los vacíos parecen al azar o siguen un patrón.")
        st.subheader("Datos faltantes")
        if st.button("Analizar faltantes", key="miss_btn"):
            st.session_state["missingness"] = analyze_missingness(df)
        miss = st.session_state.get("missingness") or artifacts.get("missingness")
        if miss:
            from estacion.ux.result_views import humanize_missingness

            humanize_missingness(miss)
        else:
            st.info("Pulsá 'Analizar faltantes' para el detalle.")

    with tabs[3]:
        st.caption("¿Para qué sirve? Mirar tus datos con gráficos antes de confiar en un número.")
        _render_explore(df, profile)

    with tabs[4]:
        st.caption("¿Para qué sirve? Comparar grupos o ajustar un modelo simple y dirigido.")
        st.subheader("Estadística dirigida")
        mode = st.radio(
            "Análisis",
            ["Comparación de grupos", "Regresión lineal", "Regresión logística"],
            horizontal=True,
        )
        if mode == "Comparación de grupos":
            g = st.selectbox("Grupo", profile.categorical_columns or list(df.columns), help="Categoría a comparar, ej. región")
            v = st.selectbox("Valor", profile.numeric_columns or list(df.columns), help="Número a medir, ej. revenue")
            if st.button("Ejecutar comparación"):
                res = group_comparison(df, v, g)
                st.session_state["group_cmp"] = res
            res = st.session_state.get("group_cmp")
            if res:
                if res.get("ok"):
                    st.write(
                        f"**Conclusión:** "
                        f"{'Hay diferencias significativas' if res.get('significant') else 'No hay diferencias claras'} "
                        f"entre grupos (p = {format_p_value(res.get('p_value'))})."
                    )
                    effect = res.get("effect_size_cohens_d")
                    eta = res.get("effect_size_eta_sq")
                    if effect is not None:
                        st.write(f"Tamaño del efecto (d de Cohen) ≈ {format_number(effect)}")
                    elif eta is not None:
                        st.write(f"Tamaño del efecto (eta²) ≈ {format_number(eta)}")
                    else:
                        st.write(
                            "Tamaño del efecto: no se pudo calcular "
                            "(la comparación no devolvió d de Cohen ni eta²)."
                        )
                    if res.get("group_means"):
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {"grupo": k, "media": format_number(val)}
                                    for k, val in res["group_means"].items()
                                ]
                            ),
                            width="stretch",
                        )
                        st.plotly_chart(group_means_bar(res["group_means"]), width="stretch")
                        st.plotly_chart(boxplot(df, v, g), width="stretch")
                    with st.expander("Datos crudos"):
                        st.json(res)
                else:
                    st.warning(res.get("reason"))
        elif mode == "Regresión lineal":
            y = st.selectbox("y", profile.numeric_columns or list(df.columns), key="lin_y")
            xs = st.multiselect(
                "x", [c for c in df.columns if c != y], key="lin_x", placeholder="Elegí opciones"
            )
            if xs and st.button("Ajustar OLS"):
                res = linear_regression(df, y, xs)
                st.text(res.get("summary_text") or res)
                st.write(res.get("limitations"))
        else:
            y = st.selectbox("y binaria", profile.possible_targets or list(df.columns), key="log_y")
            xs = st.multiselect(
                "x", [c for c in df.columns if c != y], key="log_x", placeholder="Elegí opciones"
            )
            if xs and st.button("Ajustar Logit"):
                res = logistic_regression(df, y, xs)
                st.text(res.get("summary_text") or res)
                st.write(res.get("limitations"))

    with tabs[5]:
        st.caption("¿Para qué sirve? Buscar 'tipos' de filas parecidas (clientes, pedidos, etc.).")
        st.subheader("Segmentación")
        nums = st.multiselect(
            "Numéricas",
            profile.numeric_columns,
            default=profile.numeric_columns[:5],
            placeholder="Elegí opciones",
        )
        cats = st.multiselect(
            "Categóricas (mixtas)",
            profile.categorical_columns,
            default=[],
            placeholder="Elegí opciones",
        )
        if st.button("Correr segmentación") and len(nums) >= 2:
            with st.spinner("Evaluando agrupamientos…"):
                st.session_state["seg"] = run_segmentation(df, nums, cats)
        seg = st.session_state.get("seg")
        if seg:
            render_segmentation_human(seg, df=df)
            labels = seg.get("kmeans", {}).get("labels")
            coords = seg.get("pca_coords")
            if labels is not None and coords is not None:
                natural = bool(seg.get("natural_groups_likely"))
                if not natural:
                    with st.expander("Ver gráfico (atenuado: sin grupos naturales)"):
                        st.plotly_chart(
                            cluster_scatter(coords, labels, muted=True),
                            width="stretch",
                        )
                else:
                    st.plotly_chart(cluster_scatter(coords, labels), width="stretch")

    with tabs[6]:
        st.caption("¿Para qué sirve? Explorar si una cosa se asocia a otra controlando confusores. No prueba causalidad.")
        st.subheader("¿Qué causa qué? (exploratorio)")
        st.warning("Ningún resultado demuestra causalidad. Son asociaciones bajo supuestos fuertes.")
        causal_mode = st.radio(
            "Modo",
            ["Ajuste / PSM", "DiD multi-período", "Adopción escalonada", "Editor DAG"],
            horizontal=True,
        )
        defaults_c = default_causal_columns(profile, df)
        cols = list(df.columns)
        if causal_mode == "Ajuste / PSM":
            st.caption("¿Para qué sirve? Ej.: ¿el descuento se asocia a más ventas, controlando tamaño de cliente?")
            t_opts = defaults_c["treatment_options"] or cols
            o_opts = defaults_c["outcome_options"] or [c for c in cols if c != (t_opts[0] if t_opts else None)]
            t_idx = t_opts.index(defaults_c["treatment"]) if defaults_c["treatment"] in t_opts else 0
            treatment = st.selectbox(
                "Tratamiento",
                t_opts,
                index=t_idx,
                help="Lo que querés ver si tiene efecto, ej. canal o un sí/no.",
                key="c_t",
            )
            o_opts2 = [c for c in o_opts if c != treatment] or o_opts
            o_idx = o_opts2.index(defaults_c["outcome"]) if defaults_c["outcome"] in o_opts2 else 0
            outcome = st.selectbox(
                "Resultado (numérico)",
                o_opts2,
                index=o_idx,
                help="Debe ser un número (ej. revenue). No uses categorías como región.",
                key="c_o",
            )
            confounders = st.multiselect(
                "Confusores hipotéticos",
                [c for c in cols if c not in {treatment, outcome}],
                default=[c for c in defaults_c["confounders"] if c not in {treatment, outcome}],
                help="Variables que afectan ambos, ej. tamaño de cliente.",
                key="c_c",
                placeholder="Elegí opciones",
            )
            if st.button("Explorar causalmente"):
                if outcome not in profile.numeric_columns and not pd.api.types.is_numeric_dtype(df[outcome]):
                    st.error(
                        f"Para esto el resultado tiene que ser un número; '{outcome}' no lo es. "
                        "Probá con revenue u otra columna numérica."
                    )
                else:
                    st.session_state["causal"] = exploratory_causal_screen(df, treatment, outcome, confounders)
            res = st.session_state.get("causal")
            if res:
                st.caption(res.get("disclaimer") or "")
                from estacion.ux.result_views import humanize_causal_adjusted

                humanize_causal_adjusted(res.get("adjusted_association") or {})
                psm = res.get("propensity_matching")
                if psm and psm.get("ok"):
                    st.write("**Matching por propensity (resumen)**")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"campo": k, "valor": v}
                                for k, v in psm.items()
                                if k not in {"matched_outcomes", "sensitivity"}
                                and not isinstance(v, (dict, list, pd.DataFrame))
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                    with st.expander("Ver datos crudos PSM"):
                        st.json({k: psm[k] for k in psm if k not in {"matched_outcomes"}})
            if st.button("Estimar ATE con DoWhy (opcional)"):
                st.session_state["dowhy"] = estimate_ate_dowhy(df, treatment, outcome, confounders)
            dw = st.session_state.get("dowhy")
            if dw:
                st.write(dw if dw.get("ok") else dw.get("reason"))
                st.caption(f"DoWhy instalado: {dowhy_available()}")
        elif causal_mode == "DiD multi-período":
            st.caption("¿Para qué sirve? Comparar tratados vs controles antes/después de una intervención.")
            outcome = st.selectbox(
                "Resultado",
                profile.numeric_columns or cols,
                key="did_o",
                help="Variable numérica a medir (ej. ventas).",
            )
            gcol = st.selectbox(
                "Grupo tratado/control",
                [c for c in cols if c != outcome],
                key="did_g",
                help="Columna que indica quién recibió el tratamiento.",
            )
            tcol = st.selectbox(
                "Tiempo / período",
                [c for c in cols if c not in {outcome, gcol}],
                key="did_t",
                help="Columna de tiempo o período.",
            )
            levels = sorted(df[gcol].dropna().unique().tolist(), key=lambda x: str(x))
            treated = st.selectbox("Nivel tratado", levels, key="did_tl", help="Valor que marca al grupo tratado.")
            periods = sorted(df[tcol].dropna().unique().tolist(), key=lambda x: str(x))
            post = st.multiselect(
                "Períodos post",
                periods,
                default=periods[-1:] if periods else [],
                help="Períodos después de la intervención.",
                placeholder="Elegí opciones",
            )
            if post and st.button("Correr DiD"):
                st.session_state["did"] = difference_in_differences_multipperiod(
                    df, outcome, gcol, tcol, treated, post
                )
            did = st.session_state.get("did")
            if did:
                if did.get("ok") is False:
                    st.warning(did.get("reason") or "No se pudo estimar DiD.")
                else:
                    st.success("Estimación DiD lista (exploratoria; no prueba causalidad).")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"campo": k, "valor": v}
                                for k, v in did.items()
                                if not isinstance(v, (dict, list, pd.DataFrame))
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                    with st.expander("Ver datos crudos DiD"):
                        st.write(did)
        elif causal_mode == "Adopción escalonada":
            outcome = st.selectbox(
                "Resultado",
                profile.numeric_columns or cols,
                key="cs_o",
                help="Variable numérica del panel.",
            )
            unit = st.selectbox("Unidad", [c for c in cols if c != outcome], key="cs_u", help="Ej. tienda, cliente, planta.")
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
                work["_treat_time"] = prepare_treat_time_from_binary(work, unit, time_c, tbin)
                tt = "_treat_time"
            aipw_covars = st.multiselect(
                "Covariables AIPW (opcional)",
                [c for c in cols if c not in {outcome, unit, time_c, tt}],
                key="cs_aipw_x",
                placeholder="Elegí opciones",
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Estimar ATT escalonado"):
                    st.session_state["stag"] = staggered_att(work, outcome, unit, time_c, tt)
            with b2:
                if st.button("Estimar AIPW agregado"):
                    st.session_state["aipw"] = aipw_staggered_att(
                        work, outcome, unit, time_c, tt, covariate_cols=aipw_covars or None, n_boot=200
                    )
            stag = st.session_state.get("stag")
            if stag:
                st.write(stag.get("disclaimer"))
                m1, m2, m3 = st.columns(3)
                m1.metric("ATT simple", stag.get("overall_att_simple_avg"))
                m2.metric("ATT ponderado", stag.get("overall_att_weighted"))
                dr = stag.get("doubly_robust_lite") or {}
                m3.metric("ATT DR-lite", dr.get("att") if dr.get("ok") else None)
                if stag.get("ok"):
                    st.dataframe(pd.DataFrame(stag["att_by_cohort_time"]), width="stretch")
            aipw = st.session_state.get("aipw")
            if aipw:
                if aipw.get("ok"):
                    st.write("**AIPW**", aipw)
                else:
                    st.warning(aipw.get("reason"))
        else:
            st.caption("Editor DAG: roles + aristas + canvas.")
            nodes = st.multiselect(
                "Nodos",
                cols,
                default=cols[: min(6, len(cols))],
                key="dag_nodes",
                placeholder="Elegí opciones",
            )
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
            if "dag_edges" not in st.session_state:
                st.session_state["dag_edges"] = []
            edge_src = st.selectbox("Arista desde", ["—"] + nodes, key="e_src")
            edge_dst = st.selectbox("Arista hacia", ["—"] + [n for n in nodes if n != edge_src], key="e_dst")
            if st.button("Agregar arista") and edge_src != "—" and edge_dst != "—":
                e = [edge_src, edge_dst]
                if e not in st.session_state["dag_edges"]:
                    st.session_state["dag_edges"].append(e)
            if st.button("Limpiar aristas"):
                st.session_state["dag_edges"] = []
            dag = build_dag(nodes, st.session_state["dag_edges"], roles)
            st.write("**Validación**", validate_dag(dag))
            dag_payload = {"nodes": nodes, "edges": st.session_state["dag_edges"], "roles": roles}
            st.download_button(
                "Descargar DAG JSON",
                data=json.dumps(dag_payload, ensure_ascii=False, indent=2),
                file_name="dag.json",
                mime="application/json",
            )
            if nodes:
                st.plotly_chart(dag_to_plotly(dag), width="stretch")
                render_interactive_dag(nodes, st.session_state["dag_edges"], roles)

    with tabs[7]:
        st.caption("¿Para qué sirve? Ver si dos planillas (ej. enero vs febrero) cambiaron de forma.")
        st.subheader("Comparar dos archivos")
        if len(tables) < 2:
            st.info("Cargá al menos dos archivos para compararlos de verdad.")
            with st.expander("Probar con una demo (partir este archivo al azar)"):
                st.caption(
                    "Parte la tabla actual en dos mitades al azar solo para probar la pantalla. "
                    "No reemplaza comparar dos archivos reales."
                )
                frac = st.slider("Proporción de la primera mitad", 0.1, 0.9, 0.5)
                if st.button("Comparar mitades de demo"):
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
            from estacion.ux.result_views import rename_df_columns_es

            st.write(drift.get("interpretation"))
            st.write(f"Columnas con cambio detectado: {drift.get('n_drifted', 0)}")
            st.dataframe(
                rename_df_columns_es(pd.DataFrame(drift["column_tests"])),
                width="stretch",
                hide_index=True,
            )

    with tabs[8]:
        st.caption("¿Para qué sirve? Bajar el informe y guardar el análisis para más tarde.")
        _render_informe(df, selected, store)
        st.subheader("Comparar dos análisis guardados")
        sessions = store.list_sessions()
        if len(sessions) < 2:
            st.caption("Guardá al menos dos análisis para compararlos.")
        else:
            labels = {f"{s['name']} ({s['id']})": s["id"] for s in sessions}
            a = st.selectbox("Análisis A", list(labels), key="cmp_a")
            b = st.selectbox("Análisis B", [x for x in labels if x != a] or list(labels), key="cmp_b")
            if st.button("Comparar análisis"):
                la = store.load(labels[a])
                lb = store.load(labels[b])
                titles_a = {f["title"] for f in la["findings"]}
                titles_b = {f["title"] for f in lb["findings"]}
                st.write(
                    f"Comunes: {len(titles_a & titles_b)} | "
                    f"Solo A: {len(titles_a - titles_b)} | "
                    f"Solo B: {len(titles_b - titles_a)}"
                )


if __name__ == "__main__":
    main()
