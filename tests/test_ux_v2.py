"""Tests UX v2: resumen humano, dedupe, jerga, persistencia del demo."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from estacion.core.loader import load_table
from estacion.discovery.engine import run_discovery
from estacion.ux.presentation import (
    JARGON_TERMS,
    SUMMARY_FORBIDDEN,
    build_plain_summary,
    dedupe_findings,
    format_int_es,
    format_magnitude,
    format_number,
    format_n_rows,
    humanize_finding,
    method_plain_es,
    sort_findings_for_humans,
    finding_track,
)
from estacion.ux.result_views import (
    humanize_causal_adjusted,
    _extract_adjusted_coef,
    dict_to_table,
    dtype_label_es,
    reason_plain_es,
    rename_df_columns_es,
)


ROOT = Path(__file__).resolve().parents[1]
COMMERCIAL = ROOT / "data" / "synthetic" / "commercial.csv"


@pytest.fixture(scope="module")
def commercial_pack():
    if not COMMERCIAL.exists():
        pytest.skip("commercial.csv no disponible")
    df, _ = load_table(COMMERCIAL)
    findings, profile, artifacts = run_discovery(df, name="commercial")
    return df, findings, profile, artifacts


def test_plain_summary_no_jargon(commercial_pack):
    _, findings, profile, artifacts = commercial_pack
    bullets = build_plain_summary(findings, profile, artifacts)
    assert 1 <= len(bullets) <= 5
    blob = "\n".join(bullets).lower()
    for tok in SUMMARY_FORBIDDEN:
        assert tok.lower() not in blob, f"resumen contiene '{tok}': {bullets}"
    # señales humanas esperadas en commercial
    assert any("norte" in b.lower() or "customer_size" in b.lower() or "revenue" in b.lower() for b in bullets)
    assert any("repetid" in b.lower() or "duplic" in b.lower() for b in bullets)
    assert any("discount" in b.lower() for b in bullets)


def test_dedupe_duplicate_findings(commercial_pack):
    _, findings, _, _ = commercial_pack
    before = [f for f in findings if "duplic" in (f.title + f.observation + str(f.evidence)).lower() or (f.evidence or {}).get("rule") == "max_duplicate_pct"]
    assert len(before) >= 2
    after = dedupe_findings(findings)
    dup_after = [
        f
        for f in after
        if "duplic" in (f.title + f.observation + str(f.evidence)).lower()
        or (f.evidence or {}).get("rule") == "max_duplicate_pct"
    ]
    assert len(dup_after) == 1


def test_discoveries_sorted_before_problems(commercial_pack):
    _, findings, _, _ = commercial_pack
    ordered = sort_findings_for_humans(dedupe_findings(findings))
    assert ordered
    assert finding_track(ordered[0]) == "descubrimientos"


def test_humanize_cards_without_jargon(commercial_pack):
    _, findings, _, _ = commercial_pack
    for f in dedupe_findings(findings):
        h = humanize_finding(f)
        text = " ".join(
            [
                h["what"],
                h["why"],
                " ".join(h["do"]),
                h["title_simple"],
            ]
        )
        for term in JARGON_TERMS:
            assert term not in text, f"jerga '{term}' en {f.title}: {text}"
        assert "max_duplicate_pct" not in h["title_simple"]
        assert "(default)" not in h["title_simple"]


def test_format_magnitude_no_long_decimals():
    mag = format_magnitude(0.0196078431372549)
    assert "2" in mag
    assert "0196078431372549" not in mag
    assert format_magnitude(None) == "—"
    assert format_number(None) == "—"


def test_format_int_es_ar():
    assert format_int_es(3060) == "3.060"
    assert format_int_es(3) == "3"
    assert format_n_rows(3060) == "3.060 filas"
    assert format_number(3060) == "3.060"
    assert format_number(759773.0) == "759.773"
    assert "," not in format_int_es(3060)  # no miles con coma
    assert " " not in format_number(759773.5)
    assert format_number(759773.5) == "759.773,5"


def test_causal_extract_nested_coef():
    adj = {
        "ok": True,
        "treatment": "region_Norte",
        "adjusted": {"coef": 135.78, "pvalue": 0.001, "model": "ols_adjusted"},
        "crude_coef": 146.9,
    }
    coef, p, reason = _extract_adjusted_coef(adj)
    assert coef == pytest.approx(135.78)
    assert p == pytest.approx(0.001)
    assert reason is None


def test_demo_select_persists_with_apptest():
    """El selectbox de ejemplo debe conservar 'commercial' tras un rerun de control."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    # Botón de la portada
    demo_btns = [
        b
        for b in at.button
        if (b.label or "").strip().lower() in {"ventas", "probar con un ejemplo de ventas"}
        or "ejemplo de ventas" in (b.label or "").lower()
    ]
    assert demo_btns, "no está el botón de ejemplo en la portada"
    demo_btns[0].click().run(timeout=120)
    # Debe haber análisis / no portada
    assert at.session_state.get("demo_dataset") == "commercial"
    assert "findings" in at.session_state
    # Simular cambio de filtro "Mostrar" si existe
    radios = [r for r in at.radio if (r.label or "") == "Mostrar"]
    if radios:
        radios[0].set_value("Problemas de datos").run(timeout=120)
    # Tras el rerun, el demo y el análisis siguen
    assert at.session_state.get("demo_dataset") == "commercial"
    assert "findings" in at.session_state
    # No volvió a la portada: el botón primario de ejemplo no debería ser lo único visible
    # (puede seguir existiendo en sidebar? no — solo en welcome). Si volvió, el botón reaparece
    # como único CTA; chequeamos que hay métricas o tabs.
    assert len(at.session_state.get("findings") or []) > 0


def test_group_gap_matches_summary_and_report(commercial_pack):
    from estacion.ux.presentation import (
        group_vs_rest_gap,
        format_magnitude,
        plain_finding_line,
        build_plain_summary,
    )
    from estacion.reports.generator import generate_markdown_report

    _, findings, profile, artifacts = commercial_pack
    region = next(f for f in findings if "region" in f.title and "revenue" in f.title)
    gap = group_vs_rest_gap(region)
    assert gap is not None
    assert abs(gap["pct"] - 52) < 1.5
    assert "~52" in format_magnitude(region.magnitude, finding=region)
    summary = "\n".join(build_plain_summary(findings, profile, artifacts))
    assert "~52" in summary
    md = generate_markdown_report(profile, findings, artifacts)
    assert "brecha relativa ≈ 48%" not in md
    assert "0.447522308093447" not in md
    assert "|ρ|" not in md
    assert plain_finding_line(region)


def test_non_significant_relationship_wording(commercial_pack):
    from estacion.ux.presentation import humanize_finding, relationship_confirmed

    _, findings, _, _ = commercial_pack
    weak = [
        f
        for f in findings
        if f.category.value == "relationship" and not relationship_confirmed(f)
    ]
    assert weak, "commercial debería tener al menos una relación no confirmada"
    for f in weak:
        h = humanize_finding(f)
        assert "no confirmada" in h["title_simple"].lower() or "posible" in h["title_simple"].lower()
        assert "se mueven juntos" not in h["title_simple"]
        assert h["label"] == "Info"


def test_format_p_clause_and_rows():
    from estacion.ux.presentation import format_p_clause, format_n_rows

    assert format_p_clause(1e-9) == "p < 0,001"
    assert "=" not in format_p_clause(1e-9).split("p")[1] or format_p_clause(1e-9) == "p < 0,001"
    assert format_p_clause(0.12) == "p = 0,120"
    assert format_n_rows(1) == "1 fila"
    assert "filas" in format_n_rows(3)


def test_report_category_severity_and_bools_es(commercial_pack):
    from estacion.reports.generator import generate_markdown_report

    _, findings, profile, artifacts = commercial_pack
    md = generate_markdown_report(profile, findings, artifacts)
    assert "relationship / critical" not in md
    assert "Relación / Crítico" in md
    assert "Estadísticamente significativo:** True" not in md
    assert "Estadísticamente significativo:** Sí" in md


def test_dataframe_none_to_dash():
    import pandas as pd
    from estacion.ux.presentation import dataframe_for_display

    df = pd.DataFrame({"discount": [0.1, None, float("nan")], "x": [1, 2, 3]})
    out = dataframe_for_display(df)
    assert list(out["discount"]) == ["0,1", "—", "—"]


def test_invalid_then_valid_repair_apptest():
    """Archivo inválido → Subir otro (válido) debe cargar la tabla sin error vacío."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    bad = (
        b"columna_unica\n"
        + b"\n".join(f"fila{i};con;puntos;y;coma".encode() for i in range(8))
    )
    good = (ROOT / "data" / "synthetic" / "commercial.csv").read_bytes()
    # CSV español ; / decimal ,
    ar = (
        "fecha;sucursal;ventas\n"
        "2024-01-01;Norte;100,5\n"
        "2024-01-02;Sur;200,0\n"
        "2024-01-03;Este;150,25\n"
    ).encode("utf-8")

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    welcome = next(u for u in at.file_uploader if "welcome" in (u.key or ""))
    welcome.set_value(("bad.csv", bad, "text/csv")).run(timeout=120)
    metas = at.session_state.get("load_metas") or {}
    assert any(m.get("suspicious_parse") for m in metas.values())

    repair = next(u for u in at.file_uploader if "otro" in (u.label or "").lower())
    repair.set_value(("commercial.csv", good, "text/csv")).run(timeout=180)
    tables = at.session_state.get("loaded_tables") or {}
    assert tables
    df = next(iter(tables.values()))
    assert df.shape[1] >= 2
    assert df.shape[0] > 10
    metas2 = at.session_state.get("load_metas") or {}
    assert not any(m.get("suspicious_parse") and m.get("n_cols") == 1 for m in metas2.values())
    # No quedó un dataframe vacío por read() sin seek
    assert df.shape[0] > 0

    # También con CSV argentino vía repair (reinicio del flujo)
    at2 = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at2.run()
    welcome2 = next(u for u in at2.file_uploader if "welcome" in (u.key or ""))
    welcome2.set_value(("bad.csv", bad, "text/csv")).run(timeout=120)
    repair2 = next(u for u in at2.file_uploader if "otro" in (u.label or "").lower())
    repair2.set_value(("ar.csv", ar, "text/csv")).run(timeout=120)
    df2 = next(iter((at2.session_state.get("loaded_tables") or {}).values()))
    assert list(df2.columns) == ["fecha", "sucursal", "ventas"]
    assert float(df2["ventas"].iloc[0]) == 100.5


def test_loader_single_format_note():
    raw = (
        "fecha;sucursal;ventas\n"
        "2024-01-01;Norte;1.234,56\n"
        "2024-01-02;Sur;980,5\n"
    ).encode("utf-8")
    _, meta = load_table(raw, name="ar")
    notes = [w for w in (meta.get("warnings") or []) if "separador" in w.lower() or "decimal" in w.lower()]
    assert len(notes) == 1


def test_dtype_and_reason_and_campo_valor():
    assert dtype_label_es("int64") == "entero"
    assert dtype_label_es("float64") == "decimal"
    assert dtype_label_es("object") == "texto"
    assert "faltantes en parte" in reason_plain_es("Se necesitan ≥2 columnas numéricas con missing parcial")
    tbl = dict_to_table({"reason": "missing parcial", "n_observed": 3060})
    assert list(tbl.columns) == ["Campo", "Valor"]
    assert "Motivo" in set(tbl["Campo"])
    assert "3.060" in set(map(str, tbl["Valor"])) or any("3.060" in str(v) for v in tbl["Valor"])


def test_effect_not_forced_percent():
    df = pd.DataFrame(
        {
            "missing_col": ["a"],
            "predictor": ["b"],
            "method": ["mannwhitney_missing_vs_observed"],
            "effect": [0.35],
            "missing_rate_target": [0.2],
        }
    )
    out = rename_df_columns_es(df)
    # 0.35 es diff de medias, no 35 %
    effect_col = [c for c in out.columns if "Efecto" in str(c)][0]
    assert "35" not in str(out[effect_col].iloc[0]) or "%" not in str(out[effect_col].iloc[0])
    rate_col = [c for c in out.columns if "% vacíos" in str(c)][0]
    assert "%" in str(out[rate_col].iloc[0])


def test_method_plain_es_and_report(commercial_pack):
    assert "Spearman" in method_plain_es("Spearman + FDR-BH")
    assert "corrección por muchas pruebas" in method_plain_es("Spearman + FDR-BH").lower() or (
        "Correlación por rangos" in method_plain_es("Spearman + FDR-BH")
    )
    assert "FDR-BH" in method_plain_es("Spearman + FDR-BH")  # técnico entre paréntesis
    plain = method_plain_es("Spearman + FDR-BH")
    assert plain.startswith("Correlación")
    assert "residuales" not in method_plain_es("ajuste con residuales y covariables").lower()

    from estacion.reports.generator import generate_markdown_report

    _, findings, profile, artifacts = commercial_pack
    md = generate_markdown_report(profile, findings, artifacts)
    assert "3.060" in md or format_int_es(profile.n_rows) in md
    assert "**Método:** Spearman + FDR-BH" not in md
    assert "Correlación por rangos" in md
