"""Tests del rediseño Claro SaaS + copy (Direction A + titular C) y follow-ups PR#2."""

from __future__ import annotations

from pathlib import Path

import pytest

from estacion.core.loader import load_table
from estacion.discovery.engine import run_discovery
from estacion.ux.presentation import (
    CopyContext,
    EMPTY_FINDINGS_MESSAGE,
    build_business_kpis,
    build_summary_headline,
    counted_findings_n,
    dedupe_findings,
    discoveries_for_summary,
    finding_track,
    format_amount,
    format_bullet_list,
    format_memory_mb,
    format_pct,
    friendly_dataset_name,
    group_vs_rest_gap,
    humanize_finding,
    load_dictionary_units,
    resolve_money_unit,
    suggested_next_steps,
)


ROOT = Path(__file__).resolve().parents[1]
COMMERCIAL = ROOT / "data" / "synthetic" / "commercial.csv"


@pytest.fixture(scope="module")
def commercial_pack():
    if not COMMERCIAL.exists():
        pytest.skip("commercial.csv no disponible")
    df, _ = load_table(COMMERCIAL)
    findings, profile, artifacts = run_discovery(df, name="commercial")
    units = load_dictionary_units(name_hint="commercial")
    money = resolve_money_unit(column="revenue", dictionary_units=units)
    ctx = CopyContext(df=df, profile=profile, money_unit=money, dataset_label="Ventas")
    return df, findings, profile, artifacts, ctx


def test_money_unit_override_wins_over_dictionary():
    assert resolve_money_unit(override="unidades", dictionary_units={"revenue": "$"}) == "unidades"
    assert resolve_money_unit(override="$", dictionary_units={}) == "$"
    assert resolve_money_unit(column="revenue", dictionary_units={"revenue": "$"}) == "$"
    assert resolve_money_unit(column="revenue", dictionary_units={}) == "unidades"
    assert resolve_money_unit(dictionary_units={"revenue": "$"}) == "$"


def test_dictionary_not_from_venta_substring():
    assert load_dictionary_units(name_hint="mi_venta_2026") == {}
    assert load_dictionary_units(name_hint="preventa_q1") == {}
    assert load_dictionary_units(name_hint="commercial").get("revenue") == "$"
    assert load_dictionary_units(name_hint="ventas").get("revenue") == "$"


def test_format_amount_no_money_decimals():
    assert format_amount(20330.82, "$", digits=0) == "$20.331"
    assert format_amount(20330.82, "unidades", digits=0) == "20.331 unidades"
    assert "," not in format_amount(20330.82, "$", digits=0)
    assert format_memory_mb(0.217) == "0,217 MB"
    assert format_pct(0.0196, digits=1) == "2,0 %"
    assert friendly_dataset_name("commercial") == "Ventas"


def test_explicit_unidades_in_card_and_detail(commercial_pack):
    df, findings, profile, _, _ = commercial_pack
    ctx = CopyContext(df=df, profile=profile, money_unit="unidades", dataset_label="Ventas")
    region = next(f for f in dedupe_findings(findings) if "region" in (f.title or ""))
    h = humanize_finding(region, ctx)
    assert "$" not in h["what"]
    assert "unidades" in h["what"]
    gap = group_vs_rest_gap(region)
    assert gap is not None
    detail = format_amount(gap["top_val"], ctx.money_unit, digits=0)
    assert "unidades" in detail
    assert "$" not in detail


def test_summary_headline_norte(commercial_pack):
    _, findings, profile, artifacts, ctx = commercial_pack
    headline, sub = build_summary_headline(findings, profile, artifacts, ctx)
    assert "Norte" in headline
    assert "mitad" in headline.lower() or "52" in headline
    assert "3.060" in sub
    assert "problema" in sub.lower()


def test_norte_finding_first(commercial_pack):
    _, findings, _, _, _ = commercial_pack
    disc = discoveries_for_summary(findings)
    assert disc
    assert "region" in (disc[0].title or "").lower()
    assert "Norte" in humanize_finding(disc[0])["title_simple"]


def test_business_kpis(commercial_pack):
    df, findings, profile, _, ctx = commercial_pack
    kpis = build_business_kpis(findings, profile, df, ctx)
    assert len(kpis) == 4
    assert kpis[0]["value"] == "3.060"
    assert kpis[1]["value"].startswith("$")
    assert "11" in kpis[2]["delta"]


def test_finding_copy_uses_engine_numbers(commercial_pack):
    _, findings, _, _, ctx = commercial_pack
    ordered = dedupe_findings(findings)
    region = next(f for f in ordered if f.category.value == "statistical" and "region" in (f.title or ""))
    h = humanize_finding(region, ctx)
    assert "52" in h["title_simple"]
    assert "Norte" in h["title_simple"]
    assert "pedido promedio del Norte" in h["what"]
    assert "de Norte" not in h["what"]
    assert "'" not in h["what"]
    assert "concentra" in h["what"]
    assert "4 de cada 10 pedidos" in h["what"]
    assert "regiones" in h["why"].lower()
    assert "regiónes" not in h["why"]
    assert any("Compará" in a or "chequeá" in a.lower() for a in h["do"])

    online = next(f for f in ordered if f.category.value == "statistical" and "channel" in (f.title or ""))
    ho = humanize_finding(online, ctx)
    assert "19" in ho["title_simple"]
    assert "canal" in ho["why"].lower() or "margen" in ho["why"].lower()
    assert "regiones" not in ho["why"].lower()
    assert "canals" not in ho["why"]
    assert "canales" in ho["why"]
    assert "Online" in ho["what"]
    assert any("canal" in a.lower() or "envío" in a.lower() or "descuento" in a.lower() for a in ho["do"])
    assert "más de la mitad" in ho["what"]

    rel = next(
        f
        for f in ordered
        if f.category.value == "relationship"
        and "customer_size" in f.variables
        and "revenue" in f.variables
    )
    hr = humanize_finding(rel, ctx)
    assert "272" in hr["what"] and "424" in hr["what"]
    assert "ticket alto entre" not in hr["why"].lower()
    assert "no prueba" in hr["why"].lower() or "ojo" in hr["why"].lower()

    anom = next(f for f in ordered if f.category.value == "anomaly")
    ha = humanize_finding(anom, ctx)
    assert "49" in ha["title_simple"]
    assert "49" in ha["what"]


def test_weighted_rest_average(commercial_pack):
    _, findings, _, _, _ = commercial_pack
    online = next(f for f in dedupe_findings(findings) if "channel" in (f.title or ""))
    gap = group_vs_rest_gap(online)
    assert gap is not None
    assert abs(gap["rest_avg"] - 311) < 2
    assert abs(gap["pct"] - 19) < 1.5


def test_duplicate_copy_and_merged_evidence(commercial_pack):
    _, findings, _, _, ctx = commercial_pack
    dups = [
        f
        for f in dedupe_findings(findings)
        if finding_track(f) == "problemas"
        and (
            "duplic" in (f.title or "").lower()
            or (f.evidence or {}).get("rule") == "max_duplicate_pct"
            or "duplicate" in str(f.evidence)
        )
    ]
    assert len(dups) == 1
    h = humanize_finding(dups[0], ctx)
    assert "60" in h["title_simple"] or "60" in h["what"]
    assert "$20.331" in h["what"] or "20.331" in h["what"]


def test_limitations_empty_not_brackets():
    assert format_bullet_list([]) == []
    assert format_bullet_list(None) == []
    assert format_bullet_list(["A", "", None]) == ["A"]


def test_empty_message_and_next_steps(commercial_pack):
    assert "también es información" in EMPTY_FINDINGS_MESSAGE.lower()
    _, findings, _, _, ctx = commercial_pack
    steps = suggested_next_steps(findings, ctx)
    assert 1 <= len(steps) <= 3
    assert any("repetid" in s.lower() or "Norte" in s or "PDF" in s for s in steps)


def test_demo_labels_spanish():
    text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "Causal confounded (prueba)" not in text
    assert "Causal con confusor" in text


def test_report_new_copy_and_count(commercial_pack):
    from estacion.reports.generator import generate_markdown_report
    from estacion.ux.presentation import build_plain_summary, plain_finding_line, suggested_next_steps

    _, findings, profile, artifacts, ctx = commercial_pack
    md = generate_markdown_report(profile, findings, artifacts, ctx=ctx)
    assert md.startswith("# Informe de análisis — Ventas")
    assert "Norte" in md.split("\n")[6] or "el Norte" in md
    assert "0,217 MB" in md
    assert "11 hallazgos" in md or "**11**" in md
    assert "12 hallazgos" not in md
    assert "Matriz Spearman" not in md
    assert "2,0 %" in md
    assert "2.0%" not in md
    assert "covariables" not in md.lower()
    assert "residuales" not in md.lower()
    assert "outliers o ties" not in md.lower()
    assert "Outliers" not in md
    assert "FDR controla" not in md
    assert "Relación / Crítico" not in md
    assert "customer_size" not in md.split("## Hallazgos")[0]
    assert "marketing_spend" not in "\n".join(md.splitlines()[:40])
    for b in build_plain_summary(findings, profile, artifacts, ctx=ctx):
        assert "customer_size" not in b
        assert "marketing_spend" not in b
        assert "'revenue'" not in b
        assert "discount" not in b
    for f in dedupe_findings(findings)[:5]:
        line = plain_finding_line(f, ctx)
        assert "customer_size" not in line
    for s in suggested_next_steps(findings, ctx):
        assert "'Norte'" not in s
        assert "Norte" in s or "repetid" in s.lower() or "ticket" in s.lower() or "PDF" in s


def test_chart_money_labels_and_axis(commercial_pack):
    from estacion.viz.charts import group_means_bar, relationship_means_line

    _, findings, _, _, _ = commercial_pack
    region = next(f for f in dedupe_findings(findings) if "region" in (f.title or ""))
    means = region.evidence["group_means"]
    fig = group_means_bar(means, highlight="Norte", value_prefix="$")
    texts = list(fig.data[0].text)
    assert all("," not in t.split("$")[-1] for t in texts)  # sin centavos
    assert any(t == "$430" or t.endswith("430") for t in texts)
    y_top = max(fig.layout.yaxis.range or [0, 0])
    assert y_top >= 430
    assert "Norte" in list(fig.data[0].x)

    binned = {"1": 272.0, "2": 308.0, "3": 352.0, "4": 361.0, "5": 424.0}
    line = relationship_means_line(binned, value_prefix="$")
    assert list(line.data[0].x)[0] == "1"
    assert line.layout.xaxis.type == "category"
    assert all("," not in t.split("$")[-1] for t in line.data[0].text)


def test_theme_inter_configured():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'font = "Inter"' in cfg
    assert "fontFaces" in cfg
    css = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "font-family: Inter" in css
    assert "stIconMaterial" in css
    assert "Material Symbols" in css
    assert "stHeader" in css
    assert "Descargar PDF" in css or "Descargar PDF" in (ROOT / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
