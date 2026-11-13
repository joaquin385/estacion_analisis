"""Tests del rediseño Claro SaaS + copy (Direction A + titular C)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from estacion.core.loader import load_table
from estacion.discovery.engine import run_discovery
from estacion.ux.presentation import (
    CopyContext,
    EMPTY_FINDINGS_MESSAGE,
    build_business_kpis,
    build_summary_headline,
    dedupe_findings,
    finding_track,
    format_amount,
    format_bullet_list,
    format_memory_mb,
    friendly_dataset_name,
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


def test_money_unit_from_dictionary_not_hardcoded():
    assert resolve_money_unit(column="revenue", dictionary_units={}) == "unidades"
    assert resolve_money_unit(column="revenue", dictionary_units={"revenue": "$"}) == "$"
    assert resolve_money_unit(override="unidades", dictionary_units={"revenue": "$"}) == "unidades"
    assert format_amount(430, "$", digits=0) == "$430"
    assert "unidades" in format_amount(430, "unidades", digits=0)
    assert format_memory_mb(0.217) == "0,217 MB"
    assert friendly_dataset_name("commercial") == "Ventas"


def test_dictionary_units_commercial():
    units = load_dictionary_units(name_hint="commercial")
    assert units.get("revenue") == "$"


def test_summary_headline_norte(commercial_pack):
    _, findings, profile, artifacts, ctx = commercial_pack
    headline, sub = build_summary_headline(findings, profile, artifacts, ctx)
    assert "Norte" in headline
    assert "mitad" in headline.lower() or "52" in headline
    assert "3.060" in sub
    assert "problema" in sub.lower()


def test_business_kpis(commercial_pack):
    df, findings, profile, _, ctx = commercial_pack
    kpis = build_business_kpis(findings, profile, df, ctx)
    assert len(kpis) == 4
    assert kpis[0]["value"] == "3.060"
    assert kpis[1]["value"].startswith("$")
    assert int(kpis[2]["value"].replace(".", "")) >= 1
    assert int(kpis[3]["value"]) >= 1


def test_finding_copy_uses_engine_numbers(commercial_pack):
    _, findings, _, _, ctx = commercial_pack
    ordered = dedupe_findings(findings)
    region = next(f for f in ordered if f.category.value == "statistical" and "region" in (f.title or ""))
    h = humanize_finding(region, ctx)
    assert "52" in h["title_simple"]
    assert "Norte" in h["title_simple"]
    assert "$430" in h["what"] or "430" in h["what"]
    assert "4 de cada 10" in h["what"] or "mitad" in h["what"]
    assert any("Compará" in a or "chequeá" in a.lower() for a in h["do"])
    assert "Mirá el gráfico" not in " ".join(h["do"])

    rel = next(
        f
        for f in ordered
        if f.category.value == "relationship"
        and "customer_size" in f.variables
        and "revenue" in f.variables
    )
    hr = humanize_finding(rel, ctx)
    assert "272" in hr["what"] and "424" in hr["what"]
    assert "no prueba" in hr["why"].lower() or "ojo" in hr["why"].lower()


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
    assert "2 %" in h["title_simple"] or "2 %" in h["what"]
    # impacto calculado desde df (no inventado)
    assert "20." in h["what"] or "20.000" in h["what"] or "20331" in h["what"].replace(".", "")


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
    import importlib.util

    path = ROOT / "app" / "streamlit_app.py"
    # Importar solo el dict vía exec parcial es frágil; leemos el fuente
    text = path.read_text(encoding="utf-8")
    assert "Causal confounded (prueba)" not in text
    assert "Causal con confusor" in text


def test_report_memory_comma(commercial_pack):
    from estacion.reports.generator import generate_markdown_report

    _, findings, profile, artifacts, _ = commercial_pack
    md = generate_markdown_report(profile, findings, artifacts)
    assert "0,217 MB" in md or ", MB" not in md  # coma decimal si hay fracción
    assert "Ventas" in md.split("\n")[0] or "commercial" not in md.split("\n")[0].lower()
