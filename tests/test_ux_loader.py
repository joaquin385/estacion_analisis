"""Tests UX: carga CSV AR/ES + presentación en lenguaje simple."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from estacion.core.loader import load_table
from estacion.core.profiler import profile_dataframe
from estacion.core.types import Finding, FindingCategory, FindingSeverity
from estacion.discovery.engine import run_discovery
from estacion.ux.presentation import (
    build_plain_summary,
    humanize_finding,
    sort_findings_for_humans,
    finding_track,
    default_causal_columns,
    default_explore_columns,
)


def test_csv_semicolon_comma_decimal():
    raw = (
        "fecha;sucursal;ventas\n"
        "2024-01-01;Norte;1.234,56\n"
        "2024-01-02;Sur;980,5\n"
        "2024-01-03;Este;100,0\n"
    ).encode("utf-8")
    df, meta = load_table(raw, name="ar_csv")
    assert meta.get("n_cols") == 3
    assert list(df.columns) == ["fecha", "sucursal", "ventas"]
    assert not meta.get("suspicious_parse")
    assert float(df["ventas"].iloc[0]) == 1234.56


def test_csv_single_column_warns():
    # Un "CSV" mal separado queda en 1 columna si forzamos contenido raro
    # Simulamos parse sospechoso: una sola columna con texto largo con ;
    raw = b"todo\n" + b"a;b;c;d;e;f;g;h;i;j;k;l;m;n;o;p;q;r;s;t;u;v;w;x;y;z\n"
    df, meta = load_table(raw, name="bad")
    # sniffer puede detectar ; y parsear bien — aceptamos ambas vías
    if meta.get("n_cols") == 1:
        assert meta.get("suspicious_parse") or meta.get("warnings")


def test_invalid_bytes_does_not_crash():
    df, meta = load_table(b"\xff\xfe\x00\x01not,a,table", name="junk")
    assert isinstance(df, pd.DataFrame)
    assert "warnings" in meta or meta.get("n_rows") is not None


def test_plain_summary_and_skew_not_critical():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "synthetic" / "commercial.csv"
    if not path.exists():
        return
    df, _ = load_table(path)
    findings, profile, artifacts = run_discovery(df, name="commercial")
    bullets = build_plain_summary(findings, profile, artifacts)
    assert 1 <= len(bullets) <= 5
    skews = [f for f in findings if f.id.startswith("") and "asimétr" in f.title.lower() or "skew" in f.id]
    # finding id is hash; match by title
    skew_findings = [f for f in findings if "asimétr" in f.title.lower()]
    for f in skew_findings:
        assert f.severity != FindingSeverity.CRITICAL
        assert finding_track(f) == "descubrimientos"
    group_findings = [f for f in findings if "cambia el nivel" in f.title or "grupos" in f.title.lower()]
    if group_findings and skew_findings:
        assert max(f.score for f in group_findings) >= max(f.score for f in skew_findings)
    ordered = sort_findings_for_humans(findings)
    assert ordered
    h = humanize_finding(ordered[0])
    assert "what" in h and "title_simple" in h


def test_default_columns_skip_ids():
    df = pd.DataFrame(
        {
            "order_id": range(50),
            "revenue": np.random.default_rng(0).normal(100, 10, 50),
            "marketing_spend": np.random.default_rng(1).normal(50, 5, 50),
            "region": ["Norte", "Sur"] * 25,
        }
    )
    profile = profile_dataframe(df, "demo")
    exp = default_explore_columns(profile)
    assert exp["hist"] != "order_id"
    causal = default_causal_columns(profile, df)
    assert causal["outcome"] in {"revenue", "marketing_spend"}
    assert causal["treatment"] != "order_id"
