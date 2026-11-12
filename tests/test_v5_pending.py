"""Tests v0.5: ATT ponderado, PELT varianza, forecast AIC/conformal, plugins, diccionarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from estacion.causal.staggered_did import staggered_att
from estacion.patterns.changepoint import pelt_variance, detect_changepoints
from estacion.patterns.forecast import forecast_series
from estacion.plugins import list_plugins, run_plugin_checks
from estacion.plugins.dictionaries import (
    infer_dictionary_stub,
    validate_against_dictionary,
    load_dictionary,
    DataDictionary,
    ColumnSpec,
)


def test_staggered_weighted_and_dr_lite():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(90):
        g = np.nan if u < 30 else (3 if u < 60 else 5)
        for t in range(8):
            treated = int(pd.notna(g) and t >= g)
            y = 2 + 0.1 * t + (1.5 if treated else 0) + rng.normal(0, 0.2)
            rows.append({"unit": u, "time": t, "treat_time": g, "y": y})
    df = pd.DataFrame(rows)
    res = staggered_att(df, "y", "unit", "time", "treat_time")
    assert res["ok"]
    assert abs(res["overall_att_weighted"] - 1.5) < 0.35
    assert abs(res["overall_att_simple_avg"] - 1.5) < 0.35
    assert res.get("att_by_event_time_weighted")
    assert res.get("doubly_robust_lite", {}).get("ok")


def test_pelt_variance_detects_regime():
    rng = np.random.default_rng(1)
    y = np.concatenate(
        [
            rng.normal(0, 0.5, 100),
            rng.normal(0, 3.0, 100),
            rng.normal(0, 0.5, 100),
        ]
    )
    cps = pelt_variance(y, min_size=20)
    assert len(cps) >= 1
    assert any(abs(c - 100) <= 25 or abs(c - 200) <= 25 for c in cps)
    both = detect_changepoints(y, target="both", min_size=15)
    assert both["ok"]
    assert both["n_changepoints_variance"] >= 1


def test_forecast_aic_grid_and_conformal():
    n = 150
    t = np.arange(n)
    y = 5 + 0.03 * t + 1.5 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(2).normal(0, 0.4, n)
    df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="D"), "y": y})
    res = forecast_series(df, "date", "y", horizon=7, method="aic_grid", conformal=True)
    assert res["ok"]
    assert res.get("aic_selection")
    assert len(res["forecast"]) == 7
    assert res.get("conformal", {}).get("ok")


def test_new_plugins_and_dictionary(tmp_path: Path):
    names = {p["name"] for p in list_plugins()}
    assert {"retail", "manufacturing", "healthcare", "finance"} <= names

    retail_df = pd.DataFrame(
        {
            "price": [10, 12, -3, 11],
            "units": [1, 2, 0, 3],
            "revenue": [10, 24, 5, 33],
        }
    )
    rr = run_plugin_checks("retail", retail_df)
    assert rr["ok"] and rr["n_critical"] >= 1

    mfg = pd.DataFrame(
        {
            "temperature": [70, 71, 9999, 68],
            "cycle_time": [12, 11, -1, 13],
            "defective": [0, 1, 0, 0],
        }
    )
    rm = run_plugin_checks("manufacturing", mfg)
    assert rm["ok"] and rm["n_critical"] >= 1

    dd = DataDictionary(
        name="demo",
        version="1.0.0",
        columns=[
            ColumnSpec(name="price", role="numeric", min=0),
            ColumnSpec(name="units", role="numeric", min=1),
        ],
    )
    val = validate_against_dictionary(retail_df, dd)
    assert val["n_findings"] >= 1

    stub = infer_dictionary_stub(retail_df, name="stub")
    assert stub.version.startswith("0.1")
    # example commercial dict if present
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "dictionaries" / "commercial_v1.yaml"
    if path.exists():
        loaded = load_dictionary(path)
        assert loaded.version == "1.0.0"
