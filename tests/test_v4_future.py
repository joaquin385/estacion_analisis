"""Tests v0.4: staggered DiD, forecast, DAG editor, domain plugins."""

from __future__ import annotations

import numpy as np
import pandas as pd

from estacion.causal.staggered_did import staggered_att, prepare_treat_time_from_binary
from estacion.causal.dag_editor import build_dag, validate_dag, suggest_edges_from_roles
from estacion.patterns.forecast import forecast_series
from estacion.plugins import list_plugins, run_plugin_checks


def _staggered_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # units 0-29 never treated; 30-59 treated from t=3; 60-89 treated from t=5
    for u in range(90):
        if u < 30:
            g = np.nan
        elif u < 60:
            g = 3
        else:
            g = 5
        for t in range(8):
            treated = int(pd.notna(g) and t >= g)
            y = 2 + 0.1 * t + (1.5 if treated else 0) + rng.normal(0, 0.25)
            rows.append({"unit": u, "time": t, "treat_time": g, "y": y, "treat": treated})
    return pd.DataFrame(rows)


def test_staggered_att_recovers_effect():
    df = _staggered_panel()
    res = staggered_att(df, "y", "unit", "time", "treat_time")
    assert res["ok"]
    assert res["control_mode"] == "never_treated"
    assert abs(res["overall_att_simple_avg"] - 1.5) < 0.35
    assert res["pretrend_check"] is not None
    assert res["pretrend_check"]["reject_zero_alpha05"] is False


def test_prepare_treat_time_from_binary():
    df = _staggered_panel()
    tt = prepare_treat_time_from_binary(df, "unit", "time", "treat")
    # unit 0 never
    assert pd.isna(tt[df["unit"] == 0].iloc[0])
    # unit 30 first treat at 3
    assert tt[df["unit"] == 30].iloc[0] == 3


def test_forecast_ets_smoke():
    n = 120
    t = np.arange(n)
    y = 10 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(1).normal(0, 0.3, n)
    df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="D"), "y": y})
    res = forecast_series(df, "date", "y", horizon=7, method="ets")
    assert res["ok"]
    assert len(res["forecast"]) == 7
    assert res["backtest"] is None or "mae" in res["backtest"] or res["backtest"].get("ok") is False


def test_dag_validation_and_cycle():
    roles = {"T": "treatment", "Y": "outcome", "X": "confounder", "M": "mediator"}
    edges = suggest_edges_from_roles(roles)
    dag = build_dag(list(roles), edges, roles)
    val = validate_dag(dag)
    assert val["ok"]
    assert val["is_dag"]
    assert "X" in (val["adjustment_suggestion"]["adjust_for"] or [])
    bad = build_dag(["A", "B"], [["A", "B"], ["B", "A"]], {"A": "other", "B": "other"})
    val_bad = validate_dag(bad)
    assert val_bad["ok"] is False


def test_plugins_healthcare_finance():
    assert {p["name"] for p in list_plugins()} >= {"healthcare", "finance"}
    hc = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "age": [40, 150, -2, 33],
            "sbp": [120, 80, 300, 110],
            "dbp": [80, 90, 70, 70],
        }
    )
    r = run_plugin_checks("healthcare", hc)
    assert r["ok"]
    assert r["n_critical"] >= 1
    assert any("Edad" in (f.get("detail") or "") for f in r["findings"])

    fin = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "price": [10, 11, 12, -1, 13, 14, 15, 16, 17, 18],
            "volume": [100, 110, 90, 80, -5, 70, 60, 50, 40, 30],
        }
    )
    rf = run_plugin_checks("finance", fin)
    assert rf["ok"]
    assert rf["n_critical"] >= 1
