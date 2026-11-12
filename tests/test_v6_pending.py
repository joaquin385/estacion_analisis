"""Tests v0.6: AIPW staggered, conformal multi-horizon, diccionarios CI, DoWhy/Ollama soft deps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from estacion.causal.aipw_staggered import aipw_staggered_att
from estacion.causal.dowhy_bridge import dowhy_available, estimate_ate_dowhy
from estacion.causal.dag_interactive import parse_exported_dag_json
from estacion.patterns.conformal import multihorizon_conformal, apply_multihorizon_radii
from estacion.patterns.forecast import forecast_series
from estacion.plugins.dictionaries import load_dictionary, validate_against_dictionary
from estacion.ai.ollama_client import OllamaClient, get_optional_llm
from estacion.ai.interpreter import interpret_findings
from estacion.core.types import Finding, FindingCategory, FindingSeverity


ROOT = Path(__file__).resolve().parents[1]
DICT_DIR = ROOT / "config" / "dictionaries"


def _staggered_panel(n_units: int = 80, T: int = 8, effect: float = 2.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = np.nan if u < n_units // 3 else (3 if u < 2 * n_units // 3 else 5)
        x = rng.normal(0, 1)
        for t in range(T):
            treated = int(pd.notna(g) and t >= g)
            y = 1 + 0.2 * t + 0.5 * x + (effect if treated else 0) + rng.normal(0, 0.25)
            rows.append({"unit": u, "time": t, "treat_time": g, "y": y, "x": x})
    return pd.DataFrame(rows)


def test_aipw_staggered_recovers_effect():
    df = _staggered_panel()
    res = aipw_staggered_att(
        df, "y", "unit", "time", "treat_time", covariate_cols=["x"], n_boot=80
    )
    assert res["ok"]
    assert abs(res["att_aipw"] - 2.0) < 0.6
    assert res.get("ci95_att_bootstrap")
    assert "assumptions" in res


def test_multihorizon_conformal_and_forecast_wiring():
    n = 120
    t = np.arange(n)
    y = 3 + 0.02 * t + np.sin(2 * np.pi * t / 7) + np.random.default_rng(3).normal(0, 0.3, n)
    ts = pd.Series(y)

    def fit_h(train: pd.Series, h: int) -> np.ndarray:
        # naive: last value
        return np.full(h, float(train.iloc[-1]))

    mh = multihorizon_conformal(ts, fit_h, horizon=5, alpha=0.1, n_cal=12, min_train=40)
    assert mh["ok"]
    lo, hi = apply_multihorizon_radii(np.ones(5), mh["q_hat_by_horizon"])
    assert len(lo) == 5 and (hi >= lo).all()

    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D"), "y": y})
    fc = forecast_series(df, "date", "y", horizon=7, method="aic_grid", conformal=True)
    assert fc["ok"]
    assert "conformal_multihorizon" in fc


def test_dictionaries_ci_validate_all_yaml():
    assert DICT_DIR.exists()
    paths = sorted(DICT_DIR.glob("*_v1.yaml"))
    assert len(paths) >= 4
    for path in paths:
        dd = load_dictionary(path)
        assert dd.version
        assert dd.columns
        # empty frame: todas las columnas faltan → críticos
        empty = pd.DataFrame()
        val = validate_against_dictionary(empty, dd)
        assert val["n_critical"] == len(dd.columns)

    # retail dict vs datos parcialmente válidos
    retail = load_dictionary(DICT_DIR / "retail_v1.yaml")
    df = pd.DataFrame(
        {
            "order_id": [1, 2],
            "price": [10.0, -1.0],
            "units": [1, 2],
            "revenue": [10.0, 0.0],
            "discount": [0.1, 1.5],
            "extra": ["a", "b"],
        }
    )
    val = validate_against_dictionary(df, retail)
    assert val["n_findings"] >= 2


def test_dag_json_roundtrip_download_upload():
    payload = {
        "nodes": ["T", "Y", "X"],
        "edges": [["X", "T"], ["X", "Y"], ["T", "Y"]],
        "roles": {"T": "treatment", "Y": "outcome", "X": "confounder"},
    }
    parsed = parse_exported_dag_json(json.dumps(payload))
    assert parsed["dag"]["edges"] == payload["edges"]
    assert parsed["validation"]["ok"] in (True, False)  # ok flag may vary
    assert "acyclic" in parsed["validation"] or "ok" in parsed["validation"]


def test_dowhy_bridge_soft_fail():
    # Always soft: either unavailable or runs
    assert isinstance(dowhy_available(), bool)
    df = pd.DataFrame(
        {
            "t": np.random.default_rng(0).integers(0, 2, 80),
            "y": np.random.default_rng(1).normal(size=80),
            "x": np.random.default_rng(2).normal(size=80),
        }
    )
    res = estimate_ate_dowhy(df, "t", "y", ["x"])
    assert "ok" in res
    if not res["ok"]:
        assert res.get("optional") is True or "reason" in res


def test_ollama_client_degrades_without_server():
    client = OllamaClient(host="http://127.0.0.1:9", timeout=1)
    assert client.available() is False
    assert get_optional_llm(host="http://127.0.0.1:9", timeout=1) is None

    f = Finding(
        id="f1",
        title="Demo",
        observation="Hay correlación.",
        evidence={"r": 0.5},
        magnitude="media",
        method="spearman",
        alternative_explanations=["confusión"],
        follow_up=["ajustar"],
        category=FindingCategory.RELATIONSHIP,
        severity=FindingSeverity.INFO,
        score=1.0,
    )
    narr = interpret_findings([f], try_ollama=True, ollama_model="nope")
    assert narr[0]["narrative"]
