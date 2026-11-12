"""Tests v0.3: STL, PELT, Gower, k-prototypes, DiD pretrends, multitable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from estacion.patterns.changepoint import pelt_l2, detect_changepoints
from estacion.patterns.temporal import temporal_analysis
from estacion.clustering.gower import gower_matrix, cluster_gower, kprototypes
from estacion.causal.did import (
    difference_in_differences_multipperiod,
    parallel_trends_test,
)
from estacion.causal.explorer import exploratory_causal_screen
from estacion.core.multitable import build_relationship_graph


def test_pelt_finds_known_jumps():
    y = np.concatenate(
        [
            np.random.default_rng(0).normal(0, 0.3, 80),
            np.random.default_rng(1).normal(5, 0.3, 80),
            np.random.default_rng(2).normal(1, 0.3, 80),
        ]
    )
    cps = pelt_l2(y, min_size=10)
    assert len(cps) >= 1
    # al menos un corte cerca de 80 o 160
    assert any(abs(c - 80) <= 15 or abs(c - 160) <= 15 for c in cps)
    det = detect_changepoints(y)
    assert det["ok"]
    assert det["n_changepoints"] >= 1


def test_stl_detects_weekly_seasonality():
    n = 210
    t = np.arange(n)
    y = 10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(0).normal(0, 0.2, n)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "y": y,
        }
    )
    res = temporal_analysis(df, "date", "y", freq="D", stl_period=7)
    assert res["ok"]
    assert res["stl"]["ok"]
    assert res["stl"]["detected_seasonality"] or res["stl"]["seasonal_strength"] > 0.2
    assert res["changepoints_pelt"]["ok"]


def test_gower_and_kprototypes_on_segments():
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [5, 5], [-4, 6]])
    labels = rng.integers(0, 3, 300)
    X = centers[labels] + rng.normal(0, 0.5, size=(300, 2))
    df = pd.DataFrame(
        {
            "f1": X[:, 0],
            "f2": X[:, 1],
            "cat": np.array(["a", "b", "c"])[labels],
        }
    )
    D = gower_matrix(df, ["f1", "f2"], ["cat"])
    assert D.shape == (300, 300)
    assert np.allclose(np.diag(D), 0)
    g = cluster_gower(df, ["f1", "f2"], ["cat"], n_clusters=3, max_rows=300)
    assert g["ok"]
    assert g["n_clusters"] == 3
    kp = kprototypes(df, ["f1", "f2"], ["cat"], n_clusters=3, max_rows=300)
    assert kp["ok"]
    assert sum(kp["cluster_sizes"].values()) == 300


def test_did_pretrends_and_effect():
    rng = np.random.default_rng(4)
    rows = []
    for t in range(6):
        for i in range(40):
            treat = int(i >= 20)
            post = int(t >= 4)
            # parallel pre; effect +2 post for treated
            y = 5 + 0.2 * t + treat * 1.0 + post * treat * 2.0 + rng.normal(0, 0.3)
            rows.append({"unit": i, "time": t, "treat_g": treat, "y": y})
    df = pd.DataFrame(rows)
    pre = parallel_trends_test(df, "y", "treat_g", "time", 1, post_periods=[4, 5])
    assert pre["ok"]
    assert pre["parallel_trends_rejected_alpha05"] is False
    did = difference_in_differences_multipperiod(
        df, "y", "treat_g", "time", 1, post_periods=[4, 5]
    )
    assert did["ok"]
    assert abs(did["did_coef"] - 2.0) < 0.35


def test_evalue_on_logit_adjusted():
    rng = np.random.default_rng(5)
    n = 1500
    x = rng.normal(0, 1, n)
    t = (rng.normal(0.5 * x, 1) > 0).astype(int)
    # outcome binario influido por t y x
    logits = -1 + 1.2 * t + 0.8 * x
    p = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"treatment": t, "x": x, "outcome": y})
    res = exploratory_causal_screen(df, "treatment", "outcome", ["x"])
    assert res["adjusted_association"]["ok"]
    sens = res.get("sensitivity_adjusted")
    assert sens is not None
    assert "evalue_from_adjusted_or" in sens
    assert sens["evalue_from_adjusted_or"]["ok"]


def test_multitable_graph():
    a = pd.DataFrame({"customer_id": [1, 2, 3, 4], "revenue": [10, 20, 30, 40]})
    b = pd.DataFrame({"customer_id": [2, 3, 4, 5], "region": ["N", "S", "N", "E"]})
    g = build_relationship_graph({"orders": a, "customers": b}, min_overlap=0.2)
    assert len(g["edges"]) >= 1
    assert g["edges"][0]["key"] == "customer_id"
