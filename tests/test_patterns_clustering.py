"""Tests de patrones, temporal y clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from estacion.patterns.anomalies import detect_anomalies, iqr_outliers
from estacion.patterns.temporal import temporal_analysis
from estacion.clustering.segmenter import run_segmentation
from estacion.discovery.engine import run_discovery


def test_iqr_flags_extreme():
    x = list(range(100)) + [10000]
    res = iqr_outliers(pd.Series(x))
    assert res["ok"]
    assert res["n_outliers"] >= 1


def test_isolation_forest_finds_spikes():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    })
    df.loc[:20, ["a", "b", "c"]] = 8
    res = detect_anomalies(df, contamination=0.03)
    assert res["multivariate"]["ok"]
    assert res["multivariate"]["n_anomalies"] > 0


def test_temporal_trend_and_changepoint():
    n = 300
    t = np.arange(n)
    y = 0.05 * t + np.random.default_rng(1).normal(0, 0.5, n)
    y[200:] += 10
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "y": y,
    })
    res = temporal_analysis(df, "date", "y")
    assert res["ok"]
    assert res["trend"]["significant"]
    assert res["change_point"] is not None
    assert res["change_point"]["p_value"] < 0.05


def test_segmentation_recovers_k():
    rng = np.random.default_rng(2)
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    labels = rng.integers(0, 3, 600)
    X = centers[labels] + rng.normal(0, 0.5, size=(600, 2))
    df = pd.DataFrame({"f1": X[:, 0], "f2": X[:, 1]})
    seg = run_segmentation(df, ["f1", "f2"])
    assert seg["ok"]
    assert seg["natural_groups_likely"]
    assert seg["kmeans"]["best"]["k"] == 3


def test_discovery_on_nonlinear_known():
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 3, 800)
    y = x ** 2 + rng.normal(0, 0.3, 800)
    z = rng.normal(0, 1, 800)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    findings, profile, artifacts = run_discovery(df, "nl")
    assert set(profile.numeric_columns) >= {"x", "y", "z"}
    pairs = artifacts.get("correlation", {}).get("pairs", [])
    xy = [p for p in pairs if set([p["var_a"], p["var_b"]]) == {"x", "y"}]
    assert xy, "No se calculó el par x-y"
    assert xy[0].get("nonlinear_hint") or (xy[0].get("distance_corr") or 0) >= 0.25
    assert any(set(f.variables) >= {"x", "y"} for f in findings)
