"""Tests de exactitud estadística y perfilado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from estacion.core.profiler import profile_dataframe
from estacion.core.type_inference import infer_role
from estacion.core.types import VariableRole
from estacion.stats.descriptive import mean_ci, proportion_ci
from estacion.stats.correlation import correlation_analysis, _distance_correlation
from estacion.stats.hypothesis import group_comparison, categorical_association
from estacion.stats.bootstrap import bootstrap_mean_ci
from estacion.stats.regression import linear_regression
from estacion.causal.explorer import adjusted_association, propensity_score_matching
from estacion.discovery.engine import run_discovery


def test_mean_ci_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 2, 200)
    res = mean_ci(pd.Series(x), alpha=0.05)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    assert res["mean"] == pytest.approx(mean)
    assert res["ci_low"] == pytest.approx(mean - t * se)
    assert res["ci_high"] == pytest.approx(mean + t * se)


def test_wilson_interval_bounds():
    res = proportion_ci(45, 100)
    assert 0 <= res["ci_low"] <= res["p"] <= res["ci_high"] <= 1


def test_infer_roles():
    assert infer_role(pd.Series([1, 2, 3, 4, 5]), "amount") == VariableRole.NUMERIC
    assert infer_role(pd.Series(["a", "b", "a", "b"]), "flag") == VariableRole.BINARY
    assert infer_role(pd.Series(range(100)), "user_id") == VariableRole.IDENTIFIER
    assert infer_role(pd.Series(["x"] * 50), "const") == VariableRole.CONSTANT


def test_pearson_known():
    x = np.linspace(0, 1, 100)
    y = 2 * x + 1
    df = pd.DataFrame({"x": x, "y": y})
    corr = correlation_analysis(df, ["x", "y"], include_distance=False)
    pair = corr["pairs"][0]
    assert pair["pearson_r"] == pytest.approx(1.0, abs=1e-8)


def test_distance_corr_quadratic():
    rng = np.random.default_rng(1)
    x = rng.uniform(-2, 2, 400)
    y = x ** 2 + rng.normal(0, 0.05, 400)
    # Pearson débil, distance más alto
    pr, _ = stats.pearsonr(x, y)
    dcor = _distance_correlation(x, y)
    assert abs(pr) < 0.25
    assert dcor > 0.4


def test_group_comparison_detects_difference():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "g": ["A"] * 80 + ["B"] * 80,
        "v": np.concatenate([rng.normal(0, 1, 80), rng.normal(1.2, 1, 80)]),
    })
    res = group_comparison(df, "v", "g")
    assert res["ok"]
    assert res["significant"]
    assert abs(res["effect_size_cohens_d"]) > 0.5


def test_null_group_difference_often_nonsignificant():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "g": ["A"] * 80 + ["B"] * 80,
        "v": rng.normal(0, 1, 160),
    })
    res = group_comparison(df, "v", "g")
    assert res["ok"]
    # No assert nonsignificant siempre, pero tamaño de efecto debe ser pequeño
    assert abs(res["effect_size_cohens_d"]) < 0.5


def test_chi2_association():
    df = pd.DataFrame({
        "a": ["x", "x", "y", "y"] * 40,
        "b": ["u", "u", "v", "v"] * 40,
    })
    res = categorical_association(df, "a", "b")
    assert res["ok"]
    assert res["cramers_v"] == pytest.approx(1.0, abs=0.05)


def test_bootstrap_covers_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(5, 1, 150)
    res = bootstrap_mean_ci(x, n_boot=1000, random_state=4)
    assert res["ok"]
    assert res["ci_low"] <= 5 <= res["ci_high"]


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    y = 3 + 1.5 * x + rng.normal(0, 0.2, 500)
    df = pd.DataFrame({"x": x, "y": y})
    res = linear_regression(df, "y", ["x"])
    assert res["ok"]
    assert res["params"]["x"] == pytest.approx(1.5, abs=0.1)
    assert res["params"]["const"] == pytest.approx(3.0, abs=0.1)


def test_causal_adjustment_reduces_bias():
    rng = np.random.default_rng(6)
    n = 2000
    ses = rng.normal(0, 1, n)
    treatment = (rng.normal(ses, 1) > 0).astype(int)
    outcome = 5 + 2.0 * treatment + 3.0 * ses + rng.normal(0, 1, n)
    df = pd.DataFrame({"treatment": treatment, "SES": ses, "outcome": outcome})
    crude = adjusted_association(df, "treatment", "outcome", [])
    adj = adjusted_association(df, "treatment", "outcome", ["SES"])
    assert adj["ok"] and crude["ok"]
    crude_c = crude["adjusted"]["coef"]
    adj_c = adj["adjusted"]["coef"]
    assert abs(adj_c - 2.0) < abs(crude_c - 2.0)
    assert abs(adj_c - 2.0) < 0.25


def test_psm_recovers_ate_approx():
    rng = np.random.default_rng(7)
    n = 2500
    ses = rng.normal(0, 1, n)
    treatment = (rng.normal(ses * 0.8, 1) > 0).astype(int)
    outcome = 1 + 2.0 * treatment + 2.5 * ses + rng.normal(0, 1, n)
    df = pd.DataFrame({"treatment": treatment, "SES": ses, "outcome": outcome})
    res = propensity_score_matching(df, "treatment", "outcome", ["SES"], caliper=0.05)
    assert res["ok"]
    assert abs(res["ate_estimate"] - 2.0) < 0.45


def test_profiler_and_discovery_smoke():
    rng = np.random.default_rng(8)
    df = pd.DataFrame({
        "id": np.arange(200),
        "cat": rng.choice(["a", "b", "c"], 200),
        "x": rng.normal(0, 1, 200),
        "y": rng.normal(0, 1, 200),
        "revenue": rng.lognormal(3, 1, 200),
    })
    profile = profile_dataframe(df, "t")
    assert profile.n_rows == 200
    findings, _, _ = run_discovery(df, "t")
    assert isinstance(findings, list)


def test_missing_and_incomplete_behavior():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5] * 20,
        "b": [np.nan] * 100,
        "g": ["x", "y"] * 50,
    })
    profile = profile_dataframe(df, "m")
    assert any(v.name == "b" and v.missing_pct == 1.0 for v in profile.variables)
    res = group_comparison(df, "a", "g", min_group_n=5)
    assert res["ok"]
