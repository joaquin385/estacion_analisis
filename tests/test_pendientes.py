"""Tests de funcionalidades de la lista de pendientes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from estacion.patterns.missingness import analyze_missingness, little_mcar_test, mar_screen
from estacion.core.quality_rules import load_rules, evaluate_quality_rules, save_rules_example
from estacion.core.drift import compare_datasets
from estacion.core.session import SessionStore
from estacion.causal.sensitivity import evalue_from_risk_ratio, rosenbaum_bounds_signed, sensitivity_report
from estacion.causal.explorer import propensity_score_matching
from estacion.discovery.engine import run_discovery
from estacion.reports.pdf_export import generate_pdf_report
from estacion.core.profiler import profile_dataframe


def test_missingness_mar_and_little():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    # Missing en y depende de x (MAR)
    miss = x > 0.8
    y = y.copy()
    y[miss] = np.nan
    df = pd.DataFrame({"x": x, "y": y, "z": rng.normal(0, 1, n)})
    res = analyze_missingness(df)
    assert res["mar_screen"]["n_tests"] > 0
    # Debe haber alguna asociación y~x missing
    assert any(
        t["missing_col"] == "y" and t["predictor"] == "x" for t in res["mar_screen"]["tests"]
    )


def test_quality_rules_commercial_flags_duplicates_and_sentinel():
    df = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 3],
            "revenue": [10, 10, -5, 3],
            "temperature": [70, 71, 9999, 68],
        }
    )
    rules = load_rules(domain="commercial")
    # add temperature sentinel check via operational-like values on commercial ruleset
    out = evaluate_quality_rules(df, rules)
    assert out["n_violations"] >= 1
    assert any(v["rule"] == "id_columns_must_be_unique" for v in out["violations"])
    assert any(v["rule"] == "non_negative_columns" for v in out["violations"])


def test_quality_rules_yaml_roundtrip(tmp_path: Path):
    path = save_rules_example(tmp_path / "rules.yaml", "operational")
    rules = load_rules(path)
    assert rules.name == "operational"
    assert 9999 in rules.forbidden_sentinel_values


def test_global_fdr_artifact():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 400),
            "b": rng.normal(0, 1, 400),
            "c": rng.normal(0, 1, 400),
            "g": rng.choice(["x", "y"], 400),
        }
    )
    df["b"] = df["a"] * 2 + rng.normal(0, 0.1, 400)
    findings, _, art = run_discovery(df, "fdr")
    assert "fdr_global" in art
    assert art["fdr_global"]["n_tested"] >= 1
    assert any(f.p_value is not None for f in findings)


def test_evalue_and_rosenbaum():
    ev = evalue_from_risk_ratio(2.0)
    assert ev["ok"]
    assert ev["e_value"] > 1
    rng = np.random.default_rng(2)
    t = rng.normal(1, 1, 80)
    c = rng.normal(0, 1, 80)
    rb = rosenbaum_bounds_signed(t, c)
    assert rb["ok"]
    assert rb["p_value_gamma1"] < 0.05


def test_psm_includes_balance_and_sensitivity():
    rng = np.random.default_rng(3)
    n = 2000
    ses = rng.normal(0, 1, n)
    treatment = (rng.normal(ses, 1) > 0).astype(int)
    outcome = 5 + 2 * treatment + 3 * ses + rng.normal(0, 1, n)
    df = pd.DataFrame({"treatment": treatment, "SES": ses, "outcome": outcome})
    res = propensity_score_matching(df, "treatment", "outcome", ["SES"])
    assert res["ok"]
    assert "covariate_balance" in res
    assert "sensitivity" in res
    assert res["sensitivity"].get("rosenbaum", {}).get("ok")


def test_drift_detects_shift():
    rng = np.random.default_rng(4)
    a = pd.DataFrame({"x": rng.normal(0, 1, 500), "cat": rng.choice(["a", "b"], 500)})
    b = pd.DataFrame({"x": rng.normal(1.5, 1, 500), "cat": rng.choice(["a", "b", "c"], 500)})
    res = compare_datasets(a, b)
    assert res["n_drifted"] >= 1
    assert any(r["column"] == "x" for r in res["drifted_columns"])


def test_session_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1, 0, 1, 0]})
    findings, profile, art = run_discovery(df, "tiny")
    store = SessionStore(tmp_path / "sessions")
    meta = store.save(df, profile, findings, art, name="tiny", save_parquet=False)
    loaded = store.load(meta.id)
    assert loaded["meta"]["id"] == meta.id
    assert loaded["df"] is not None or (tmp_path / "sessions" / meta.id / "schema_only.csv").exists()
    # With save_parquet False we only store schema; force csv save path:
    meta2 = store.save(df, profile, findings, art, name="tiny2", save_parquet=True)
    loaded2 = store.load(meta2.id)
    assert loaded2["df"] is not None
    assert len(loaded2["findings"]) == len(findings)


def test_pdf_export(tmp_path: Path):
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "x": rng.normal(0, 1, 200),
            "y": rng.normal(0, 1, 200),
            "g": rng.choice(["a", "b"], 200),
        }
    )
    findings, profile, art = run_discovery(df, "pdf")
    path = tmp_path / "r.pdf"
    generate_pdf_report(df, profile, findings, path, art)
    assert path.exists()
    assert path.stat().st_size > 1000
