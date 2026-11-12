"""
Experimento reproducible: ¿recupera el motor patrones conocidos?

Ejecutar: python experiments/validate_discovery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estacion.discovery.engine import run_discovery
from estacion.causal.explorer import adjusted_association, propensity_score_matching
from estacion.clustering.segmenter import run_segmentation
from estacion.stats.correlation import correlation_analysis


def check(name: str, condition: bool, detail: str = "") -> dict:
    print(("PASS" if condition else "FAIL"), name, detail)
    return {"name": name, "pass": bool(condition), "detail": detail}


def main():
    results = []
    syn = ROOT / "data" / "synthetic"

    # Nonlinear
    df = pd.read_csv(syn / "nonlinear_null.csv")
    findings, _, art = run_discovery(df, "nonlinear_null")
    pairs = art.get("correlation", {}).get("pairs", [])
    xy = next(p for p in pairs if set([p["var_a"], p["var_b"]]) == {"x", "y"})
    zw = next(p for p in pairs if set([p["var_a"], p["var_b"]]) == {"z", "w"})
    results.append(check("nonlinear_xy_detected", xy.get("nonlinear_hint") or abs(xy["distance_corr"] or 0) > 0.3, str(xy)))
    results.append(check("null_zw_not_practically_important", not zw.get("practically_important", True), str(zw)))

    # Commercial concentration / region effect
    df = pd.read_csv(syn / "commercial.csv")
    findings, _, art = run_discovery(df, "commercial")
    blob = " ".join(f.title + f.observation for f in findings).lower()
    results.append(check("commercial_region_or_skew", "region" in blob or "asimétr" in blob or "revenue" in blob))
    results.append(check("commercial_duplicates", any("duplic" in f.title.lower() for f in findings)))

    # Causal
    df = pd.read_csv(syn / "causal_confounded.csv")
    crude = adjusted_association(df, "treatment", "outcome", [])
    adj = adjusted_association(df, "treatment", "outcome", ["SES"])
    results.append(check(
        "causal_adjustment_closer_to_2",
        abs(adj["adjusted"]["coef"] - 2) < abs(crude["adjusted"]["coef"] - 2),
        f"crude={crude['adjusted']['coef']:.3f} adj={adj['adjusted']['coef']:.3f}",
    ))
    psm = propensity_score_matching(df, "treatment", "outcome", ["SES"])
    results.append(check("psm_ate_near_2", psm.get("ok") and abs(psm["ate_estimate"] - 2) < 0.5, str(psm.get("ate_estimate"))))

    # Segments
    df = pd.read_csv(syn / "segments.csv")
    seg = run_segmentation(df, ["f1", "f2"])
    results.append(check("segments_k3", seg["ok"] and seg["kmeans"]["best"]["k"] == 3, str(seg["kmeans"]["best"])))

    # Financial trend
    df = pd.read_csv(syn / "financial.csv")
    findings, _, art = run_discovery(df, "financial")
    temporal_hit = any(f.category.value == "temporal" for f in findings)
    results.append(check("financial_temporal_finding", temporal_hit))

    # False discovery screen on pure noise
    import numpy as np
    rng = np.random.default_rng(99)
    noise = pd.DataFrame({f"n{i}": rng.normal(0, 1, 1000) for i in range(6)})
    corr = correlation_analysis(noise, include_distance=False)
    n_sig = sum(1 for p in corr["pairs"] if p["significant_fdr"] and p["practically_important"])
    results.append(check("noise_few_practical_fdr_hits", n_sig == 0, f"n_sig={n_sig}"))

    out = ROOT / "outputs" / "validation_report.json"
    out.parent.mkdir(exist_ok=True)
    summary = {
        "n_tests": len(results),
        "n_pass": sum(r["pass"] for r in results),
        "n_fail": sum(not r["pass"] for r in results),
        "results": results,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:", summary["n_pass"], "/", summary["n_tests"], "->", out)
    return summary


if __name__ == "__main__":
    main()
