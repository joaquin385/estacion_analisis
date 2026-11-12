"""Correlaciones lineales y no lineales con corrección por comparaciones múltiples."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation (Székely) — detecta dependencia no lineal. O(n²)."""
    n = len(x)
    if n < 10 or n > 2000:
        # Demasiado costoso o insuficiente
        return np.nan
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    dcov2 = (A * B).mean()
    dvarx = (A * A).mean()
    dvary = (B * B).mean()
    if dvarx <= 0 or dvary <= 0:
        return np.nan
    # dCor = sqrt( dCov² / (dVar(X) * dVar(Y)) ) con dVar = dCov² consigo misma
    return float(np.sqrt(dcov2 / (dvarx * dvary)))


def correlation_analysis(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "both",
    alpha: float = 0.05,
    max_pairs: int = 200,
    include_distance: bool = True,
) -> dict[str, Any]:
    """
    Calcula pares de correlación con p-values y FDR (Benjamini-Hochberg).

    Returns matrices y lista de pares significativos.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if df[c].nunique(dropna=True) > 1]
    if len(columns) < 2:
        return {"pairs": [], "pearson_matrix": None, "spearman_matrix": None}

    data = df[columns].apply(pd.to_numeric, errors="coerce")
    pairs = list(combinations(columns, 2))
    if len(pairs) > max_pairs:
        # Priorizar por |corr| preliminar en muestra
        sample = data.dropna().sample(min(len(data.dropna()), 5000), random_state=42)
        prelim = []
        for a, b in pairs:
            r = sample[a].corr(sample[b])
            prelim.append((abs(r) if pd.notna(r) else 0, a, b))
        prelim.sort(reverse=True)
        pairs = [(a, b) for _, a, b in prelim[:max_pairs]]

    results = []
    pvals = []
    for a, b in pairs:
        sub = data[[a, b]].dropna()
        x, y = sub[a].to_numpy(), sub[b].to_numpy()
        pr, pp = _safe_pearson(x, y)
        sr, sp = _safe_spearman(x, y)
        dcor = np.nan
        if include_distance and len(sub) >= 10:
            # O(n²): siempre subsamplear para mantener coste acotado
            if len(sub) > 800:
                idx = np.random.default_rng(42).choice(len(sub), 800, replace=False)
                dcor = _distance_correlation(x[idx], y[idx])
            else:
                dcor = _distance_correlation(x, y)
        nonlinear_hint = (
            pd.notna(dcor)
            and pd.notna(pr)
            and abs(dcor) >= 0.25
            and abs(pr) < 0.25
            and abs(dcor) > abs(pr) + 0.1
        )
        row = {
            "var_a": a,
            "var_b": b,
            "n": len(sub),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "distance_corr": dcor,
            "nonlinear_hint": bool(nonlinear_hint),
        }
        results.append(row)
        pvals.append(sp if pd.notna(sp) else 1.0)

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    for i, row in enumerate(results):
        row["spearman_p_fdr"] = float(p_adj[i])
        row["significant_fdr"] = bool(reject[i])
        # Importancia práctica: |rho| >= 0.3
        row["practically_important"] = bool(
            pd.notna(row["spearman_r"]) and abs(row["spearman_r"]) >= 0.3
        )

    pearson_mat = data.corr(method="pearson")
    spearman_mat = data.corr(method="spearman")

    results.sort(
        key=lambda r: (
            -(abs(r["spearman_r"]) if pd.notna(r["spearman_r"]) else 0),
            r["spearman_p_fdr"],
        )
    )
    return {
        "pairs": results,
        "pearson_matrix": pearson_mat,
        "spearman_matrix": spearman_mat,
        "alpha": alpha,
        "correction": "fdr_bh",
        "n_pairs_tested": len(results),
    }
