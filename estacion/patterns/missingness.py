"""
Análisis de missingness: patrones, Little's MCAR (aproximado) y pantallas MAR.

No afirma MNAR (no demostrable solo con los datos observados).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "n_missing": miss,
                "missing_pct": miss / max(n, 1),
                "n_observed": n - miss,
            }
        )
    out = pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
    return out.reset_index(drop=True)


def missing_pairwise_rates(df: pd.DataFrame, max_cols: int = 30) -> pd.DataFrame:
    """Proporción de filas donde ambas columnas faltan simultáneamente."""
    cols = [c for c in df.columns if df[c].isna().any()][:max_cols]
    if len(cols) < 2:
        return pd.DataFrame()
    masks = {c: df[c].isna().to_numpy() for c in cols}
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n = len(df)
    for a in cols:
        for b in cols:
            mat.loc[a, b] = float((masks[a] & masks[b]).mean()) if n else 0.0
    return mat


def little_mcar_test(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> dict[str, Any]:
    """
    Implementación práctica del test de Little (MCAR) para variables numéricas.

    H0: los datos son MCAR.
    p bajo → evidencia contra MCAR (sugiere MAR/MNAR u otro mecanismo).
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # quedarse con columnas que tengan al menos algún missing y algún observado
    cols = [c for c in data.columns if data[c].isna().any() and data[c].notna().any()]
    if len(cols) < 2:
        return {
            "ok": False,
            "reason": "Se necesitan ≥2 columnas numéricas con missing parcial",
        }
    data = data[cols]
    # Patrones de missingness
    pattern = data.isna().astype(int)
    pattern_ids = pattern.apply(lambda r: tuple(r.tolist()), axis=1)
    unique_patterns = pattern_ids.unique()
    if len(unique_patterns) < 2:
        return {"ok": False, "reason": "Un solo patrón de missingness"}

    # Media global (listwise sobre filas completas si existen; si no, media por columna)
    complete = data.dropna()
    if len(complete) >= max(10, len(cols) + 1):
        mu = complete.mean().to_numpy()
        cov = complete.cov().to_numpy()
    else:
        mu = data.mean().to_numpy()
        cov = data.cov().to_numpy()
    # Regularizar cov
    cov = cov + np.eye(len(cols)) * 1e-6
    try:
        cov_inv = np.linalg.pinv(cov)
    except Exception:
        return {"ok": False, "reason": "Covarianza no invertible"}

    d2 = 0.0
    df_chi = 0
    pattern_stats = []
    for pat in unique_patterns:
        idx = pattern_ids == pat
        n_g = int(idx.sum())
        if n_g < 2:
            continue
        observed = [i for i, flag in enumerate(pat) if flag == 0]
        if not observed:
            continue
        sub = data.loc[idx].iloc[:, observed]
        mean_g = sub.mean().to_numpy()
        mu_g = mu[observed]
        # Submatriz de cov
        cov_g = cov[np.ix_(observed, observed)]
        try:
            inv_g = np.linalg.pinv(cov_g)
        except Exception:
            continue
        diff = mean_g - mu_g
        d2 += float(n_g * diff.T @ inv_g @ diff)
        df_chi += len(observed)
        pattern_stats.append(
            {
                "pattern": pat,
                "n": n_g,
                "n_observed_vars": len(observed),
            }
        )

    # df del test de Little ≈ (# medias estimadas en patrones) - p
    df_chi = max(df_chi - len(cols), 1)
    p_value = float(1 - stats.chi2.cdf(d2, df_chi))
    return {
        "ok": True,
        "method": "little_mcar",
        "statistic": d2,
        "df": df_chi,
        "p_value": p_value,
        "reject_mcar_alpha05": bool(p_value < 0.05),
        "n_patterns_used": len(pattern_stats),
        "columns": cols,
        "interpretation": (
            "Se rechaza MCAR (p<0.05): el missingness probablemente depende de variables observadas "
            "u omitidas (MAR/MNAR)."
            if p_value < 0.05
            else "No se rechaza MCAR a α=0.05; no prueba que sea MCAR, solo falta evidencia en contra."
        ),
        "limitations": [
            "Asume normalidad aproximada de las variables numéricas.",
            "No distingue MAR de MNAR.",
            "Patrones raros se omiten; resultado exploratorio.",
        ],
    }


def mar_screen(
    df: pd.DataFrame,
    max_targets: int = 8,
    max_predictors: int = 12,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Pantalla MAR: para cada columna con missing, ¿el indicador de missing
    se asocia a otras variables observadas?
    """
    miss_cols = [c for c in df.columns if 0 < df[c].isna().mean() < 1.0][:max_targets]
    predictors = [c for c in df.columns if c not in miss_cols][:max_predictors]
    tests: list[dict[str, Any]] = []
    pvals: list[float] = []

    for target in miss_cols:
        indicator = df[target].isna().astype(int)
        if indicator.nunique() < 2:
            continue
        for pred in predictors:
            s = df[pred]
            observed = s.notna()
            if observed.sum() < 30:
                continue
            y = indicator[observed]
            x = s[observed]
            if pd.api.types.is_numeric_dtype(x):
                a = x[y == 1].dropna().to_numpy(dtype=float)
                b = x[y == 0].dropna().to_numpy(dtype=float)
                if len(a) < 5 or len(b) < 5:
                    continue
                stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                method = "mannwhitney_missing_vs_observed"
                effect = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
            else:
                tab = pd.crosstab(x.astype(str), y)
                if tab.shape[0] < 2 or tab.shape[1] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(tab)
                stat = float(chi2)
                method = "chi2_missing_indicator"
                effect = np.nan
            row = {
                "missing_col": target,
                "predictor": pred,
                "method": method,
                "statistic": float(stat),
                "p_value": float(p),
                "effect": effect,
                "missing_rate_target": float(df[target].isna().mean()),
            }
            tests.append(row)
            pvals.append(float(p))

    if pvals:
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        for i, row in enumerate(tests):
            row["p_fdr"] = float(p_adj[i])
            row["significant_fdr"] = bool(reject[i])
        tests.sort(key=lambda r: r["p_fdr"])

    n_sig = sum(1 for t in tests if t.get("significant_fdr"))
    return {
        "ok": True,
        "n_tests": len(tests),
        "n_significant_fdr": n_sig,
        "alpha": alpha,
        "tests": tests[:100],
        "interpretation": (
            f"{n_sig} asociaciones missing~observado significativas tras FDR: "
            "compatible con MAR (o MNAR con proxy observado)."
            if n_sig
            else "Sin asociaciones FDR-significativas missing~observado en la pantalla."
        ),
        "limitations": [
            "Pantalla exploratoria; no prueba MAR formalmente.",
            "MNAR no se puede confirmar solo con datos observados.",
        ],
    }


def analyze_missingness(df: pd.DataFrame) -> dict[str, Any]:
    summary = missing_summary(df)
    pairwise = missing_pairwise_rates(df)
    little = little_mcar_test(df)
    mar = mar_screen(df)
    overall_rate = float(df.isna().mean().mean()) if len(df.columns) else 0.0
    return {
        "overall_missing_cell_rate": overall_rate,
        "summary": summary,
        "pairwise_both_missing": pairwise,
        "little_mcar": little,
        "mar_screen": mar,
    }
