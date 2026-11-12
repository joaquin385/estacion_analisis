"""Pruebas de hipótesis y comparaciones entre grupos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def group_comparison(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    alpha: float = 0.05,
    min_group_n: int = 10,
) -> dict[str, Any]:
    """
    Compara una variable numérica entre grupos categóricos.
    Elige t-test / Mann-Whitney (2 grupos) o ANOVA / Kruskal-Wallis (k grupos).
    Incluye tamaño del efecto y advertencias por n pequeño.
    """
    data = df[[value_col, group_col]].dropna()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    groups = {str(g): gdf[value_col].to_numpy() for g, gdf in data.groupby(group_col)}
    groups = {k: v for k, v in groups.items() if len(v) >= min_group_n}
    warnings: list[str] = []

    if len(groups) < 2:
        return {
            "ok": False,
            "reason": "Menos de 2 grupos con n suficiente",
            "min_group_n": min_group_n,
            "group_sizes": {k: len(v) for k, v in groups.items()},
        }

    sizes = {k: len(v) for k, v in groups.items()}
    if any(n < 30 for n in sizes.values()):
        warnings.append("Algunos grupos tienen n<30; preferir métodos no paramétricos.")

    arrays = list(groups.values())
    labels = list(groups.keys())

    # Normalidad aproximada (Shapiro en muestra)
    normality = {}
    for lab, arr in groups.items():
        sample = arr if len(arr) <= 500 else np.random.default_rng(0).choice(arr, 500, replace=False)
        if len(sample) >= 8:
            _, p = stats.shapiro(sample)
            normality[lab] = float(p)
        else:
            normality[lab] = np.nan
    mostly_normal = all(pd.isna(p) or p > 0.05 for p in normality.values())

    result: dict[str, Any] = {
        "ok": True,
        "value_col": value_col,
        "group_col": group_col,
        "group_sizes": sizes,
        "group_means": {k: float(v.mean()) for k, v in groups.items()},
        "group_medians": {k: float(np.median(v)) for k, v in groups.items()},
        "normality_shapiro_p": normality,
        "warnings": warnings,
        "alpha": alpha,
    }

    if len(groups) == 2:
        a, b = arrays[0], arrays[1]
        if mostly_normal:
            stat, p = stats.ttest_ind(a, b, equal_var=False)
            method = "welch_t"
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            method = "mannwhitney_u"
        d = _cohens_d(a, b)
        result.update(
            {
                "method": method,
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < alpha),
                "effect_size_cohens_d": d,
                "practically_important": bool(pd.notna(d) and abs(d) >= 0.5),
                "groups_compared": labels,
            }
        )
    else:
        if mostly_normal:
            stat, p = stats.f_oneway(*arrays)
            method = "anova_oneway"
        else:
            stat, p = stats.kruskal(*arrays)
            method = "kruskal_wallis"
        # Eta-squared aproximado
        grand = np.concatenate(arrays)
        ss_between = sum(len(a) * (a.mean() - grand.mean()) ** 2 for a in arrays)
        ss_total = ((grand - grand.mean()) ** 2).sum()
        eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
        result.update(
            {
                "method": method,
                "statistic": float(stat),
                "p_value": float(p),
                "significant": bool(p < alpha),
                "effect_size_eta_sq": eta2,
                "practically_important": bool(pd.notna(eta2) and eta2 >= 0.06),
                "groups_compared": labels,
            }
        )
        # Pairwise con FDR
        pair_p = []
        pair_rows = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                u, pv = stats.mannwhitneyu(arrays[i], arrays[j], alternative="two-sided")
                pair_rows.append(
                    {
                        "a": labels[i],
                        "b": labels[j],
                        "p_raw": float(pv),
                        "cohens_d": _cohens_d(arrays[i], arrays[j]),
                    }
                )
                pair_p.append(pv)
        if pair_p:
            rej, padj, _, _ = multipletests(pair_p, alpha=alpha, method="fdr_bh")
            for i, row in enumerate(pair_rows):
                row["p_fdr"] = float(padj[i])
                row["significant_fdr"] = bool(rej[i])
            result["pairwise"] = pair_rows

    return result


def categorical_association(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Chi-cuadrado / Fisher + V de Cramer."""
    tab = pd.crosstab(df[col_a], df[col_b])
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return {"ok": False, "reason": "Tabla de contingencia insuficiente"}

    chi2, p, dof, expected = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    r, k = tab.shape
    cramers_v = float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if n and min(r, k) > 1 else np.nan

    use_fisher = tab.shape == (2, 2) and (expected < 5).any()
    fisher_p = None
    if use_fisher:
        odds, fisher_p = stats.fisher_exact(tab)

    return {
        "ok": True,
        "col_a": col_a,
        "col_b": col_b,
        "chi2": float(chi2),
        "p_value": float(fisher_p) if fisher_p is not None else float(p),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "method": "fisher_exact" if use_fisher else "chi2",
        "significant": bool((fisher_p if fisher_p is not None else p) < alpha),
        "practically_important": bool(pd.notna(cramers_v) and cramers_v >= 0.1),
        "contingency": tab.to_dict(),
        "warning_expected_lt5": bool((expected < 5).any()),
    }
