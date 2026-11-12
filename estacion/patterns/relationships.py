"""Efectos entre variables categóricas y numéricas; interacciones simples."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from estacion.stats.hypothesis import group_comparison, categorical_association


def categorical_numeric_effects(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
    max_tests: int = 40,
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Evalúa qué categorías explican diferencias en variables numéricas."""
    results = []
    tests = 0
    for cat in categorical_cols:
        nlevels = df[cat].nunique(dropna=True)
        if nlevels < 2 or nlevels > 30:
            continue
        for num in numeric_cols:
            if tests >= max_tests:
                break
            res = group_comparison(df, num, cat, alpha=alpha)
            tests += 1
            if not res.get("ok"):
                continue
            score = 0.0
            if res.get("significant"):
                score += 40
            if res.get("practically_important"):
                score += 40
            effect = res.get("effect_size_cohens_d") or res.get("effect_size_eta_sq") or 0
            score += min(20, abs(float(effect)) * 20)
            results.append({**res, "priority_score": score})
        if tests >= max_tests:
            break
    results.sort(key=lambda r: -r["priority_score"])
    return results


def screen_categorical_pairs(
    df: pd.DataFrame,
    categorical_cols: list[str],
    max_pairs: int = 30,
) -> list[dict[str, Any]]:
    results = []
    cols = [c for c in categorical_cols if 1 < df[c].nunique(dropna=True) <= 20]
    count = 0
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            if count >= max_pairs:
                break
            res = categorical_association(df, a, b)
            count += 1
            if res.get("ok"):
                results.append(res)
        if count >= max_pairs:
            break
    results.sort(key=lambda r: -(r.get("cramers_v") or 0))
    return results


def interaction_screen(
    df: pd.DataFrame,
    y_col: str,
    factor_a: str,
    factor_b: str,
) -> dict[str, Any]:
    """
    Pantalla exploratoria de interacción: compara medias de y en celdas A×B
    y verifica si el rango entre celdas supera el de efectos marginales.
    No es un test formal de interacción ANOVA completo.
    """
    data = df[[y_col, factor_a, factor_b]].dropna()
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna()
    if len(data) < 40:
        return {"ok": False, "reason": "n insuficiente"}
    pivot = data.pivot_table(values=y_col, index=factor_a, columns=factor_b, aggfunc="mean")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return {"ok": False, "reason": "matriz incompleta"}
    cell_range = float(np.nanmax(pivot.values) - np.nanmin(pivot.values))
    marg_a = data.groupby(factor_a)[y_col].mean()
    marg_b = data.groupby(factor_b)[y_col].mean()
    max_marginal = float(
        max(marg_a.max() - marg_a.min(), marg_b.max() - marg_b.min())
    )
    return {
        "ok": True,
        "y": y_col,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "cell_range": cell_range,
        "max_marginal_range": max_marginal,
        "interaction_hint": bool(cell_range > max_marginal * 1.25),
        "cell_means": pivot.to_dict(),
        "limitations": [
            "Heurística exploratoria; no sustituye ANOVA/regresión con término de interacción.",
            "Celdas desbalanceadas pueden inducir artefactos.",
        ],
    }
