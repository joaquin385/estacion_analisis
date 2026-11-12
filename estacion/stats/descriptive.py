"""Estadística descriptiva e intervalos de confianza."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(series: pd.Series, alpha: float = 0.05) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return {"n": n, "mean": float(s.mean()) if n else np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": float(s.std(ddof=1)),
        "se": se,
        "ci_low": mean - tcrit * se,
        "ci_high": mean + tcrit * se,
        "alpha": alpha,
        "method": "t-interval",
    }


def proportion_ci(successes: int, n: int, alpha: float = 0.05) -> dict[str, float]:
    """Intervalo de Wilson para proporciones."""
    if n <= 0:
        return {"n": 0, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    p = successes / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return {
        "n": n,
        "successes": successes,
        "p": p,
        "ci_low": float(max(0, center - half)),
        "ci_high": float(min(1, center + half)),
        "alpha": alpha,
        "method": "wilson",
    }


def descriptive_report(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> dict[str, Any]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out: dict[str, Any] = {}
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = {
            **mean_ci(s),
            "median": float(s.median()) if s.notna().any() else np.nan,
            "skewness": float(s.skew()) if s.notna().sum() > 2 else np.nan,
            "kurtosis": float(s.kurtosis()) if s.notna().sum() > 3 else np.nan,
            "missing": int(s.isna().sum()),
        }
    return out
