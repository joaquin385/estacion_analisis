"""Bootstrap no paramétrico para intervalos de confianza."""

from __future__ import annotations

from typing import Callable, Any

import numpy as np
import pandas as pd


def bootstrap_statistic(
    data: np.ndarray | pd.Series,
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    arr = np.asarray(pd.Series(data).dropna(), dtype=float)
    n = len(arr)
    if n < 5:
        return {"ok": False, "reason": "n demasiado pequeño", "n": n}
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = float(stat_fn(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {
        "ok": True,
        "n": n,
        "n_boot": n_boot,
        "estimate": float(stat_fn(arr)),
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std(ddof=1)),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "method": "percentile_bootstrap",
    }


def bootstrap_mean_ci(
    data: np.ndarray | pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    return bootstrap_statistic(data, np.mean, n_boot, alpha, random_state)


def bootstrap_difference(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """IC bootstrap de la diferencia de medias (a - b)."""
    aa = np.asarray(pd.Series(a).dropna(), dtype=float)
    bb = np.asarray(pd.Series(b).dropna(), dtype=float)
    if len(aa) < 5 or len(bb) < 5:
        return {"ok": False, "reason": "n insuficiente"}
    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(aa, size=len(aa), replace=True)
        sb = rng.choice(bb, size=len(bb), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    est = float(aa.mean() - bb.mean())
    return {
        "ok": True,
        "estimate": est,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant_at_alpha": not (lo <= 0 <= hi),
        "alpha": alpha,
        "method": "percentile_bootstrap_mean_diff",
    }
