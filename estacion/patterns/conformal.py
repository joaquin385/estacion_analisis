"""Intervalos conformal multi-horizon (EnbPI-lite / residuales por horizonte)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def multihorizon_conformal(
    ts: pd.Series,
    fit_forecast_fn: Callable[[pd.Series, int], np.ndarray],
    horizon: int,
    alpha: float = 0.05,
    n_cal: int = 20,
    min_train: int = 40,
) -> dict[str, Any]:
    """
    Calibra un radio |y_{t+h} - ŷ_{t+h}| por cada h=1..H con orígenes rodantes.

    fit_forecast_fn(train_series, horizon) -> array shape (horizon,)
    """
    y = ts.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < min_train + horizon + 5:
        return {"ok": False, "reason": "Serie corta para conformal multi-horizon"}

    n_cal = min(n_cal, max(5, n - min_train - horizon))
    scores = {h: [] for h in range(1, horizon + 1)}

    # orígenes: terminan en t, predicen t+1..t+H
    last_origin = n - horizon - 1
    first_origin = max(min_train - 1, last_origin - n_cal + 1)
    for t in range(first_origin, last_origin + 1):
        train = pd.Series(y[: t + 1])
        try:
            pred = np.asarray(fit_forecast_fn(train, horizon), dtype=float)
        except Exception:
            continue
        if len(pred) < horizon:
            continue
        for h in range(1, horizon + 1):
            yhat = float(pred[h - 1])
            ytrue = float(y[t + h])
            scores[h].append(abs(ytrue - yhat))

    q_hat = {}
    n_scores = {}
    for h, sc in scores.items():
        n_scores[h] = len(sc)
        if len(sc) < 5:
            q_hat[h] = np.nan
            continue
        # quantile conformal split
        lvl = min(1.0, np.ceil((len(sc) + 1) * (1 - alpha)) / len(sc))
        q_hat[h] = float(np.quantile(sc, lvl))

    ok = sum(np.isfinite(v) for v in q_hat.values()) >= max(1, horizon // 2)
    return {
        "ok": ok,
        "method": "multihorizon_residual_conformal",
        "alpha": alpha,
        "q_hat_by_horizon": {str(k): v for k, v in q_hat.items()},
        "n_scores_by_horizon": {str(k): v for k, v in n_scores.items()},
        "note": (
            "EnbPI-lite: radios distintos por horizonte con orígenes rodantes. "
            "No reentrena ensemble; usa el mismo fit_forecast_fn."
        ),
        "limitations": [
            "Asume errores aproximadamente intercambiables en la ventana de calibración.",
            "Costoso si fit_forecast_fn es pesado (se limita n_cal).",
        ],
    }


def apply_multihorizon_radii(
    mean: np.ndarray,
    q_hat_by_horizon: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    lo = np.empty_like(mean, dtype=float)
    hi = np.empty_like(mean, dtype=float)
    for i, m in enumerate(mean):
        h = str(i + 1)
        q = q_hat_by_horizon.get(h, np.nan)
        if not np.isfinite(q):
            # fallback: max q disponible
            vals = [v for v in q_hat_by_horizon.values() if np.isfinite(v)]
            q = float(max(vals)) if vals else 0.0
        lo[i] = m - q
        hi[i] = m + q
    return lo, hi
