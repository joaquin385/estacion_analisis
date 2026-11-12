"""Detección de múltiples change-points (PELT con coste L2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _l2_seg_cost(prefix_sum: np.ndarray, prefix_sq: np.ndarray, a: int, b: int) -> float:
    """Coste L2 del segmento [a, b) (0-based, b exclusivo)."""
    n = b - a
    if n <= 0:
        return 0.0
    s = prefix_sum[b] - prefix_sum[a]
    q = prefix_sq[b] - prefix_sq[a]
    return float(q - (s * s) / n)


def pelt_l2(
    y: np.ndarray,
    min_size: int = 5,
    pen: float | None = None,
) -> list[int]:
    """
    PELT (Killick et al.) con coste gaussiano L2.

    Returns
    -------
    list[int]
        Índices de fin de segmento (change-points), sin incluir len(y).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * min_size:
        return []
    if pen is None:
        # BIC-like: pen = log(n) * sigma^2
        pen = float(np.log(n) * np.var(y))

    prefix_sum = np.concatenate([[0.0], np.cumsum(y)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(y * y)])

    # F[t] = óptimo hasta t; R candidatos
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    cp_prev = np.zeros(n + 1, dtype=int)
    R = [0]

    for tstar in range(min_size, n + 1):
        candidates = []
        for t in R:
            if tstar - t < min_size:
                continue
            cost = F[t] + _l2_seg_cost(prefix_sum, prefix_sq, t, tstar) + pen
            candidates.append((cost, t))
        if not candidates:
            continue
        best_cost, best_t = min(candidates, key=lambda x: x[0])
        F[tstar] = best_cost
        cp_prev[tstar] = best_t
        # Poda PELT
        new_R = []
        for t in R + [tstar]:
            if t >= tstar:
                continue
            if tstar - t < min_size:
                new_R.append(t)
                continue
            val = F[t] + _l2_seg_cost(prefix_sum, prefix_sq, t, tstar)
            if val <= F[tstar] + pen:  # K=0 para L2 homogéneo
                new_R.append(t)
        if tstar not in new_R:
            new_R.append(tstar)
        R = sorted(set(new_R))

    # Backtrack
    cps = []
    curr = n
    while curr > 0:
        prev = int(cp_prev[curr])
        if prev == 0:
            break
        cps.append(prev)
        curr = prev
    return sorted(cps)


def _var_seg_cost(y: np.ndarray, a: int, b: int) -> float:
    """Coste gaussiano de cambios de varianza (media constante por segmento = media local)."""
    n = b - a
    if n <= 1:
        return 0.0
    seg = y[a:b]
    v = float(np.var(seg, ddof=0))
    v = max(v, 1e-12)
    # n*log(var) + n  (up to constants)
    return float(n * np.log(v) + n)


def pelt_variance(
    y: np.ndarray,
    min_size: int = 10,
    pen: float | None = None,
) -> list[int]:
    """PELT para cambios de varianza (coste ~ n log σ²)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * min_size:
        return []
    if pen is None:
        pen = float(2 * np.log(n))

    F = np.full(n + 1, np.inf)
    F[0] = -pen
    cp_prev = np.zeros(n + 1, dtype=int)
    R = [0]
    for tstar in range(min_size, n + 1):
        candidates = []
        for t in R:
            if tstar - t < min_size:
                continue
            cost = F[t] + _var_seg_cost(y, t, tstar) + pen
            candidates.append((cost, t))
        if not candidates:
            continue
        best_cost, best_t = min(candidates, key=lambda x: x[0])
        F[tstar] = best_cost
        cp_prev[tstar] = best_t
        new_R = []
        for t in R + [tstar]:
            if t >= tstar:
                continue
            if tstar - t < min_size:
                new_R.append(t)
                continue
            val = F[t] + _var_seg_cost(y, t, tstar)
            if val <= F[tstar] + pen:
                new_R.append(t)
        if tstar not in new_R:
            new_R.append(tstar)
        R = sorted(set(new_R))

    cps = []
    curr = n
    while curr > 0:
        prev = int(cp_prev[curr])
        if prev == 0:
            break
        cps.append(prev)
        curr = prev
    return sorted(cps)


def detect_changepoints(
    series: pd.Series | np.ndarray,
    index: pd.Index | None = None,
    min_size: int = 5,
    pen: float | None = None,
    max_points: int = 8,
    target: str = "mean",
) -> dict[str, Any]:
    """
    target: 'mean' | 'variance' | 'both'
    """
    y = np.asarray(pd.Series(series).dropna(), dtype=float)
    if index is None:
        index = pd.RangeIndex(len(y))
    else:
        index = index[: len(y)]

    if len(y) < 2 * min_size:
        return {"ok": False, "reason": "Serie demasiado corta", "changepoints": []}

    def _pack(cps: list[int], method: str) -> dict[str, Any]:
        if len(cps) > max_points:
            jumps = []
            for c in cps:
                left = y[max(0, c - min_size) : c]
                right = y[c : min(len(y), c + min_size)]
                if method.endswith("VAR"):
                    score = abs(np.var(right) - np.var(left)) if len(left) and len(right) else 0
                else:
                    score = abs(left.mean() - right.mean()) if len(left) and len(right) else 0
                jumps.append((score, c))
            jumps.sort(reverse=True)
            cps = sorted(c for _, c in jumps[:max_points])

        details = []
        bounds = [0] + cps + [len(y)]
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            details.append(
                {
                    "segment": i,
                    "start_idx": a,
                    "end_idx": b,
                    "start": str(index[a]) if a < len(index) else None,
                    "end": str(index[min(b - 1, len(index) - 1)]),
                    "mean": float(y[a:b].mean()),
                    "std": float(y[a:b].std(ddof=0)),
                    "n": int(b - a),
                }
            )
        cp_rows = []
        for c in cps:
            left = y[max(0, c - min_size) : c]
            right = y[c : min(len(y), c + min_size)]
            cp_rows.append(
                {
                    "index": int(c),
                    "timestamp": str(index[c]) if c < len(index) else None,
                    "mean_before_window": float(left.mean()) if len(left) else np.nan,
                    "mean_after_window": float(right.mean()) if len(right) else np.nan,
                    "std_before_window": float(left.std(ddof=0)) if len(left) else np.nan,
                    "std_after_window": float(right.std(ddof=0)) if len(right) else np.nan,
                    "delta_mean": float(right.mean() - left.mean()) if len(left) and len(right) else np.nan,
                    "delta_std": float(right.std(ddof=0) - left.std(ddof=0)) if len(left) and len(right) else np.nan,
                }
            )
        return {
            "ok": True,
            "method": method,
            "penalty": pen if pen is not None else "auto",
            "min_size": min_size,
            "n_changepoints": len(cps),
            "changepoints": cp_rows,
            "segments": details,
        }

    if target == "variance":
        out = _pack(pelt_variance(y, min_size=max(min_size, 10), pen=pen), "PELT_VAR")
        out["limitations"] = [
            "Detecta cambios de varianza; la media puede cambiar también dentro del segmento.",
            "Penalización automática heurística.",
        ]
        return out

    mean_out = _pack(pelt_l2(y, min_size=min_size, pen=pen), "PELT_L2")
    mean_out["limitations"] = [
        "PELT L2 asume cambios en la media con ruido gaussiano homogéneo.",
        "La penalización automática es heurística (log(n)·σ²).",
    ]
    if target == "mean":
        return mean_out

    var_out = _pack(pelt_variance(y, min_size=max(min_size, 10), pen=pen), "PELT_VAR")
    return {
        "ok": True,
        "method": "PELT_mean_and_variance",
        "mean": mean_out,
        "variance": var_out,
        "n_changepoints_mean": mean_out["n_changepoints"],
        "n_changepoints_variance": var_out["n_changepoints"],
        "limitations": mean_out["limitations"] + [
            "Los cortes de media y varianza no tienen por qué coincidir.",
        ],
    }
