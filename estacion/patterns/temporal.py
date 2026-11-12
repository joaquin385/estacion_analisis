"""Análisis temporal: tendencias, STL, change-points y estacionalidad."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from estacion.patterns.changepoint import detect_changepoints


def _infer_freq(data: pd.DataFrame, date_col: str) -> str:
    span = (data[date_col].max() - data[date_col].min()).days
    if span > 365 * 2:
        return "MS"
    if span > 60:
        return "W"
    return "D"


def _stl_decompose(ts: pd.Series, period: int | None = None) -> dict[str, Any]:
    """STL via statsmodels; degrada a seasonal_decompose clásico si hace falta."""
    if not isinstance(ts.index, pd.DatetimeIndex):
        return {"ok": False, "reason": "Se requiere DatetimeIndex"}
    if len(ts) < 24:
        return {"ok": False, "reason": "Serie corta para STL (<24)"}

    if period is None:
        # Heurística por frecuencia
        freq = pd.infer_freq(ts.index)
        if freq and freq.upper().startswith("M"):
            period = 12
        elif freq and freq.upper().startswith("W"):
            period = 52
        elif freq and freq.upper().startswith("D"):
            period = 7
        else:
            period = 12 if len(ts) >= 36 else 7
    period = max(2, min(period, len(ts) // 2))

    try:
        from statsmodels.tsa.seasonal import STL

        stl = STL(ts, period=period, robust=True)
        res = stl.fit()
        method = "STL"
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid
    except Exception:
        from statsmodels.tsa.seasonal import seasonal_decompose

        res = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
        method = "seasonal_decompose"
        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

    # Fuerza estacional: var(seasonal) / var(seasonal+resid)
    seas = seasonal.dropna()
    rem = (seasonal + resid).dropna()
    strength = float(max(0.0, 1 - (resid.dropna().var() / (rem.var() + 1e-12))))
    # Correlación de estacionalidad lag=period
    y = ts.dropna().to_numpy()
    acf_p = np.nan
    if len(y) > period + 5:
        y0, y1 = y[:-period], y[period:]
        if np.std(y0) > 0 and np.std(y1) > 0:
            acf_p = float(np.corrcoef(y0, y1)[0, 1])

    return {
        "ok": True,
        "method": method,
        "period": period,
        "seasonal_strength": strength,
        "seasonal_acf_period": acf_p,
        "trend_slope": float(np.polyfit(np.arange(len(trend.dropna())), trend.dropna(), 1)[0])
        if trend.notna().sum() > 5
        else np.nan,
        "residual_std": float(resid.dropna().std()),
        "detected_seasonality": bool(strength >= 0.3 or (pd.notna(acf_p) and abs(acf_p) >= 0.3)),
        "components_tail": {
            "trend": {str(k): float(v) for k, v in trend.dropna().tail(8).items()},
            "seasonal": {str(k): float(v) for k, v in seasonal.dropna().tail(8).items()},
        },
        "limitations": [
            "Periodo estacional inferido heurísticamente si no se especifica.",
            "STL asume estacionalidad aditiva aproximadamente constante.",
        ],
    }


def temporal_analysis(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str | None = None,
    stl_period: int | None = None,
) -> dict[str, Any]:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if len(data) < 20:
        return {"ok": False, "reason": "Serie demasiado corta (<20 puntos tras limpieza)"}

    if freq is None:
        freq = _infer_freq(data, date_col)

    ts = data.set_index(date_col)[value_col].resample(freq).mean().dropna()
    if len(ts) < 12:
        ts = data.set_index(date_col)[value_col]
        freq = "raw"

    x = np.arange(len(ts))
    y = ts.to_numpy(dtype=float)
    slope, intercept, r, p, se = stats.linregress(x, y)

    # Change-point único (legacy, contraste de medias)
    change_point = None
    if len(y) >= 20:
        best = {"idx": None, "stat": 0.0, "p": 1.0}
        for i in range(max(5, len(y) // 10), len(y) - max(5, len(y) // 10)):
            a, b = y[:i], y[i:]
            if len(a) < 5 or len(b) < 5:
                continue
            tstat, pv = stats.ttest_ind(a, b, equal_var=False)
            if abs(tstat) > abs(best["stat"]):
                best = {"idx": i, "stat": float(tstat), "p": float(pv)}
        if best["idx"] is not None:
            change_point = {
                "index": int(best["idx"]),
                "timestamp": str(ts.index[best["idx"]]),
                "t_stat": best["stat"],
                "p_value": best["p"],
                "mean_before": float(y[: best["idx"]].mean()),
                "mean_after": float(y[best["idx"] :].mean()),
            }

    # Change-points múltiples (PELT media + varianza)
    pelt = detect_changepoints(
        y, index=ts.index, min_size=max(5, len(y) // 30), target="both"
    )

    # Estacionalidad Kruskal (rápida)
    seasonality = None
    if isinstance(ts.index, pd.DatetimeIndex) and len(ts) >= 24:
        month = ts.index.month
        if month.nunique() >= 4:
            groups = [
                ts[month == m].to_numpy()
                for m in sorted(month.unique())
                if (month == m).sum() >= 2
            ]
            if len(groups) >= 3:
                stat, p_s = stats.kruskal(*groups)
                seasonality = {
                    "method": "kruskal_by_month",
                    "statistic": float(stat),
                    "p_value": float(p_s),
                    "detected": bool(p_s < 0.05),
                }

    stl = _stl_decompose(ts, period=stl_period) if freq != "raw" else {"ok": False, "reason": "freq raw"}

    return {
        "ok": True,
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq,
        "n_points": len(ts),
        "trend": {
            "slope_per_step": float(slope),
            "intercept": float(intercept),
            "r_value": float(r),
            "p_value": float(p),
            "std_err": float(se),
            "significant": bool(p < 0.05),
            "direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
        },
        "change_point": change_point,
        "changepoints_pelt": pelt,
        "seasonality": seasonality,
        "stl": stl,
        "start": str(ts.index.min()),
        "end": str(ts.index.max()),
        "series_tail": {str(k): float(v) for k, v in ts.tail(12).items()},
        "limitations": [
            "PELT detecta cambios de media; validar con contexto de negocio.",
            "STL depende del periodo; revisar si la frecuencia es correcta.",
            "Agregación (resample) puede ocultar patrones intra-período.",
        ],
    }
