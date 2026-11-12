"""Pronóstico exploratorio: ETS/ARIMA, grid AIC y intervalos conformal."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _prepare_ts(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str | None,
) -> tuple[pd.Series, str] | dict[str, Any]:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if len(data) < 24:
        return {"ok": False, "reason": "Se necesitan ≥24 puntos para un pronóstico exploratorio"}

    indexed = data.set_index(date_col)[value_col].sort_index()
    if freq is None:
        span = (indexed.index.max() - indexed.index.min()).days
        n = len(indexed)
        density = n / max(span, 1)
        if density >= 0.7 and n >= 24:
            freq = "D"
        elif span > 365 * 2 and n >= 36:
            freq = "MS"
        elif span > 180 and n >= 36:
            freq = "W"
        else:
            freq = "D"

    if freq == "raw":
        ts = indexed.dropna()
    else:
        ts = indexed.resample(freq).mean().dropna()
        if len(ts) < 24:
            ts = indexed.dropna()
            freq = "raw"
    if len(ts) < 24:
        return {"ok": False, "reason": f"Tras agregar a {freq} quedan <24 puntos"}
    return ts, freq


def _seasonal_period(freq: str) -> int:
    if freq.startswith("M"):
        return 12
    if freq.startswith("W"):
        return 52
    return 7


def _fit_forecast_candidates(
    ts: pd.Series,
    horizon: int,
    seasonal_periods: int,
) -> list[dict[str, Any]]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    cands = []
    # ETS variants
    for seasonal in (None, "add"):
        if seasonal and len(ts) < 2 * seasonal_periods:
            continue
        try:
            fit = ExponentialSmoothing(
                ts,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
            ).fit(optimized=True)
            pred = fit.forecast(horizon)
            aic = float(fit.aic) if getattr(fit, "aic", None) is not None else np.inf
            cands.append(
                {
                    "name": f"ets_add_{seasonal or 'noseas'}",
                    "aic": aic,
                    "kind": "ets",
                    "fit": fit,
                    "mean": pred.to_numpy(dtype=float),
                    "index": pred.index,
                    "resid": fit.resid.dropna().to_numpy(dtype=float),
                }
            )
        except Exception:
            continue

    # ARIMA/SARIMA grid pequeño
    orders = [(1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
    seasonal_orders = [(0, 0, 0, 0)]
    if len(ts) >= 2 * seasonal_periods + 8:
        seasonal_orders.append((0, 1, 1, seasonal_periods))
        seasonal_orders.append((1, 0, 0, seasonal_periods))

    for order in orders:
        for seas in seasonal_orders:
            try:
                if seas == (0, 0, 0, 0):
                    fit = ARIMA(ts, order=order).fit()
                    name = f"arima_{order[0]}{order[1]}{order[2]}"
                else:
                    fit = ARIMA(ts, order=order, seasonal_order=seas).fit()
                    name = f"sarima_{order}_{seas}"
                pred = fit.get_forecast(horizon).summary_frame(alpha=0.05)
                cands.append(
                    {
                        "name": name,
                        "aic": float(fit.aic),
                        "bic": float(fit.bic),
                        "kind": "arima",
                        "fit": fit,
                        "mean": pred["mean"].to_numpy(dtype=float),
                        "ci_low": pred["mean_ci_lower"].to_numpy(dtype=float),
                        "ci_high": pred["mean_ci_upper"].to_numpy(dtype=float),
                        "index": pred.index,
                        "resid": fit.resid.dropna().to_numpy(dtype=float),
                    }
                )
            except Exception:
                continue
    return cands


def _conformal_intervals(
    ts: pd.Series,
    fit_fn: Callable[[pd.Series], np.ndarray],
    horizon: int,
    alpha: float = 0.05,
    n_cal: int = 12,
) -> dict[str, Any]:
    """
    Conformal split residual: usa los últimos n_cal orígenes de 1-step (o h-step truncado)
    para calibrar un radio |y - ŷ|.
    """
    n = len(ts)
    if n < n_cal + horizon + 10:
        n_cal = max(5, n // 5)
    scores = []
    # rolling origin 1-step for stability
    starts = range(n - n_cal - 1, n - 1)
    for t in starts:
        train = ts.iloc[: t + 1]
        try:
            pred = fit_fn(train)
            if len(pred) < 1:
                continue
            yhat = float(pred[0])
            ytrue = float(ts.iloc[t + 1])
            scores.append(abs(ytrue - yhat))
        except Exception:
            continue
    if len(scores) < 5:
        return {"ok": False, "reason": "calibración conformal insuficiente", "n_scores": len(scores)}
    q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))))
    return {
        "ok": True,
        "alpha": alpha,
        "q_hat": q,
        "n_calibration": len(scores),
        "method": "absolute_residual_split_conformal_1step",
        "note": "Radio conformal calibrado con errores 1-step; se aplica a todo el horizonte (conservador/simple).",
    }


def forecast_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    horizon: int = 12,
    freq: str | None = None,
    method: str = "auto",
    conformal: bool = True,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    method: 'ets' | 'arima' | 'auto' | 'aic_grid'
    """
    prepared = _prepare_ts(df, date_col, value_col, freq)
    if isinstance(prepared, dict):
        return prepared
    ts, freq = prepared
    seasonal_periods = _seasonal_period(freq if freq != "raw" else "D")

    errors: list[str] = []
    used = None
    model_summary: dict[str, Any] = {}
    fc_mean = fc_lo = fc_hi = None
    index_fc = None
    resid = None
    selection = None

    if method in {"auto", "aic_grid"}:
        cands = _fit_forecast_candidates(ts, horizon, seasonal_periods)
        if not cands:
            return {"ok": False, "reason": "Ningún candidato AIC convergió", "errors": errors}
        cands_sorted = sorted(cands, key=lambda c: c.get("aic", np.inf))
        best = cands_sorted[0]
        selection = [
            {"name": c["name"], "aic": c.get("aic"), "bic": c.get("bic")}
            for c in cands_sorted[:8]
        ]
        used = best["name"]
        fc_mean = best["mean"]
        index_fc = best["index"]
        resid = best.get("resid")
        model_summary = {"aic": best.get("aic"), "bic": best.get("bic"), "selected": used}
        if "ci_low" in best:
            fc_lo, fc_hi = best["ci_low"], best["ci_high"]
        else:
            resid_std = float(np.std(resid, ddof=1)) if resid is not None and len(resid) > 2 else float(ts.std())
            z = 1.96
            fc_lo = fc_mean - z * resid_std * np.sqrt(np.arange(1, horizon + 1))
            fc_hi = fc_mean + z * resid_std * np.sqrt(np.arange(1, horizon + 1))
    else:
        # legacy paths
        try:
            if method == "ets":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                seasonal = "add" if len(ts) >= 2 * seasonal_periods else None
                fit = ExponentialSmoothing(
                    ts,
                    trend="add",
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None,
                ).fit(optimized=True)
                pred = fit.forecast(horizon)
                resid_std = float(np.std(fit.resid.dropna(), ddof=1))
                fc_mean = pred.to_numpy(dtype=float)
                fc_lo = fc_mean - 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
                fc_hi = fc_mean + 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
                index_fc = pred.index
                used = "holt_winters_ets"
                resid = fit.resid.dropna().to_numpy(dtype=float)
                model_summary = {"aic": float(fit.aic) if hasattr(fit, "aic") else None}
            else:
                from statsmodels.tsa.arima.model import ARIMA

                if len(ts) >= 2 * seasonal_periods + 10:
                    fit = ARIMA(ts, order=(1, 1, 1), seasonal_order=(0, 1, 1, seasonal_periods)).fit()
                    used = "sarima_1_1_1_0_1_1"
                else:
                    fit = ARIMA(ts, order=(1, 1, 1)).fit()
                    used = "arima_1_1_1"
                pred = fit.get_forecast(horizon).summary_frame(alpha=0.05)
                fc_mean = pred["mean"].to_numpy(dtype=float)
                fc_lo = pred["mean_ci_lower"].to_numpy(dtype=float)
                fc_hi = pred["mean_ci_upper"].to_numpy(dtype=float)
                index_fc = pred.index
                resid = fit.resid.dropna().to_numpy(dtype=float)
                model_summary = {"aic": float(fit.aic), "bic": float(fit.bic)}
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    # Conformal (1-step legacy + multi-horizon EnbPI-lite)
    conformal_info = None
    multi_conf = None
    if conformal:
        def _refit_1step(train: pd.Series) -> np.ndarray:
            from statsmodels.tsa.arima.model import ARIMA

            try:
                return ARIMA(train, order=(1, 1, 1)).fit().forecast(1).to_numpy(dtype=float)
            except Exception:
                return np.array([float(train.iloc[-1])])

        conformal_info = _conformal_intervals(ts, _refit_1step, horizon, alpha=alpha)

        from estacion.patterns.conformal import multihorizon_conformal, apply_multihorizon_radii

        def _fit_h(train: pd.Series, h: int) -> np.ndarray:
            from statsmodels.tsa.arima.model import ARIMA

            try:
                return ARIMA(train, order=(1, 1, 1)).fit().forecast(h).to_numpy(dtype=float)
            except Exception:
                return np.full(h, float(train.iloc[-1]))

        multi_conf = multihorizon_conformal(
            ts, _fit_h, horizon=horizon, alpha=alpha, n_cal=min(16, max(6, len(ts) // 10))
        )
        if multi_conf.get("ok"):
            fc_lo, fc_hi = apply_multihorizon_radii(fc_mean, multi_conf["q_hat_by_horizon"])
        elif conformal_info.get("ok"):
            q = conformal_info["q_hat"]
            fc_lo = fc_mean - q
            fc_hi = fc_mean + q

    # Backtest holdout
    hold = min(8, max(4, horizon // 2), len(ts) // 5)
    backtest = None
    if hold >= 4:
        train, test = ts.iloc[:-hold], ts.iloc[-hold:]
        try:
            cands_bt = _fit_forecast_candidates(train, hold, seasonal_periods)
            if cands_bt:
                best_bt = sorted(cands_bt, key=lambda c: c.get("aic", np.inf))[0]
                pred_bt = best_bt["mean"][:hold]
                mae = float(np.mean(np.abs(pred_bt - test.to_numpy())))
                mape = float(
                    np.mean(np.abs((pred_bt - test.to_numpy()) / (np.abs(test.to_numpy()) + 1e-9)))
                )
                backtest = {
                    "holdout_n": hold,
                    "mae": mae,
                    "mape": mape,
                    "selected_on_train": best_bt["name"],
                }
        except Exception as exc:
            backtest = {"ok": False, "reason": str(exc)}

    forecast_rows = [
        {
            "timestamp": str(index_fc[i]),
            "mean": float(fc_mean[i]),
            "ci95_low": float(fc_lo[i]),
            "ci95_high": float(fc_hi[i]),
        }
        for i in range(min(horizon, len(fc_mean)))
    ]
    return {
        "ok": True,
        "method": used,
        "freq": freq,
        "horizon": horizon,
        "history_end": str(ts.index.max()),
        "history_n": len(ts),
        "model": model_summary,
        "aic_selection": selection,
        "conformal": conformal_info,
        "conformal_multihorizon": multi_conf,
        "forecast": forecast_rows,
        "backtest": backtest,
        "limitations": [
            "Pronóstico exploratorio; no valida régimen futuro ni cambios estructurales.",
            "Grid AIC es acotado (no exhaustivo).",
            "Conformal multi-horizon calibra radios por h con orígenes rodantes (EnbPI-lite).",
            "No usar como única base de decisión operativa/financiera.",
        ],
    }
