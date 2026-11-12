"""Regresión lineal/logística, residuos y multicolinealidad."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _prepare_xy(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = [y_col] + x_cols
    data = df[cols].copy()
    for c in x_cols + [y_col]:
        if not pd.api.types.is_numeric_dtype(data[c]):
            # One-hot simple para categóricas de baja cardinalidad
            if data[c].nunique(dropna=True) <= 12 and c != y_col:
                dummies = pd.get_dummies(data[c], prefix=c, drop_first=True, dtype=float)
                data = pd.concat([data.drop(columns=[c]), dummies], axis=1)
            else:
                data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    y = data[y_col]
    X = data.drop(columns=[y_col])
    # Solo numéricas
    X = X.select_dtypes(include=[np.number])
    return X, y, list(X.columns)


def compute_vif(X: pd.DataFrame) -> dict[str, float]:
    if X.shape[1] < 2 or len(X) < X.shape[1] + 2:
        return {}
    Xc = sm.add_constant(X)
    vifs = {}
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        try:
            vifs[col] = float(variance_inflation_factor(Xc.values, i))
        except Exception:
            vifs[col] = np.nan
    return vifs


def linear_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
) -> dict[str, Any]:
    X, y, used = _prepare_xy(df, y_col, x_cols)
    if len(X) < max(20, len(used) + 5) or not used:
        return {
            "ok": False,
            "reason": "Muestra insuficiente o sin predictores válidos",
            "n": len(X),
        }
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    resid = model.resid
    fitted = model.fittedvalues
    # Breusch-Pagan approx via correlación |resid|~fitted
    bp_r = float(pd.Series(np.abs(resid)).corr(pd.Series(fitted))) if len(resid) > 5 else np.nan
    vifs = compute_vif(X)
    return {
        "ok": True,
        "type": "ols",
        "n": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "params": {k: float(v) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "conf_int": {
            str(k): [float(v.iloc[0]), float(v.iloc[1])]
            for k, v in model.conf_int().iterrows()
        },
        "vif": vifs,
        "multicollinearity_flag": any(v > 10 for v in vifs.values() if pd.notna(v)),
        "residual_mean": float(resid.mean()),
        "residual_std": float(resid.std()),
        "heteroscedasticity_hint_corr": bp_r,
        "predictors_used": used,
        "summary_text": model.summary().as_text(),
        "limitations": [
            "Asociación, no causalidad.",
            "Supuestos: linealidad, independencia, homocedasticidad, normalidad de residuos.",
            "VIF>10 sugiere multicolinealidad problemática.",
        ],
    }


def logistic_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
) -> dict[str, Any]:
    X, y, used = _prepare_xy(df, y_col, x_cols)
    y_num = pd.to_numeric(y, errors="coerce")
    # Binarizar si es necesario
    uniq = sorted(y_num.dropna().unique())
    if len(uniq) != 2:
        return {"ok": False, "reason": "La variable objetivo no es binaria", "levels": uniq[:10]}
    y_bin = (y_num == uniq[-1]).astype(int)
    if len(X) < max(30, 10 * len(used)) or not used:
        return {
            "ok": False,
            "reason": "Muestra insuficiente (regla aproximada: ≥10 eventos por predictor)",
            "n": len(X),
        }
    Xc = sm.add_constant(X)
    try:
        model = sm.Logit(y_bin, Xc).fit(disp=False, maxiter=200)
    except Exception as exc:
        return {"ok": False, "reason": f"No convergió: {exc}"}
    return {
        "ok": True,
        "type": "logit",
        "n": int(model.nobs),
        "pseudo_r_squared": float(model.prsquared),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "params": {k: float(v) for k, v in model.params.items()},
        "odds_ratios": {k: float(np.exp(v)) for k, v in model.params.items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.items()},
        "predictors_used": used,
        "positive_class": float(uniq[-1]),
        "summary_text": model.summary().as_text(),
        "limitations": [
            "Odds ratios ≠ riesgo relativo; no implican causalidad.",
            "Requiere eventos suficientes por predictor.",
            "Verificar separación perfecta y especificación del modelo.",
        ],
    }
