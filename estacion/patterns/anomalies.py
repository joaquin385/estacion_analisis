"""Detección de anomalías globales y contextuales."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def iqr_outliers(series: pd.Series, k: float = 1.5) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 10:
        return {"ok": False, "reason": "n insuficiente"}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"ok": True, "n_outliers": 0, "pct": 0.0, "method": "iqr"}
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (s < lower) | (s > upper)
    return {
        "ok": True,
        "method": "iqr",
        "n_outliers": int(mask.sum()),
        "pct": float(mask.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "indices": s.index[mask].tolist()[:100],
    }


def detect_anomalies(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    contamination: float = 0.02,
    max_rows: int = 20_000,
) -> dict[str, Any]:
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 5]
    univariate = {c: iqr_outliers(df[c]) for c in numeric_cols}

    result: dict[str, Any] = {"univariate": univariate, "multivariate": None}
    if len(numeric_cols) < 2:
        return result

    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 50:
        result["multivariate"] = {"ok": False, "reason": "n insuficiente"}
        return result

    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
        sampled = True

    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)
    clf = IsolationForest(
        contamination=min(max(contamination, 0.005), 0.1),
        random_state=42,
        n_estimators=200,
    )
    labels = clf.fit_predict(X)
    scores = clf.decision_function(X)
    anomaly_idx = data.index[labels == -1]
    result["multivariate"] = {
        "ok": True,
        "method": "isolation_forest",
        "n_anomalies": int((labels == -1).sum()),
        "pct": float((labels == -1).mean()),
        "contamination": contamination,
        "sampled": sampled,
        "score_min": float(scores.min()),
        "score_median": float(np.median(scores)),
        "anomaly_indices": anomaly_idx.tolist()[:200],
        "limitations": [
            "IsolationForest es no supervisado; anomalía ≠ error.",
            "contamination es un hiperparámetro; resultados sensibles a su valor.",
            "No distingue anomalías contextuales (p.ej. por segmento) sin estratificar.",
        ],
    }
    return result


def contextual_anomalies(
    df: pd.DataFrame,
    value_col: str,
    context_col: str,
    z_thresh: float = 3.0,
) -> dict[str, Any]:
    """Valores extremos relativos al grupo contextual."""
    data = df[[value_col, context_col]].dropna()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna()
    if len(data) < 30:
        return {"ok": False, "reason": "n insuficiente"}

    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=1)
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    data = data.copy()
    data["z_ctx"] = data.groupby(context_col)[value_col].transform(zscore)
    mask = data["z_ctx"].abs() >= z_thresh
    return {
        "ok": True,
        "method": "group_zscore",
        "z_thresh": z_thresh,
        "n_anomalies": int(mask.sum()),
        "pct": float(mask.mean()),
        "examples": data.loc[mask].head(20).to_dict(orient="records"),
    }
