"""Inferencia de roles de variables (ID, fecha, categórica, numérica, etc.)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .types import VariableRole

ID_NAME_RE = re.compile(
    r"(^id$|_id$|^uuid$|^guid$|codigo|código|key$|_key$|dni|cuit|sku)",
    re.IGNORECASE,
)
DATE_NAME_RE = re.compile(
    r"(fecha|date|time|timestamp|periodo|período|anio|año|year|month|día|dia)",
    re.IGNORECASE,
)
TARGET_NAME_RE = re.compile(
    r"(target|label|y$|churn|conversion|conversión|fraude|default|"
    r"outcome|resultado|score|rating|precio|price|amount|monto|revenue|"
    r"ventas|sales|ingreso|profit|margen)",
    re.IGNORECASE,
)


def _try_parse_datetime(series: pd.Series, sample: int = 200) -> float:
    """Proporción de valores parseables como fecha."""
    s = series.dropna().astype(str).head(sample)
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    return float(parsed.notna().mean())


def infer_role(series: pd.Series, name: str) -> VariableRole:
    n = len(series)
    n_non_null = int(series.notna().sum())
    if n_non_null == 0:
        return VariableRole.UNKNOWN

    n_unique = int(series.nunique(dropna=True))
    unique_ratio = n_unique / max(n_non_null, 1)

    if n_unique <= 1:
        return VariableRole.CONSTANT

    # Datetime nativo
    if pd.api.types.is_datetime64_any_dtype(series):
        return VariableRole.DATETIME

    # Numérico
    if pd.api.types.is_numeric_dtype(series):
        if n_unique == 2:
            return VariableRole.BINARY
        # ID numérico: casi único Y nombre típico de ID, o unicidad total en n grande
        if ID_NAME_RE.search(name) and unique_ratio > 0.9:
            return VariableRole.IDENTIFIER
        if unique_ratio > 0.995 and n_non_null >= 100 and series.dtype.kind in "iu":
            # enteros casi únicos en muestra grande → probable surrogado
            return VariableRole.IDENTIFIER
        return VariableRole.NUMERIC

    # Object / string
    if DATE_NAME_RE.search(name):
        if _try_parse_datetime(series) > 0.5:
            return VariableRole.DATETIME
    elif n_unique > 5 and _try_parse_datetime(series) > 0.85:
        return VariableRole.DATETIME

    if ID_NAME_RE.search(name) and unique_ratio > 0.9:
        return VariableRole.IDENTIFIER

    if unique_ratio > 0.95 and n_non_null > 50:
        # Alta cardinalidad textual → probablemente ID o texto libre
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len and avg_len > 40:
            return VariableRole.TEXT
        return VariableRole.IDENTIFIER

    if n_unique == 2:
        return VariableRole.BINARY

    if n_unique <= min(50, max(20, int(0.05 * n_non_null))):
        return VariableRole.CATEGORICAL

    avg_len = series.dropna().astype(str).str.len().mean()
    if avg_len and avg_len > 60:
        return VariableRole.TEXT

    return VariableRole.CATEGORICAL


def is_target_candidate(name: str, role: VariableRole) -> bool:
    if role in {VariableRole.IDENTIFIER, VariableRole.CONSTANT, VariableRole.TEXT}:
        return False
    return bool(TARGET_NAME_RE.search(name)) or role in {
        VariableRole.BINARY,
        VariableRole.NUMERIC,
    }


def coerce_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def summarize_numeric(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    skew = float(s.skew()) if len(s) > 2 else 0.0
    kurt = float(s.kurtosis()) if len(s) > 3 else 0.0
    # IQR outliers
    q1, q3 = float(desc["25%"]), float(desc["75%"])
    iqr = q3 - q1
    if iqr > 0:
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
    else:
        n_out = 0
        lower = upper = None
    return {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not np.isnan(desc["std"]) else 0.0,
        "min": float(desc["min"]),
        "p01": float(desc["1%"]),
        "p05": float(desc["5%"]),
        "p25": float(desc["25%"]),
        "p50": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "p95": float(desc["95%"]),
        "p99": float(desc["99%"]),
        "max": float(desc["max"]),
        "skewness": skew,
        "kurtosis": kurt,
        "iqr_outlier_count": n_out,
        "iqr_outlier_pct": n_out / len(s),
        "iqr_lower": lower,
        "iqr_upper": upper,
        "zeros_pct": float((s == 0).mean()),
        "negatives_pct": float((s < 0).mean()),
        "cv": float(desc["std"] / desc["mean"]) if desc["mean"] not in (0, np.nan) else None,
    }


def summarize_categorical(series: pd.Series, top_n: int = 10) -> dict[str, Any]:
    s = series.dropna()
    if s.empty:
        return {}
    vc = s.astype(str).value_counts()
    top = vc.head(top_n)
    entropy = float(-(vc / len(s) * np.log2(vc / len(s) + 1e-15)).sum())
    return {
        "n_levels": int(vc.shape[0]),
        "top_values": {str(k): int(v) for k, v in top.items()},
        "top_share": float(top.iloc[0] / len(s)) if len(top) else 0.0,
        "entropy": entropy,
        "mode": str(top.index[0]) if len(top) else None,
    }
