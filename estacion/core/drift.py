"""Comparación entre dos datasets (train/test drift / data shift)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index para una variable numérica."""
    eps = 1e-6
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=cuts)[0].astype(float)
    a_counts = np.histogram(actual, bins=cuts)[0].astype(float)
    e_perc = e_counts / max(e_counts.sum(), 1) + eps
    a_perc = a_counts / max(a_counts.sum(), 1) + eps
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def compare_datasets(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str = "A",
    name_b: str = "B",
    alpha: float = 0.05,
) -> dict[str, Any]:
    common = [c for c in df_a.columns if c in df_b.columns]
    only_a = [c for c in df_a.columns if c not in df_b.columns]
    only_b = [c for c in df_b.columns if c not in df_a.columns]

    rows = []
    pvals = []
    for col in common:
        sa = df_a[col]
        sb = df_b[col]
        miss_a = float(sa.isna().mean())
        miss_b = float(sb.isna().mean())
        row: dict[str, Any] = {
            "column": col,
            "missing_pct_a": miss_a,
            "missing_pct_b": miss_b,
            "missing_delta": miss_b - miss_a,
        }
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            xa = pd.to_numeric(sa, errors="coerce").dropna().to_numpy(dtype=float)
            xb = pd.to_numeric(sb, errors="coerce").dropna().to_numpy(dtype=float)
            if len(xa) < 20 or len(xb) < 20:
                row.update({"ok": False, "reason": "n insuficiente"})
                rows.append(row)
                continue
            stat, p = stats.ks_2samp(xa, xb)
            psi = _psi(xa, xb)
            # Cohen d
            pooled = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2)
            d = float((xb.mean() - xa.mean()) / pooled) if pooled > 0 else np.nan
            row.update(
                {
                    "type": "numeric",
                    "method": "ks_2samp",
                    "statistic": float(stat),
                    "p_value": float(p),
                    "psi": psi,
                    "mean_a": float(xa.mean()),
                    "mean_b": float(xb.mean()),
                    "cohens_d": d,
                    "practically_important": bool(abs(d) >= 0.2 or psi >= 0.1),
                }
            )
            pvals.append(float(p))
        else:
            # Categorical: chi2 on aligned levels
            va = sa.astype(str).fillna("__NA__")
            vb = sb.astype(str).fillna("__NA__")
            levels = sorted(set(va.unique()) | set(vb.unique()))
            if len(levels) < 2:
                row.update({"ok": False, "reason": "sin niveles"})
                rows.append(row)
                continue
            ca = va.value_counts().reindex(levels, fill_value=0).to_numpy()
            cb = vb.value_counts().reindex(levels, fill_value=0).to_numpy()
            table = np.vstack([ca, cb])
            if table.sum() == 0 or (table == 0).all():
                rows.append(row)
                continue
            chi2, p, _, _ = stats.chi2_contingency(table)
            # TV distance
            pa = ca / max(ca.sum(), 1)
            pb = cb / max(cb.sum(), 1)
            tvd = 0.5 * float(np.abs(pa - pb).sum())
            row.update(
                {
                    "type": "categorical",
                    "method": "chi2",
                    "statistic": float(chi2),
                    "p_value": float(p),
                    "tvd": tvd,
                    "practically_important": bool(tvd >= 0.1),
                }
            )
            pvals.append(float(p))
        rows.append(row)

    if pvals:
        # Alinear FDR solo a filas con p_value
        idx = [i for i, r in enumerate(rows) if "p_value" in r]
        raw = [rows[i]["p_value"] for i in idx]
        rej, padj, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")
        for j, i in enumerate(idx):
            rows[i]["p_fdr"] = float(padj[j])
            rows[i]["significant_fdr"] = bool(rej[j])

    drifted = [
        r for r in rows
        if r.get("significant_fdr") and r.get("practically_important")
    ]
    drifted.sort(key=lambda r: (r.get("p_fdr", 1), -(abs(r.get("cohens_d") or r.get("tvd") or 0))))

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_rows_a": len(df_a),
        "n_rows_b": len(df_b),
        "common_columns": common,
        "only_in_a": only_a,
        "only_in_b": only_b,
        "column_tests": rows,
        "drifted_columns": drifted,
        "n_drifted": len(drifted),
        "alpha": alpha,
        "interpretation": (
            f"{len(drifted)} columnas con drift estadístico (FDR) y magnitud práctica."
            if drifted
            else "No se detectó drift práctico+FDR en columnas comunes."
        ),
        "limitations": [
            "KS/chi2 detectan cualquier diferencia de distribución; no todas importan al negocio.",
            "PSI umbrales clásicos: <0.1 estable, 0.1–0.25 leve, >0.25 fuerte.",
            "Esquema distinto (columnas solo en A/B) también es una forma de drift.",
        ],
    }
