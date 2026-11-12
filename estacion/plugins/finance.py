"""Checks semánticos de dominio financiero (heurísticos)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from estacion.core.quality_rules import load_rules, evaluate_quality_rules

PRICE_RE = re.compile(r"(price|close|adj_close|open|px_|precio)", re.I)
RET_RE = re.compile(r"(return|ret|r_t|pct_change|rendimiento)", re.I)
VOL_RE = re.compile(r"(volume|vol|shares|cantidad)", re.I)
DATE_RE = re.compile(r"(date|fecha|timestamp|trade_date)", re.I)
TICKER_RE = re.compile(r"(ticker|symbol|isin|asset)", re.I)


def run_checks(df: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    q = evaluate_quality_rules(df, load_rules(domain="financial"))
    for v in q.get("violations", [])[:20]:
        findings.append({"type": "quality_rule", **v})

    for c in df.columns:
        name = str(c)
        s = df[c]
        if PRICE_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            n_nonpos = int((pd.to_numeric(s, errors="coerce") <= 0).sum())
            if n_nonpos:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Precio ≤ 0 en {n_nonpos} filas",
                        "value": n_nonpos,
                    }
                )
            # saltos diarios extremos si hay orden temporal implícito
            px = pd.to_numeric(s, errors="coerce")
            rets = px.pct_change()
            extreme = int((rets.abs() > 0.5).sum())
            if extreme:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"Variaciones |Δ|>50% en {extreme} pasos (splits/errores?)",
                        "value": extreme,
                    }
                )
        if RET_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            r = pd.to_numeric(s, errors="coerce").dropna()
            if len(r):
                if r.abs().median() > 1:
                    findings.append(
                        {
                            "type": "semantic",
                            "severity": "interesting",
                            "column": name,
                            "detail": "Retornos con mediana |r|>1: ¿están en % (50) en vez de decimal (0.5)?",
                        }
                    )
                extreme = int((r.abs() > 0.5).sum())
                if extreme:
                    findings.append(
                        {
                            "type": "semantic",
                            "severity": "notable",
                            "column": name,
                            "detail": f"Retornos extremos |r|>0.5: {extreme}",
                            "value": extreme,
                        }
                    )
        if VOL_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            n_neg = int((pd.to_numeric(s, errors="coerce") < 0).sum())
            if n_neg:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Volumen negativo: {n_neg}",
                        "value": n_neg,
                    }
                )
        if DATE_RE.search(name):
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() > 5 and not dt.dropna().is_monotonic_increasing:
                # puede ser panel multi-activo
                if not any(TICKER_RE.search(str(c2)) for c2 in df.columns):
                    findings.append(
                        {
                            "type": "semantic",
                            "severity": "interesting",
                            "column": name,
                            "detail": "Fechas no monótonas crecientes (¿desorden / panel sin ticker?).",
                        }
                    )

    # correlación precio-volumen si existen
    price_cols = [c for c in df.columns if PRICE_RE.search(str(c))]
    vol_cols = [c for c in df.columns if VOL_RE.search(str(c))]
    if price_cols and vol_cols:
        p = pd.to_numeric(df[price_cols[0]], errors="coerce")
        v = pd.to_numeric(df[vol_cols[0]], errors="coerce")
        if p.notna().sum() > 30:
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "detail": (
                        f"Par precio/volumen detectado ({price_cols[0]}, {vol_cols[0]}). "
                        "Útil para análisis de liquidez; cuidado con non-trading days."
                    ),
                }
            )

    n_crit = sum(1 for f in findings if f.get("severity") == "critical")
    return {
        "ok": True,
        "n_findings": len(findings),
        "n_critical": n_crit,
        "findings": findings,
        "notes": [
            "Heurísticas por nombre; no cubren corporate actions completas.",
            "No constituye asesoramiento financiero.",
        ],
    }
