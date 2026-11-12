"""Checks semánticos retail / e-commerce."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from estacion.core.quality_rules import load_rules, evaluate_quality_rules

PRICE_RE = re.compile(r"(price|precio|unit_price|amount|monto)", re.I)
QTY_RE = re.compile(r"(qty|quantity|units|cantidad|units_sold)", re.I)
REV_RE = re.compile(r"(revenue|sales|ventas|gmv|net_sales)", re.I)
DISC_RE = re.compile(r"(discount|descuento|promo)", re.I)
SKU_RE = re.compile(r"(sku|product_id|item_id)", re.I)
CHURN_RE = re.compile(r"(churn|cancel|refund|devol)", re.I)


def run_checks(df: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    q = evaluate_quality_rules(df, load_rules(domain="commercial"))
    for v in q.get("violations", [])[:20]:
        findings.append({"type": "quality_rule", **v})

    for c in df.columns:
        name = str(c)
        s = df[c]
        if PRICE_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            n_neg = int((pd.to_numeric(s, errors="coerce") < 0).sum())
            if n_neg:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Precio/monto negativo: {n_neg}",
                        "value": n_neg,
                    }
                )
        if QTY_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            n_nonpos = int((pd.to_numeric(s, errors="coerce") <= 0).sum())
            if n_nonpos:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"Cantidad ≤0: {n_nonpos} (¿cancelaciones sin filtrar?)",
                        "value": n_nonpos,
                    }
                )
        if DISC_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            # si parece proporción
            if x.dropna().between(0, 1).mean() > 0.8:
                over = int((x > 1).sum())
            else:
                over = int((x > 100).sum())
            if over:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "interesting",
                        "column": name,
                        "detail": f"Descuentos fuera de rango esperado: {over}",
                        "value": over,
                    }
                )
        if SKU_RE.search(name):
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "column": name,
                    "detail": f"SKU/product id con {s.nunique(dropna=True)} niveles.",
                }
            )
        if CHURN_RE.search(name):
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "column": name,
                    "detail": "Señal de churn/refund detectada; útil como target o filtro de calidad.",
                }
            )

    # consistencia revenue ≈ price * qty si existen
    price_cols = [c for c in df.columns if PRICE_RE.search(str(c))]
    qty_cols = [c for c in df.columns if QTY_RE.search(str(c))]
    rev_cols = [c for c in df.columns if REV_RE.search(str(c))]
    if price_cols and qty_cols and rev_cols:
        p = pd.to_numeric(df[price_cols[0]], errors="coerce")
        qn = pd.to_numeric(df[qty_cols[0]], errors="coerce")
        r = pd.to_numeric(df[rev_cols[0]], errors="coerce")
        approx = p * qn
        mask = p.notna() & qn.notna() & r.notna() & (approx.abs() > 1e-6)
        if mask.sum() > 20:
            rel = ((r - approx).abs() / approx.abs())[mask]
            bad = int((rel > 0.25).sum())
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "detail": (
                            f"revenue vs price×qty discrepa >25% en {bad} filas "
                            f"({rev_cols[0]} vs {price_cols[0]}×{qty_cols[0]})"
                        ),
                        "value": bad,
                    }
                )

    return {
        "ok": True,
        "n_findings": len(findings),
        "n_critical": sum(1 for f in findings if f.get("severity") == "critical"),
        "findings": findings,
        "notes": ["Heurísticas retail por nombre de columna.", "Validar con catálogo de productos real."],
    }
