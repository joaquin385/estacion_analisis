"""Checks semánticos de dominio healthcare (heurísticos)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from estacion.core.quality_rules import load_rules, evaluate_quality_rules

AGE_RE = re.compile(r"^(age|edad|age_years)$", re.I)
SEX_RE = re.compile(r"^(sex|gender|sexo|genero|género)$", re.I)
BP_SYS = re.compile(r"(systolic|pas|sbp|tension_sys)", re.I)
BP_DIA = re.compile(r"(diastolic|pad|dbp|tension_dia)", re.I)
HR_RE = re.compile(r"(heart_rate|hr|pulso|freq_card)", re.I)
ENCOUNTER = re.compile(r"(encounter|visit|admission|admission_id|patient_id|mrn)", re.I)
MORTALITY = re.compile(r"(death|mortal|expired|deceased|fallec)", re.I)


def run_checks(df: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    cols = {c: str(c) for c in df.columns}

    # quality preset
    q = evaluate_quality_rules(df, load_rules(domain="healthcare"))
    for v in q.get("violations", [])[:20]:
        findings.append({"type": "quality_rule", **v})

    for c, name in cols.items():
        s = df[c]
        if AGE_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            bad = ((s < 0) | (s > 120)).sum()
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Edad fuera de [0,120]: {int(bad)} filas",
                        "value": int(bad),
                    }
                )
        if SEX_RE.search(name):
            levels = set(s.dropna().astype(str).str.lower().unique())
            allowed = {"m", "f", "male", "female", "h", "mujer", "hombre", "other", "o", "unknown", "u", "0", "1"}
            weird = [x for x in levels if x not in allowed]
            if len(levels) > 8 or len(weird) > 3:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "interesting",
                        "column": name,
                        "detail": f"Sex/gender con niveles poco estándar: {sorted(list(levels))[:12]}",
                    }
                )
        if BP_SYS.search(name) and pd.api.types.is_numeric_dtype(s):
            bad = ((s < 50) | (s > 250)).sum()
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"PAS fuera de [50,250]: {int(bad)}",
                        "value": int(bad),
                    }
                )
        if BP_DIA.search(name) and pd.api.types.is_numeric_dtype(s):
            bad = ((s < 20) | (s > 150)).sum()
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"PAD fuera de [20,150]: {int(bad)}",
                        "value": int(bad),
                    }
                )
        if HR_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            bad = ((s < 20) | (s > 250)).sum()
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"FC fuera de [20,250]: {int(bad)}",
                        "value": int(bad),
                    }
                )
        if ENCOUNTER.search(name):
            dup = int(s.duplicated().sum())
            if dup and ENCOUNTER.search(name) and "patient" not in name.lower():
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"Posible ID de encuentro no único ({dup} duplicados)",
                        "value": dup,
                    }
                )
        if MORTALITY.search(name):
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "column": name,
                    "detail": "Variable sensible de mortalidad/desenlace detectada; revisar gobernanza y sesgo de censura.",
                }
            )

    # inconsistencia PAS < PAD si ambas existen
    sys_cols = [c for c in df.columns if BP_SYS.search(str(c))]
    dia_cols = [c for c in df.columns if BP_DIA.search(str(c))]
    if sys_cols and dia_cols:
        a = pd.to_numeric(df[sys_cols[0]], errors="coerce")
        b = pd.to_numeric(df[dia_cols[0]], errors="coerce")
        n_inv = int(((a < b) & a.notna() & b.notna()).sum())
        if n_inv:
            findings.append(
                {
                    "type": "semantic",
                    "severity": "critical",
                    "detail": f"PAS < PAD en {n_inv} filas ({sys_cols[0]} vs {dia_cols[0]})",
                    "value": n_inv,
                }
            )

    n_crit = sum(1 for f in findings if f.get("severity") == "critical")
    return {
        "ok": True,
        "n_findings": len(findings),
        "n_critical": n_crit,
        "findings": findings,
        "notes": [
            "Checks heurísticos por nombre de columna; adaptar a tu diccionario de datos.",
            "No sustituye validación clínica ni cumplimiento normativo.",
        ],
    }
