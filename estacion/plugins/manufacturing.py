"""Checks semánticos manufacturing / operaciones."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from estacion.core.quality_rules import load_rules, evaluate_quality_rules

TEMP_RE = re.compile(r"(temp|temperature|temperatura)", re.I)
CYCLE_RE = re.compile(r"(cycle|tact|lead_time|cycle_time|duracion|duration)", re.I)
DEFECT_RE = re.compile(r"(defect|scrap|reject|falla|defective|ng)", re.I)
PLANT_RE = re.compile(r"(plant|factory|linea|line|station|planta)", re.I)
SENSOR_RE = re.compile(r"(sensor|reading|measure|medicion)", re.I)


def run_checks(df: pd.DataFrame) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    q = evaluate_quality_rules(df, load_rules(domain="operational"))
    for v in q.get("violations", [])[:20]:
        findings.append({"type": "quality_rule", **v})

    for c in df.columns:
        name = str(c)
        s = df[c]
        if TEMP_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            sent = int(((x == 9999) | (x == -9999) | (x == 99999)).sum())
            if sent:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Sentinelas de sensor en temperatura: {sent}",
                        "value": sent,
                    }
                )
            # rango físico amplio
            bad = int(((x < -100) | (x > 2000)).sum())
            if bad:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"Temperatura fuera de [-100,2000]: {bad}",
                        "value": bad,
                    }
                )
        if CYCLE_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            n_nonpos = int((x <= 0).sum())
            if n_nonpos:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "critical",
                        "column": name,
                        "detail": f"Tiempos de ciclo ≤0: {n_nonpos}",
                        "value": n_nonpos,
                    }
                )
        if DEFECT_RE.search(name):
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "column": name,
                    "detail": "Métrica de defectos/scrap detectada; candidata a target de calidad.",
                }
            )
        if PLANT_RE.search(name):
            nlev = s.nunique(dropna=True)
            findings.append(
                {
                    "type": "semantic",
                    "severity": "info",
                    "column": name,
                    "detail": f"Factor de planta/línea con {nlev} niveles.",
                }
            )
        if SENSOR_RE.search(name) and pd.api.types.is_numeric_dtype(s):
            miss = float(s.isna().mean())
            if miss > 0.1:
                findings.append(
                    {
                        "type": "semantic",
                        "severity": "notable",
                        "column": name,
                        "detail": f"Sensor con {miss:.1%} faltantes (posible downtime).",
                        "value": miss,
                    }
                )

    return {
        "ok": True,
        "n_findings": len(findings),
        "n_critical": sum(1 for f in findings if f.get("severity") == "critical"),
        "findings": findings,
        "notes": [
            "Heurísticas de manufactura; calibrar rangos a tu proceso.",
            "Sentinelas 9999 suelen ser errores de PLC/DAQ.",
        ],
    }
