"""Reglas de calidad configurables por dominio."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class QualityRuleSet:
    name: str = "default"
    max_missing_pct: float = 0.3
    max_duplicate_pct: float = 0.01
    max_iqr_outlier_pct: float = 0.05
    max_constant_cols: int = 0
    max_dominant_category_pct: float = 0.9
    forbidden_sentinel_values: list[float | int | str] = field(
        default_factory=lambda: [-999, -9999, 9999, "NA", "N/A", "null", "#N/A"]
    )
    required_columns: list[str] = field(default_factory=list)
    non_negative_columns: list[str] = field(default_factory=list)
    date_columns_must_parse: list[str] = field(default_factory=list)
    id_columns_must_be_unique: list[str] = field(default_factory=list)
    custom_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QualityRuleSet":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


DOMAIN_PRESETS: dict[str, QualityRuleSet] = {
    "default": QualityRuleSet(name="default"),
    "commercial": QualityRuleSet(
        name="commercial",
        max_missing_pct=0.2,
        max_duplicate_pct=0.005,
        non_negative_columns=["revenue", "units", "price", "amount", "marketing_spend"],
        id_columns_must_be_unique=["order_id", "customer_id"],
        custom_notes=["Montos no negativos; IDs de pedido únicos."],
    ),
    "financial": QualityRuleSet(
        name="financial",
        max_missing_pct=0.05,
        max_iqr_outlier_pct=0.08,
        date_columns_must_parse=["date", "trade_date", "fecha"],
        custom_notes=["Series financieras suelen tolerar colas; faltantes son críticos."],
    ),
    "operational": QualityRuleSet(
        name="operational",
        max_missing_pct=0.15,
        forbidden_sentinel_values=[-999, -9999, 9999, 99999, "NA"],
        non_negative_columns=["cycle_time", "temperature"],
        custom_notes=["Vigilar sentinelas de sensores (9999, etc.)."],
    ),
    "healthcare": QualityRuleSet(
        name="healthcare",
        max_missing_pct=0.1,
        max_duplicate_pct=0.0,
        id_columns_must_be_unique=["patient_id", "encounter_id"],
        custom_notes=["Duplicados exactos suelen ser errores de carga."],
    ),
}


def load_rules(path: str | Path | None = None, domain: str = "default") -> QualityRuleSet:
    if path:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:
                raise ImportError("Instalá PyYAML para cargar reglas .yaml")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return QualityRuleSet.from_dict(data)
    return DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["default"])


def evaluate_quality_rules(df: pd.DataFrame, rules: QualityRuleSet) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    n = len(df)

    for col in rules.required_columns:
        if col not in df.columns:
            violations.append(
                {
                    "rule": "required_columns",
                    "severity": "critical",
                    "column": col,
                    "detail": f"Falta columna requerida '{col}'",
                }
            )

    dup_pct = float(df.duplicated().mean()) if n else 0.0
    if dup_pct > rules.max_duplicate_pct:
        violations.append(
            {
                "rule": "max_duplicate_pct",
                "severity": "critical" if dup_pct > rules.max_duplicate_pct * 3 else "notable",
                "detail": f"Duplicados {dup_pct:.2%} > umbral {rules.max_duplicate_pct:.2%}",
                "value": dup_pct,
                "threshold": rules.max_duplicate_pct,
            }
        )

    for col in df.columns:
        miss = float(df[col].isna().mean()) if n else 0.0
        if miss > rules.max_missing_pct:
            violations.append(
                {
                    "rule": "max_missing_pct",
                    "severity": "notable",
                    "column": col,
                    "detail": f"'{col}' tiene {miss:.1%} faltantes (umbral {rules.max_missing_pct:.0%})",
                    "value": miss,
                    "threshold": rules.max_missing_pct,
                }
            )
        if df[col].nunique(dropna=True) <= 1:
            violations.append(
                {
                    "rule": "constant_column",
                    "severity": "interesting",
                    "column": col,
                    "detail": f"Columna constante '{col}'",
                }
            )
        # sentinels
        for sent in rules.forbidden_sentinel_values:
            try:
                cnt = int((df[col] == sent).sum())
            except Exception:
                cnt = 0
            if cnt:
                violations.append(
                    {
                        "rule": "forbidden_sentinel_values",
                        "severity": "critical",
                        "column": col,
                        "detail": f"'{col}' contiene sentinela {sent!r} ({cnt} veces)",
                        "value": cnt,
                        "sentinel": sent,
                    }
                )
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            vc = df[col].astype(str).value_counts(normalize=True, dropna=True)
            if len(vc) and float(vc.iloc[0]) > rules.max_dominant_category_pct:
                violations.append(
                    {
                        "rule": "max_dominant_category_pct",
                        "severity": "interesting",
                        "column": col,
                        "detail": (
                            f"'{col}' dominada por '{vc.index[0]}' "
                            f"({vc.iloc[0]:.1%} > {rules.max_dominant_category_pct:.0%})"
                        ),
                        "value": float(vc.iloc[0]),
                    }
                )

    for col in rules.non_negative_columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        n_neg = int((s < 0).sum())
        if n_neg:
            violations.append(
                {
                    "rule": "non_negative_columns",
                    "severity": "critical",
                    "column": col,
                    "detail": f"'{col}' tiene {n_neg} valores negativos",
                    "value": n_neg,
                }
            )

    for col in rules.id_columns_must_be_unique:
        if col not in df.columns:
            continue
        n_dup = int(df[col].duplicated().sum())
        if n_dup:
            violations.append(
                {
                    "rule": "id_columns_must_be_unique",
                    "severity": "critical",
                    "column": col,
                    "detail": f"ID '{col}' no es único ({n_dup} duplicados)",
                    "value": n_dup,
                }
            )

    for col in rules.date_columns_must_parse:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        fail = float(parsed.isna().mean())
        if fail > 0.05:
            violations.append(
                {
                    "rule": "date_columns_must_parse",
                    "severity": "notable",
                    "column": col,
                    "detail": f"'{col}' no parsea como fecha en {fail:.1%} de filas",
                    "value": fail,
                }
            )

    # IQR outliers en numéricas
    for col in df.select_dtypes(include=[np.number]).columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 20:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
        if pct > rules.max_iqr_outlier_pct:
            violations.append(
                {
                    "rule": "max_iqr_outlier_pct",
                    "severity": "interesting",
                    "column": col,
                    "detail": f"'{col}' outliers IQR {pct:.1%} > {rules.max_iqr_outlier_pct:.0%}",
                    "value": pct,
                    "threshold": rules.max_iqr_outlier_pct,
                }
            )

    sev_rank = {"critical": 0, "notable": 1, "interesting": 2, "info": 3}
    violations.sort(key=lambda v: sev_rank.get(v.get("severity", "info"), 9))
    return {
        "ruleset": rules.name,
        "n_violations": len(violations),
        "n_critical": sum(1 for v in violations if v.get("severity") == "critical"),
        "violations": violations,
        "passed": len(violations) == 0,
        "notes": rules.custom_notes,
    }


def save_rules_example(path: str | Path, domain: str = "commercial") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rules = DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["default"])
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(rules.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            path.write_text(yaml.dump(rules.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(rules.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
