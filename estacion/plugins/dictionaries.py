"""Diccionarios de datos versionados (YAML/JSON)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

import pandas as pd


@dataclass
class ColumnSpec:
    name: str
    role: str = "unknown"
    dtype: str | None = None
    unit: str | None = None
    min: float | None = None
    max: float | None = None
    allowed_values: list[Any] | None = None
    description: str = ""


@dataclass
class DataDictionary:
    name: str
    version: str = "1.0.0"
    domain: str = "general"
    columns: list[ColumnSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "domain": self.domain,
            "columns": [asdict(c) for c in self.columns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataDictionary":
        cols = [ColumnSpec(**c) for c in data.get("columns", [])]
        return cls(
            name=data.get("name", "dict"),
            version=str(data.get("version", "1.0.0")),
            domain=data.get("domain", "general"),
            columns=cols,
        )


def load_dictionary(path: str | Path) -> DataDictionary:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML requerido para diccionarios YAML")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return DataDictionary.from_dict(data)


def save_dictionary(dd: DataDictionary, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dd.to_dict()
    if path.suffix.lower() in {".yml", ".yaml"} and yaml is not None:
        path.write_text(yaml.dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        path = path.with_suffix(".json") if path.suffix.lower() in {".yml", ".yaml"} else path
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_against_dictionary(df: pd.DataFrame, dd: DataDictionary) -> dict[str, Any]:
    findings = []
    present = set(map(str, df.columns))
    expected = {c.name for c in dd.columns}
    for miss in sorted(expected - present):
        findings.append(
            {
                "type": "dictionary",
                "severity": "critical",
                "column": miss,
                "detail": f"Falta columna definida en diccionario v{dd.version}",
            }
        )
    for extra in sorted(present - expected):
        findings.append(
            {
                "type": "dictionary",
                "severity": "info",
                "column": extra,
                "detail": "Columna no documentada en el diccionario",
            }
        )
    for spec in dd.columns:
        if spec.name not in df.columns:
            continue
        s = df[spec.name]
        if spec.min is not None or spec.max is not None:
            x = pd.to_numeric(s, errors="coerce")
            if spec.min is not None:
                n = int((x < spec.min).sum())
                if n:
                    findings.append(
                        {
                            "type": "dictionary",
                            "severity": "notable",
                            "column": spec.name,
                            "detail": f"{n} valores < min={spec.min}",
                            "value": n,
                        }
                    )
            if spec.max is not None:
                n = int((x > spec.max).sum())
                if n:
                    findings.append(
                        {
                            "type": "dictionary",
                            "severity": "notable",
                            "column": spec.name,
                            "detail": f"{n} valores > max={spec.max}",
                            "value": n,
                        }
                    )
        if spec.allowed_values is not None:
            bad = ~s.dropna().astype(str).isin([str(v) for v in spec.allowed_values])
            n = int(bad.sum())
            if n:
                findings.append(
                    {
                        "type": "dictionary",
                        "severity": "notable",
                        "column": spec.name,
                        "detail": f"{n} valores fuera del catálogo permitido",
                        "value": n,
                    }
                )
    return {
        "ok": True,
        "dictionary": dd.name,
        "version": dd.version,
        "domain": dd.domain,
        "n_findings": len(findings),
        "n_critical": sum(1 for f in findings if f.get("severity") == "critical"),
        "findings": findings,
    }


def infer_dictionary_stub(df: pd.DataFrame, name: str = "auto", domain: str = "general") -> DataDictionary:
    cols = []
    for c in df.columns:
        s = df[c]
        spec = ColumnSpec(name=str(c), dtype=str(s.dtype))
        if pd.api.types.is_numeric_dtype(s):
            spec.min = float(pd.to_numeric(s, errors="coerce").min())
            spec.max = float(pd.to_numeric(s, errors="coerce").max())
            spec.role = "numeric"
        else:
            levels = s.dropna().astype(str).nunique()
            if levels <= 20:
                spec.allowed_values = sorted(s.dropna().astype(str).unique().tolist())
                spec.role = "categorical"
            else:
                spec.role = "text_or_id"
        cols.append(spec)
    return DataDictionary(name=name, version="0.1.0-draft", domain=domain, columns=cols)
