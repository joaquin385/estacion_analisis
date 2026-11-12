"""Tipos y estructuras de datos del núcleo analítico."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional
import json


class VariableRole(str, Enum):
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    BINARY = "binary"
    TEXT = "text"
    CONSTANT = "constant"
    UNKNOWN = "unknown"
    TARGET_CANDIDATE = "target_candidate"


class FindingSeverity(str, Enum):
    INFO = "info"
    INTERESTING = "interesting"
    NOTABLE = "notable"
    CRITICAL = "critical"


class FindingCategory(str, Enum):
    QUALITY = "quality"
    DISTRIBUTION = "distribution"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    SEGMENT = "segment"
    ANOMALY = "anomaly"
    CAUSAL = "causal"
    STATISTICAL = "statistical"


@dataclass
class VariableProfile:
    name: str
    dtype: str
    role: VariableRole
    n_missing: int
    missing_pct: float
    n_unique: int
    unique_pct: float
    is_constant: bool = False
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d


@dataclass
class DatasetProfile:
    name: str
    n_rows: int
    n_cols: int
    memory_mb: float
    variables: list[VariableProfile]
    duplicate_rows: int
    duplicate_pct: float
    possible_targets: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    granularity_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": self.memory_mb,
            "variables": [v.to_dict() for v in self.variables],
            "duplicate_rows": self.duplicate_rows,
            "duplicate_pct": self.duplicate_pct,
            "possible_targets": self.possible_targets,
            "datetime_columns": self.datetime_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "identifier_columns": self.identifier_columns,
            "quality_issues": self.quality_issues,
            "granularity_notes": self.granularity_notes,
        }


@dataclass
class Finding:
    """Hallazgo priorizado con evidencia cuantitativa."""

    id: str
    title: str
    observation: str
    evidence: dict[str, Any]
    magnitude: str
    method: str
    alternative_explanations: list[str]
    follow_up: list[str]
    category: FindingCategory
    severity: FindingSeverity
    score: float  # prioridad 0-100
    variables: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    statistically_significant: Optional[bool] = None
    practically_important: Optional[bool] = None
    p_value: Optional[float] = None
    p_value_fdr: Optional[float] = None
    significant_fdr_global: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)
