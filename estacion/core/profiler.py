"""Perfilado completo de un DataFrame."""

from __future__ import annotations

import pandas as pd

from .type_inference import (
    infer_role,
    is_target_candidate,
    summarize_categorical,
    summarize_numeric,
    coerce_datetime_columns,
)
from .types import DatasetProfile, VariableProfile, VariableRole


def profile_dataframe(
    df: pd.DataFrame,
    name: str = "dataset",
    sample_for_heavy: int = 100_000,
) -> DatasetProfile:
    """Genera perfil estructural, tipológico y de calidad básica."""
    work = df
    if len(df) > sample_for_heavy:
        work = df.sample(sample_for_heavy, random_state=42)

    variables: list[VariableProfile] = []
    numeric_cols: list[str] = []
    cat_cols: list[str] = []
    id_cols: list[str] = []
    dt_cols: list[str] = []
    targets: list[str] = []
    quality: list[str] = []
    granularity: list[str] = []

    n_rows, n_cols = df.shape
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
    dup = int(df.duplicated().sum())
    dup_pct = dup / max(n_rows, 1)

    if dup_pct > 0.01:
        quality.append(f"Filas duplicadas: {dup} ({dup_pct:.1%})")

    for col in df.columns:
        s = df[col]
        role = infer_role(s, str(col))
        n_miss = int(s.isna().sum())
        miss_pct = n_miss / max(n_rows, 1)
        n_unique = int(s.nunique(dropna=True))
        unique_pct = n_unique / max(n_rows - n_miss, 1) if (n_rows - n_miss) else 0.0
        warnings: list[str] = []
        stats: dict = {}

        if miss_pct > 0.3:
            warnings.append(f"Alta proporción de faltantes ({miss_pct:.1%})")
            quality.append(f"Columna '{col}': {miss_pct:.1%} faltantes")

        if role == VariableRole.NUMERIC or (
            role == VariableRole.BINARY and pd.api.types.is_numeric_dtype(s)
        ):
            stats = summarize_numeric(work[col] if col in work.columns else s)
            if stats.get("skewness") is not None and abs(stats["skewness"]) > 2:
                warnings.append(f"Asimetría elevada (skew={stats['skewness']:.2f})")
            if stats.get("iqr_outlier_pct", 0) > 0.05:
                warnings.append(
                    f"Outliers IQR: {stats['iqr_outlier_pct']:.1%} de valores"
                )
            numeric_cols.append(str(col))
        elif role in {VariableRole.CATEGORICAL, VariableRole.BINARY}:
            stats = summarize_categorical(work[col] if col in work.columns else s)
            if stats.get("top_share", 0) > 0.9:
                warnings.append(
                    f"Categoría dominante ({stats['top_share']:.1%} en un solo nivel)"
                )
            cat_cols.append(str(col))
        elif role == VariableRole.DATETIME:
            # Perfil temporal ligero
            parsed = pd.to_datetime(s, errors="coerce")
            valid = parsed.dropna()
            stats = {
                "min": str(valid.min()) if len(valid) else None,
                "max": str(valid.max()) if len(valid) else None,
                "n_valid": int(len(valid)),
                "parse_fail_pct": float(1 - len(valid) / max(n_rows - n_miss, 1))
                if (n_rows - n_miss)
                else 0.0,
            }
            dt_cols.append(str(col))
        elif role == VariableRole.IDENTIFIER:
            id_cols.append(str(col))
            if unique_pct < 0.99:
                warnings.append("Marcada como ID pero no es única al 100%")
        elif role == VariableRole.CONSTANT:
            warnings.append("Variable constante (sin variabilidad)")
            quality.append(f"Columna constante: '{col}'")

        if is_target_candidate(str(col), role) and role not in {
            VariableRole.IDENTIFIER,
            VariableRole.DATETIME,
            VariableRole.CONSTANT,
        }:
            # Solo marcar como candidata si el nombre sugiere target
            from .type_inference import TARGET_NAME_RE

            if TARGET_NAME_RE.search(str(col)):
                targets.append(str(col))

        variables.append(
            VariableProfile(
                name=str(col),
                dtype=str(s.dtype),
                role=role,
                n_missing=n_miss,
                missing_pct=miss_pct,
                n_unique=n_unique,
                unique_pct=unique_pct,
                is_constant=role == VariableRole.CONSTANT,
                stats=stats,
                warnings=warnings,
            )
        )

    # Granularidad: IDs casi únicos → fila ≈ entidad
    for c in id_cols:
        u = df[c].nunique(dropna=True) / max(n_rows, 1)
        if u > 0.98:
            granularity.append(
                f"Granularidad aparente: 1 fila ≈ 1 entidad ({c} casi único)"
            )
        elif u < 0.5:
            granularity.append(
                f"Posible granularidad múltiple: '{c}' se repite (ratio único={u:.2f})"
            )

    if dt_cols:
        work2 = coerce_datetime_columns(df, dt_cols[:1])
        col = dt_cols[0]
        diffs = work2[col].sort_values().diff().dropna()
        if len(diffs):
            med = diffs.median()
            granularity.append(f"Espaciado temporal mediano en '{col}': {med}")

    # Si no hay targets por nombre, sugerir binarias / numéricas de alta varianza
    if not targets:
        for v in variables:
            if v.role == VariableRole.BINARY:
                targets.append(v.name)
                break

    return DatasetProfile(
        name=name,
        n_rows=n_rows,
        n_cols=n_cols,
        memory_mb=round(mem_mb, 3),
        variables=variables,
        duplicate_rows=dup,
        duplicate_pct=dup_pct,
        possible_targets=targets,
        datetime_columns=dt_cols,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        identifier_columns=id_cols,
        quality_issues=quality,
        granularity_notes=granularity,
    )
