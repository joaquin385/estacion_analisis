"""Carga de CSV/Excel con DuckDB cuando conviene."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import io

import duckdb
import pandas as pd


PathLike = Union[str, Path]


def _is_excel(path: Path) -> bool:
    return path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}


def load_table(
    source: PathLike | bytes | io.BytesIO,
    name: str | None = None,
    sheet_name: str | int | None = 0,
    max_sample_for_infer: int = 50_000,
) -> tuple[pd.DataFrame, dict]:
    """
    Carga un CSV o Excel a DataFrame.

    Returns
    -------
    df, meta
        DataFrame y metadatos (fuente, filas, columnas, método de carga).
    """
    meta: dict = {"method": "pandas", "name": name or "dataset"}

    if isinstance(source, (bytes, io.BytesIO)):
        buf = source if isinstance(source, io.BytesIO) else io.BytesIO(source)
        # Intentar Excel primero por magic bytes; si falla, CSV
        try:
            buf.seek(0)
            df = pd.read_excel(buf, sheet_name=sheet_name or 0)
            meta["format"] = "excel"
            meta["name"] = name or "uploaded_excel"
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf)
            meta["format"] = "csv"
            meta["name"] = name or "uploaded_csv"
        meta["n_rows"], meta["n_cols"] = df.shape
        return df, meta

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    meta["path"] = str(path)
    meta["name"] = name or path.stem

    if _is_excel(path):
        df = pd.read_excel(path, sheet_name=sheet_name or 0)
        meta["format"] = "excel"
    else:
        # DuckDB es más eficiente para CSV grandes
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 20:
                con = duckdb.connect()
                # escape path for SQL
                p = str(path).replace("\\", "/").replace("'", "''")
                df = con.execute(f"SELECT * FROM read_csv_auto('{p}')").df()
                con.close()
                meta["method"] = "duckdb"
            else:
                df = pd.read_csv(path)
            meta["format"] = "csv"
        except Exception:
            df = pd.read_csv(path)
            meta["format"] = "csv"
            meta["method"] = "pandas_fallback"

    meta["n_rows"], meta["n_cols"] = df.shape
    meta["sampled"] = False
    if len(df) > max_sample_for_infer:
        meta["full_n_rows"] = len(df)
        # No subsampleamos la carga; el perfilado puede muestrear internamente
    return df, meta


def load_multiple(paths: list[PathLike]) -> dict[str, pd.DataFrame]:
    """Carga varios archivos y retorna dict nombre -> DataFrame."""
    tables: dict[str, pd.DataFrame] = {}
    for p in paths:
        df, meta = load_table(p)
        tables[meta["name"]] = df
    return tables


def suggest_join_keys(
    tables: dict[str, pd.DataFrame],
    min_overlap: float = 0.3,
) -> list[dict]:
    """
    Sugiere posibles claves de unión entre tablas por nombre de columna
    y solapamiento de valores (muestreado).
    """
    suggestions: list[dict] = []
    names = list(tables.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cols_a = set(tables[a].columns.astype(str))
            cols_b = set(tables[b].columns.astype(str))
            common = cols_a & cols_b
            for col in common:
                sa = set(tables[a][col].dropna().astype(str).head(5000))
                sb = set(tables[b][col].dropna().astype(str).head(5000))
                if not sa or not sb:
                    continue
                overlap = len(sa & sb) / min(len(sa), len(sb))
                if overlap >= min_overlap:
                    suggestions.append(
                        {
                            "left": a,
                            "right": b,
                            "key": col,
                            "overlap": round(overlap, 4),
                        }
                    )
    suggestions.sort(key=lambda x: -x["overlap"])
    return suggestions
