"""Carga de CSV/Excel con detección de separador/decimal (Excel AR/ES) y avisos."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import csv
import io
import re

import duckdb
import pandas as pd


PathLike = Union[str, Path]


def _is_excel(path: Path) -> bool:
    return path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}


def _sample_text(raw: bytes, n: int = 8192) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            return raw[:n].decode(enc)
        except Exception:
            continue
    return raw[:n].decode("utf-8", errors="replace")


def _sniff_csv_params(sample: str) -> dict:
    """Detecta separador y decimal típicos de Excel LATAM."""
    sep = ","
    decimal = "."
    warnings: list[str] = []

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter or ","
    except Exception:
        # Heurística: si hay más ; que , en la primera línea → ;
        first = sample.splitlines()[0] if sample else ""
        if first.count(";") > first.count(","):
            sep = ";"
        elif first.count("\t") > first.count(","):
            sep = "\t"

    # Decimal: si sep es ; y aparecen números tipo 1.234,56 o 12,5
    body = "\n".join(sample.splitlines()[1:40])
    comma_nums = len(re.findall(r"(?<![A-Za-z])\d+,\d+", body))
    dot_nums = len(re.findall(r"(?<![A-Za-z])\d+\.\d+", body))
    if sep == ";" and comma_nums >= max(1, dot_nums):
        decimal = ","
        warnings.append(
            "Detectamos separador ';' y decimal ',' (formato típico de Excel en español)."
        )
    elif sep == "," and comma_nums > dot_nums * 2 and comma_nums >= 3:
        # raro pero posible
        decimal = ","
        warnings.append("Detectamos muchos números con coma decimal.")

    if sep != ",":
        warnings.append(f"Separador de columnas detectado: '{sep}'.")

    return {"sep": sep, "decimal": decimal, "warnings": warnings}


def _post_load_checks(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    warnings = list(meta.get("warnings") or [])
    if df.shape[1] == 1:
        warnings.append(
            "Se leyó 1 sola columna. Si el archivo viene de Excel en español, "
            "probá que el separador sea ';' (la app intenta detectarlo sola)."
        )
        meta["suspicious_parse"] = True
    if df.shape[0] == 0:
        warnings.append("El archivo no tiene filas de datos.")
        meta["suspicious_parse"] = True
    # columnas Unnamed masivas
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if len(unnamed) >= max(2, int(0.5 * df.shape[1])):
        warnings.append("Muchas columnas sin nombre: el archivo puede no ser una tabla válida.")
        meta["suspicious_parse"] = True
    # todo texto en una sola celda enorme
    if df.shape[1] == 1 and df.shape[0] > 0:
        sample_val = str(df.iloc[0, 0])
        if len(sample_val) > 200 and ("," in sample_val or ";" in sample_val):
            warnings.append(
                "La única columna parece contener una fila entera sin separar. "
                "Revisá el formato del CSV."
            )
            meta["suspicious_parse"] = True
    meta["warnings"] = warnings
    meta["n_rows"], meta["n_cols"] = df.shape
    meta["columns"] = [str(c) for c in df.columns]
    return df


def _read_csv_bytes(raw: bytes) -> tuple[pd.DataFrame, dict]:
    sample = _sample_text(raw)
    params = _sniff_csv_params(sample)
    meta = {
        "method": "pandas",
        "format": "csv",
        "sep": params["sep"],
        "decimal": params["decimal"],
        "warnings": list(params["warnings"]),
    }
    buf = io.BytesIO(raw)
    thousands = "." if params["decimal"] == "," else None
    try:
        df = pd.read_csv(
            buf,
            sep=params["sep"],
            decimal=params["decimal"],
            thousands=thousands,
        )
    except Exception as exc:
        buf.seek(0)
        try:
            df = pd.read_csv(buf)
            meta["warnings"].append(
                f"Lectura con parámetros detectados falló ({exc}); se usó CSV por defecto."
            )
        except Exception as exc2:
            meta["ok"] = False
            meta["error"] = str(exc2)
            meta["suspicious_parse"] = True
            meta["warnings"].append(f"No se pudo leer el archivo como tabla: {exc2}")
            return pd.DataFrame(), meta
    return _post_load_checks(df, meta), meta


def _read_csv_path(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = path.read_bytes()
    size_mb = path.stat().st_size / (1024 * 1024)
    # Para archivos grandes con coma estándar, DuckDB sigue siendo útil
    sample = _sample_text(raw)
    params = _sniff_csv_params(sample)
    if size_mb > 20 and params["sep"] == "," and params["decimal"] == ".":
        try:
            con = duckdb.connect()
            p = str(path).replace("\\", "/").replace("'", "''")
            df = con.execute(f"SELECT * FROM read_csv_auto('{p}')").df()
            con.close()
            meta = {
                "method": "duckdb",
                "format": "csv",
                "sep": ",",
                "decimal": ".",
                "warnings": [],
            }
            return _post_load_checks(df, meta), meta
        except Exception:
            pass
    df, meta = _read_csv_bytes(raw)
    meta["method"] = "pandas"
    return df, meta


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
        DataFrame y metadatos (fuente, filas, columnas, método, warnings de parseo).
    """
    meta: dict = {"method": "pandas", "name": name or "dataset", "ok": True, "warnings": []}

    if isinstance(source, (bytes, io.BytesIO)):
        raw = source.getvalue() if isinstance(source, io.BytesIO) else source
        # Excel magic / OLE
        is_xlsx = raw[:2] == b"PK"
        is_xls = raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        if is_xlsx or is_xls:
            try:
                buf = io.BytesIO(raw)
                df = pd.read_excel(buf, sheet_name=sheet_name or 0)
                meta["format"] = "excel"
                meta["name"] = name or "uploaded_excel"
                return _post_load_checks(df, meta), meta
            except Exception as exc:
                meta["warnings"].append(f"No pareció Excel válido ({exc}); intentando CSV.")
        df, csv_meta = _read_csv_bytes(raw)
        meta.update(csv_meta)
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
        return _post_load_checks(df, meta), meta

    df, csv_meta = _read_csv_path(path)
    meta.update(csv_meta)
    meta["name"] = name or path.stem
    meta["path"] = str(path)
    meta["sampled"] = False
    if len(df) > max_sample_for_infer:
        meta["full_n_rows"] = len(df)
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
