"""Traducción a lenguaje humano de hallazgos y defaults de UI."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from estacion.core.types import DatasetProfile, Finding, FindingCategory, FindingSeverity


QUALITY_LIKE = {
    FindingCategory.QUALITY,
    FindingCategory.ANOMALY,
}

# Nombres amigables (es-AR) para columnas frecuentes en demos / retail.
COLUMN_FRIENDLY_ES = {
    "customer_size": "tamaño de cliente",
    "revenue": "facturación",
    "marketing_spend": "gasto en marketing",
    "discount": "descuento",
    "region": "región",
    "channel": "canal",
    "units": "unidades pedidas",
    "order_id": "nº de pedido",
    "date": "fecha",
    "fecha": "fecha",
}

DATASET_FRIENDLY_ES = {
    "commercial": "Ventas",
    "financial": "Finanzas",
    "operational": "Operaciones",
    "segments": "Segmentos",
    "causal_confounded": "Causal con confusor",
    "nonlinear_null": "Relación nula",
}

MONEY_NAME_HINTS = (
    "revenue",
    "factur",
    "precio",
    "price",
    "amount",
    "monto",
    "sales",
    "venta",
    "spend",
    "gasto",
    "cost",
    "ingreso",
)


@dataclass
class CopyContext:
    """Contexto opcional para textos ricos (sin inventar números)."""

    df: pd.DataFrame | None = None
    profile: DatasetProfile | None = None
    money_unit: str = "unidades"  # "$" | "unidades" | otra unidad del diccionario
    column_labels: dict[str, str] = field(default_factory=dict)
    dataset_label: str | None = None


_DEFAULT_CTX = CopyContext()

# Términos técnicos que no deben aparecer en el texto principal de las tarjetas.
JARGON_TERMS = (
    "IsolationForest",
    "Spearman",
    "Pearson",
    "Distance correlation",
    "silhouette",
    "K-means",
    "K-Means",
    "DBSCAN",
    "scatterplot",
    "Scatterplot",
    "residuales",
    "covariables",
    "contamination",
    "MCAR",
    "MAR",
    "MNAR",
    "p FDR",
    "FDR-BH",
    "ρ",
    "rho=",
)

# Fragmentos prohibidos en el resumen en lenguaje simple.
SUMMARY_FORBIDDEN = (
    "r=",
    "ρ",
    "p=",
    "p FDR",
    "efecto=",
    "max_",
    "default",
)


def format_int_es(n: Any) -> str:
    """Entero con separador de miles punto (es-AR): 3060 → 3.060."""
    if n is None:
        return "—"
    try:
        if isinstance(n, bool):
            return bool_es(n)
        v = int(n)
    except Exception:
        return str(n)
    neg = v < 0
    body = f"{abs(v):,}".replace(",", ".")
    return f"-{body}" if neg else body


def _format_es_float(v: float, decimals: int) -> str:
    """Float es-AR: miles con punto, decimales con coma."""
    neg = v < 0
    v = abs(float(v))
    if decimals <= 0:
        body = f"{int(round(v)):,}".replace(",", ".")
    else:
        raw = f"{v:,.{decimals}f}"  # 1,234.56
        if "." in raw:
            whole, frac = raw.rsplit(".", 1)
        else:
            whole, frac = raw, "0" * decimals
        body = f"{whole.replace(',', '.')},{frac}"
    return f"-{body}" if neg else body


def format_number(x: Any, digits: int = 2) -> str:
    """Número es-AR: miles con punto, decimales con coma. Enteros sin ,0."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return bool_es(x)
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
        # enteros / numpy int
        if isinstance(x, int) or (hasattr(x, "dtype") and str(getattr(x, "dtype", "")).startswith("int")):
            return format_int_es(x)
        v = float(x)
    except Exception:
        s = str(x)
        if s.lower() in {"none", "nan", "null"}:
            return "—"
        return s
    if abs(v - round(v)) < 1e-9:
        return format_int_es(int(round(v)))
    if abs(v) >= 1000:
        raw = _format_es_float(v, max(digits, 1))
        # quitar ceros sobrantes tras la coma (759.773,50 → 759.773,5)
        if "," in raw:
            whole, frac = raw.rsplit(",", 1)
            frac = frac.rstrip("0")
            raw = f"{whole},{frac}" if frac else whole
        return raw
    d = max(digits, 2) if abs(v) < 10 else digits
    raw = f"{v:.{d}f}"
    if abs(v) < 10 and d >= 2:
        raw = raw.rstrip("0").rstrip(".")
        if "." not in raw and digits >= 1:
            raw = f"{v:.0f}"
    return raw.replace(".", ",")


def bool_es(x: Any) -> str:
    if x is True:
        return "Sí"
    if x is False:
        return "No"
    return "—"


CATEGORY_LABELS_ES = {
    "quality": "Calidad",
    "distribution": "Distribución",
    "relationship": "Relación",
    "temporal": "Temporal",
    "segment": "Segmentación",
    "anomaly": "Anomalía",
    "causal": "Causal",
    "statistical": "Estadística",
}

SEVERITY_LABELS_ES = {
    "info": "Info",
    "interesting": "Interesante",
    "notable": "Notable",
    "critical": "Crítico",
}


def format_pct(x: float | None, digits: int = 0) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    # aceptar 0–1 o ya en porcentaje
    if abs(v) <= 1.0:
        v = 100.0 * v
    if digits == 0:
        return f"{v:.0f} %"
    return f"{v:.{digits}f} %".replace(".", ",")


def format_p_value(p: float | None) -> str:
    if p is None:
        return "—"
    try:
        if isinstance(p, float) and math.isnan(p):
            return "—"
    except Exception:
        return "—"
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


def format_p_clause(p: float | None) -> str:
    """'p = 0,012' o 'p < 0,001' (sin 'p = < …')."""
    txt = format_p_value(p)
    if txt == "—":
        return "p = —"
    if txt.startswith("<"):
        return f"p {txt}"
    return f"p = {txt}"


def format_n_rows(n: int) -> str:
    n = int(n)
    return "1 fila" if n == 1 else f"{format_int_es(n)} filas"


def format_memory_mb(mb: Any) -> str:
    """Memoria en MB con coma decimal (es-AR): 0.217 → 0,217 MB."""
    try:
        v = float(mb)
    except Exception:
        return f"{mb} MB"
    if abs(v - round(v)) < 1e-9:
        return f"{format_int_es(int(round(v)))} MB"
    raw = f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{raw.replace('.', ',')} MB"


def friendly_column(name: str, ctx: CopyContext | None = None) -> str:
    if not name:
        return "una columna"
    key = str(name)
    if ctx and ctx.column_labels.get(key):
        return ctx.column_labels[key]
    return COLUMN_FRIENDLY_ES.get(key, COLUMN_FRIENDLY_ES.get(key.lower(), key))


def friendly_dataset_name(name: str | None) -> str:
    if not name:
        return "Datos"
    key = str(name).strip()
    stem = Path(key).stem if "/" in key or "\\" in key or key.endswith(".csv") else key
    return DATASET_FRIENDLY_ES.get(stem, DATASET_FRIENDLY_ES.get(stem.lower(), stem.replace("_", " ").capitalize()))


def looks_like_money_column(name: str | None) -> bool:
    if not name:
        return False
    low = str(name).lower()
    return any(h in low for h in MONEY_NAME_HINTS)


def resolve_money_unit(
    *,
    column: str | None = None,
    dictionary_units: dict[str, str] | None = None,
    override: str | None = None,
) -> str:
    """
    Unidad para montos: override de UI > diccionario > 'unidades'.
    Si override es explícito ('$' o 'unidades'), gana siempre.
    No inventa '$' si no viene del diccionario/override.
    """
    if override is not None and str(override).strip() and str(override).strip() != "Auto":
        return str(override).strip()
    if dictionary_units:
        if column:
            u = dictionary_units.get(column) or dictionary_units.get(str(column).lower())
            if u and str(u).strip():
                return str(u).strip()
        # Auto: primera unidad monetaria del diccionario
        for u in dictionary_units.values():
            if u and str(u).strip() in {"$", "USD", "ARS", "u$s", "U$S"}:
                return str(u).strip()
    return "unidades"


# Stems exactos que mapean a un diccionario (no substring "venta" en cualquier archivo)
_DICT_STEM_ALIASES = {
    "commercial": "commercial",
    "ventas": "commercial",
    "financial": "finance",
    "finance": "finance",
    "finanzas": "finance",
    "retail": "retail",
    "healthcare": "healthcare",
    "manufacturing": "manufacturing",
}


def load_dictionary_units(path: str | Path | None = None, name_hint: str | None = None) -> dict[str, str]:
    """Carga {columna: unit} solo con match exacto de stem (no por substring)."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    root = Path(__file__).resolve().parents[2]
    dict_dir = root / "config" / "dictionaries"
    if name_hint:
        stem = Path(str(name_hint)).stem.lower().replace("-", "_")
        alias = _DICT_STEM_ALIASES.get(stem)
        if alias:
            candidates.append(dict_dir / f"{alias}_v1.yaml")
            candidates.append(dict_dir / f"{alias}_v1.json")
        # stem literal si existe archivo homónimo
        candidates.append(dict_dir / f"{stem}_v1.yaml")
        candidates.append(dict_dir / f"{stem}_v1.json")
    out: dict[str, str] = {}
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if not p.exists():
            continue
        try:
            from estacion.plugins.dictionaries import load_dictionary

            dd = load_dictionary(p)
            for c in dd.columns:
                if c.unit:
                    out[c.name] = str(c.unit)
            if out:
                return out
        except Exception:
            continue
    return out


def format_amount(value: Any, unit: str = "unidades", *, digits: int | None = None) -> str:
    """Formatea un monto con unidad: '$430' o '430 unidades'. Sin decimales si digits=0."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if digits is None:
        digits = 0 if abs(v) >= 20 else 2
    if digits <= 0:
        num = format_int_es(int(round(v)))
    else:
        num = format_number(v, digits=digits)
    u = (unit or "unidades").strip()
    if u in {"$", "USD", "ARS", "u$s", "U$S"}:
        return f"${num}"
    if u.lower() in {"unidades", "unidad", "units"}:
        return f"{num} unidades"
    return f"{num} {u}"


def money_unit_for_finding(f: Finding, ctx: CopyContext | None = None) -> str:
    ctx = ctx or _DEFAULT_CTX
    # Preferir variable numérica tipo monto
    candidates = list(f.variables or [])
    ev = f.evidence or {}
    for k in ("var_b", "var_a", "column"):
        if ev.get(k):
            candidates.append(str(ev[k]))
    for c in candidates:
        if looks_like_money_column(c):
            return ctx.money_unit if ctx.money_unit else "unidades"
    # Si el money_unit del contexto es $ pero la columna no parece dinero, igual
    # usamos unidades para no poner $ en customer_size etc.
    return "unidades"


def display_value(x: Any, *, pct_if_unit: bool = False) -> Any:
    """None/nan → '—'; enteros/floats en es-AR."""
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    if isinstance(x, bool):
        return bool_es(x)
    if isinstance(x, int) and not isinstance(x, bool):
        return format_int_es(x)
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return "—"
        if pct_if_unit and abs(x) <= 1.0:
            return format_pct(x, digits=1)
        return format_number(x, digits=4 if abs(x) < 1 else 2)
    # numpy enteros
    try:
        if hasattr(x, "item") and type(x).__name__ in {"int64", "int32", "int16", "uint64"}:
            return format_int_es(int(x))
    except Exception:
        pass
    if isinstance(x, str) and x.lower() in {"none", "nan", "null"}:
        return "—"
    return x


def dataframe_for_display(
    df: pd.DataFrame,
    *,
    percent_cols: set[str] | None = None,
    format_numbers: bool = False,
) -> pd.DataFrame:
    """Copia amigable: None/nan/NaT → —, % unitarios, sin bool crudos feos."""
    if df is None or getattr(df, "empty", True):
        return df
    percent_cols = {c.lower() for c in (percent_cols or set())}
    out = df.copy()
    for col in out.columns:
        is_pct = str(col).lower() in percent_cols or "pct" in str(col).lower() or "%" in str(col)

        def _cell(v, p=is_pct):
            if v is None:
                return "—"
            try:
                if pd.isna(v):
                    return "—"
            except Exception:
                pass
            if isinstance(v, str) and v.lower() in {"none", "nan", "null"}:
                return "—"
            if isinstance(v, bool):
                return bool_es(v)
            if format_numbers and isinstance(v, (int, float)) and not isinstance(v, bool):
                return display_value(v, pct_if_unit=p)
            if p and isinstance(v, (int, float)) and not isinstance(v, bool):
                return format_pct(float(v), digits=1)
            if isinstance(v, int) and not isinstance(v, bool):
                return format_int_es(v)
            if isinstance(v, float):
                return format_number(v, digits=4 if abs(v) < 1 else 2)
            try:
                if hasattr(v, "item") and "int" in type(v).__name__.lower():
                    return format_int_es(int(v))
            except Exception:
                pass
            return v

        out[col] = out[col].map(_cell)
    return out


def group_vs_rest_gap(f: Finding) -> dict[str, Any] | None:
    """
    Brecha extremo vs resto: promedio ponderado por tamaño de grupo cuando hay
    group_sizes (promedio real de las filas del resto). Si no hay tamaños,
    cae al promedio simple de medias de grupo.
    """
    ev = f.evidence or {}
    means = ev.get("group_means")
    if not isinstance(means, dict) or len(means) < 2:
        return None
    items = [(str(k), float(v)) for k, v in means.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    top_name, top_val = items[0]
    sizes = ev.get("group_sizes")
    rest_items = items[1:]
    if isinstance(sizes, dict) and all(str(k) in sizes for k, _ in rest_items):
        w_sum = 0.0
        n_sum = 0.0
        for k, v in rest_items:
            n = float(sizes[str(k)])
            w_sum += v * n
            n_sum += n
        rest_avg = w_sum / n_sum if n_sum else 0.0
    else:
        rest_avg = sum(v for _, v in rest_items) / len(rest_items)
    pct = None if rest_avg == 0 else 100.0 * (top_val - rest_avg) / abs(rest_avg)
    var = f.variables[0] if f.variables else "valor"
    return {
        "top_name": top_name,
        "top_val": top_val,
        "rest_avg": rest_avg,
        "pct": pct,
        "var": var,
    }


def format_magnitude(mag: Any, finding: Finding | None = None) -> str:
    """Magnitud legible para detalle técnico / informes."""
    if finding is not None and finding.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(finding)
        if gap and gap.get("pct") is not None:
            return f"~{abs(gap['pct']):.0f} % vs el resto"

    if mag is None:
        return "—"
    if isinstance(mag, (int, float)):
        try:
            if isinstance(mag, float) and math.isnan(mag):
                return "—"
        except Exception:
            return "—"
        if abs(float(mag)) <= 1.0:
            return format_pct(float(mag), digits=1)
        return format_number(mag, digits=2)
    s = str(mag).strip()
    if not s or s.lower() in {"none", "nan"}:
        return "—"

    # Patrones típicos del motor
    m = re.match(r"^\|?[ρrR]\|?\s*=\s*([0-9.eE+-]+)$", s) or re.match(
        r"^\|ρ\|?\s*=\s*([0-9.eE+-]+)$", s
    )
    if m:
        return f"relación ≈ {format_number(float(m.group(1)))}"
    m = re.match(r"^\|?skew\|?\s*=\s*([0-9.eE+-]+)$", s, re.I)
    if m:
        return f"asimetría ≈ {format_number(float(m.group(1)))}"
    m = re.match(r"^silhouette\s*=\s*([0-9.eE+-]+)$", s, re.I)
    if m:
        return f"calidad de grupos ≈ {format_number(float(m.group(1)))}"
    m = re.match(r"^brecha relativa\s*≈\s*([0-9.eE+-]+)\s*%?$", s, re.I)
    if m and finding is not None:
        gap = group_vs_rest_gap(finding)
        if gap and gap.get("pct") is not None:
            return f"~{abs(gap['pct']):.0f} % vs el resto"
    m = re.match(r"^([0-9.eE+-]+)\s*%?\s*(duplicadas|faltantes|anomalías|extremos.*)?$", s, re.I)
    if m and re.search(r"%", s):
        # ya viene tipo "2.0% duplicadas"
        return s.replace("%", " %")

    # float crudo
    try:
        v = float(s)
        if abs(v) <= 1.0 and "ρ" not in s and "|" not in s and "brecha" not in s.lower():
            return format_pct(v, digits=1)
        return format_number(v, digits=2)
    except Exception:
        # limpiar brecha del motor si no pudimos recalcular
        if "brecha relativa" in s.lower() and finding is not None:
            gap = group_vs_rest_gap(finding)
            if gap and gap.get("pct") is not None:
                return f"~{abs(gap['pct']):.0f} % vs el resto"
        return s


def evidence_pairs_plain(ev: dict[str, Any] | None, max_items: int = 6) -> list[tuple[str, str]]:
    """Pares clave/valor redondeados para el informe (sin dump crudo)."""
    if not ev:
        return []
    labels = {
        "spearman_r": "correlación",
        "pearson_r": "correlación lineal",
        "distance_corr": "dependencia",
        "missing_pct": "% vacíos",
        "duplicate_pct": "% duplicados",
        "duplicate_rows": "filas repetidas",
        "n_missing": "nº vacíos",
        "n_anomalies": "filas raras",
        "pct": "%",
        "effect_size": "efecto",
        "effect_size_cohens_d": "d de Cohen",
        "relative_mean_gap": "brecha (max−min / media)",
        "n": "n",
        "column": "columna",
        "var_a": "variable A",
        "var_b": "variable B",
        "rule": "regla",
        "value": "valor",
        "threshold": "umbral",
        "skewness": "asimetría",
        "mean": "promedio",
        "median": "mediana",
    }
    skip = {
        "group_means",
        "group_sizes",
        "anomaly_indices",
        "limitations",
        "ok",
        "sampled",
        "score_min",
        "score_median",
        "nonlinear_hint",
        "significant_fdr",
        "practically_important",
        "best",
        "caveat",
        "method",
        "detail",
        "severity",
        "role",
        "pearson_p",
        "spearman_p",
        "spearman_p_fdr",
        "p_value",
    }
    out: list[tuple[str, str]] = []
    if isinstance(ev.get("group_means"), dict):
        items = list(ev["group_means"].items())[:4]
        out.append(
            (
                "medias por grupo",
                " · ".join(f"{k}: {format_number(v)}" for k, v in items),
            )
        )
    for k, v in ev.items():
        if k in skip or isinstance(v, (dict, list)):
            continue
        label = labels.get(k, k)
        if k in {"missing_pct", "duplicate_pct"} or (k == "pct" and isinstance(v, (int, float)) and float(v) <= 1):
            out.append((label, format_pct(float(v), digits=1)))
        elif k == "pct" and isinstance(v, (int, float)) and float(v) > 1:
            out.append((label, f"{format_number(v, 0)} %"))
        elif isinstance(v, float):
            out.append((label, format_number(v, digits=4 if abs(v) < 1 else 2)))
        else:
            out.append((label, str(display_value(v))))
        if len(out) >= max_items:
            break
    return out


def relationship_confirmed(f: Finding) -> bool:
    ev = f.evidence or {}
    if ev.get("significant_fdr") is True:
        return True
    if f.statistically_significant is True:
        return True
    if f.significant_fdr_global is True:
        return True
    return False


def finding_track(f: Finding) -> str:
    """problemas | descubrimientos"""
    if f.category in QUALITY_LIKE:
        return "problemas"
    return "descubrimientos"


def severity_label_es(f: Finding) -> tuple[str, str]:
    """
    Returns (etiqueta, color_hint)
    Problemas: Revisar / Atención / Info
    Descubrimientos: Muy confiable / Confiable / Para mirar / Info
    """
    track = finding_track(f)
    sev = f.severity
    if track == "problemas":
        if sev == FindingSeverity.CRITICAL:
            return "Revisar", "red"
        if sev == FindingSeverity.NOTABLE:
            return "Atención", "orange"
        if sev == FindingSeverity.INTERESTING:
            return "Atención", "orange"
        return "Info datos", "gray"
    # Relaciones sin respaldo FDR → Info (no “se mueven juntos”)
    if f.category == FindingCategory.RELATIONSHIP and not relationship_confirmed(f):
        return "Info", "gray"
    # descubrimientos (tono maqueta A: "Muy confiable")
    if sev == FindingSeverity.CRITICAL or (
        sev == FindingSeverity.NOTABLE and f.practically_important
    ):
        return "Muy confiable", "teal"
    if sev == FindingSeverity.NOTABLE:
        return "Confiable", "teal"
    if sev == FindingSeverity.INTERESTING:
        return "Para mirar", "blue"
    return "Info", "gray"


def category_pill_es(f: Finding) -> str:
    if f.category == FindingCategory.STATISTICAL:
        return "Oportunidad"
    if f.category == FindingCategory.RELATIONSHIP:
        return "Relación"
    if f.category == FindingCategory.QUALITY:
        return "Calidad"
    if f.category == FindingCategory.ANOMALY:
        return "Anomalía"
    if f.category == FindingCategory.DISTRIBUTION:
        return "Distribución"
    if f.category == FindingCategory.SEGMENT:
        return "Segmentos"
    if f.category == FindingCategory.TEMPORAL:
        return "Temporal"
    if f.category == FindingCategory.CAUSAL:
        return "Causal"
    return "Hallazgo"


def _strength_word(r: float | None) -> str:
    if r is None:
        return "moderada"
    a = abs(float(r))
    if a < 0.25:
        return "débil"
    if a < 0.45:
        return "moderada"
    return "fuerte"


def _problem_dedupe_key(f: Finding) -> tuple:
    """Agrupa problemas que hablan de lo mismo (p.ej. duplicados)."""
    ev = f.evidence or {}
    title_l = (f.title or "").lower()
    obs_l = (f.observation or "").lower()
    rule = str(ev.get("rule") or "")
    if (
        "duplic" in title_l
        or "duplic" in obs_l
        or rule == "max_duplicate_pct"
        or "duplicate_pct" in ev
    ):
        return ("duplicates",)
    if f.category == FindingCategory.ANOMALY or "anomal" in title_l:
        return ("anomaly",)
    if "faltant" in title_l or "vacío" in obs_l or "missing" in title_l or "missing_pct" in ev:
        var = (f.variables[0] if f.variables else ev.get("column") or "")
        return ("missing", str(var))
    if "outlier" in title_l or "atípic" in title_l or "extremos" in title_l:
        var = f.variables[0] if f.variables else ""
        return ("outlier", str(var))
    vars_key = tuple(sorted(str(v) for v in (f.variables or [])))
    return (f.category.value, f.title, vars_key)


def _evidence_richness(f: Finding) -> tuple:
    """Para desempate: preferir evidencia con números útiles."""
    ev = f.evidence or {}
    return (
        1 if ev.get("duplicate_rows") is not None else 0,
        1 if isinstance(ev.get("group_means"), dict) else 0,
        len(ev),
        float(f.score or 0),
    )


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Quita tarjetas repetidas; fusiona evidencia útil del duplicado descartado."""
    from dataclasses import replace

    best: dict[tuple, Finding] = {}
    ranked = sorted(findings, key=lambda f: (-_evidence_richness(f)[0], -_evidence_richness(f)[1], -f.score))
    for f in ranked:
        if finding_track(f) == "problemas":
            key = _problem_dedupe_key(f)
        else:
            key = ("disc", f.category.value, tuple(sorted(f.variables or [])), f.title)
        prev = best.get(key)
        if prev is None:
            best[key] = f
            continue
        # Fusionar evidencia: keys del más rico + del otro
        if _evidence_richness(f) > _evidence_richness(prev):
            rich, poor = f, prev
        else:
            rich, poor = prev, f
        rich_ev = dict(rich.evidence or {})
        for k, v in (poor.evidence or {}).items():
            rich_ev.setdefault(k, v)
        best[key] = replace(
            rich,
            evidence=rich_ev,
            score=max(rich.score, poor.score),
            limitations=rich.limitations or poor.limitations,
            alternative_explanations=rich.alternative_explanations or poor.alternative_explanations,
        )
    return sort_findings_for_humans(list(best.values()))


def _approx_num(v: float) -> str:
    if abs(v) >= 20:
        return f"{v:.0f}"
    return format_number(v)


def _pretty_group(name: str) -> str:
    """Nombre de grupo sin comillas; capitaliza la primera letra."""
    s = str(name or "").strip()
    if not s:
        return s
    return s[:1].upper() + s[1:]


_GROUP_PLURAL_ES = {
    "channel": "canales",
    "canal": "canales",
    "region": "regiones",
    "región": "regiones",
}


def _group_plural(group_var: str, group_label: str) -> str:
    key = str(group_var or "").lower()
    if key in _GROUP_PLURAL_ES:
        return _GROUP_PLURAL_ES[key]
    # etiqueta amigable conocida
    low = str(group_label or "").lower()
    if low in _GROUP_PLURAL_ES:
        return _GROUP_PLURAL_ES[low]
    return f"niveles de {group_label}"


def _share_phrase(share: float, noun: str) -> str | None:
    """Frase de participación; 'más de la mitad' si ≥55 %."""
    # noun: "los pedidos" | "la facturación"
    if share >= 0.55:
        return f"más de la mitad de {noun}"
    if abs(share - 0.5) < 0.05:
        return f"la mitad de {noun}"
    if abs(share - 0.4) < 0.06 and "pedido" in noun:
        return "4 de cada 10 pedidos"
    if abs(share - 1 / 3) < 0.05 and "pedido" in noun:
        return "1 de cada 3 pedidos"
    if 0.15 <= share <= 0.85:
        return f"el {format_pct(share)} de {noun}"
    return None


def _group_share_phrases(f: Finding, top_name: str) -> list[str]:
    """Peso del grupo (filas / facturación) con números ya en evidence."""
    ev = f.evidence or {}
    sizes = ev.get("group_sizes")
    if not isinstance(sizes, dict) or top_name not in sizes:
        return []
    total = sum(int(v) for v in sizes.values()) or 1
    row_share = int(sizes[top_name]) / total
    bits: list[str] = []
    row_bit = _share_phrase(row_share, "los pedidos")
    if row_bit:
        bits.append(row_bit)
    means = ev.get("group_means")
    value_col = f.variables[0] if f.variables else None
    if isinstance(means, dict) and value_col and looks_like_money_column(value_col):
        weighted = {k: float(means.get(k, 0)) * float(sizes.get(k, 0)) for k in sizes}
        tot_w = sum(weighted.values()) or 1.0
        rev_share = weighted.get(top_name, 0) / tot_w
        rev_bit = _share_phrase(rev_share, "la facturación")
        if rev_bit:
            bits.append(rev_bit)
    return bits


def _group_vs_rest_phrase(f: Finding, ctx: CopyContext | None = None) -> str | None:
    """Frase clásica para resumen/informe (mantiene ~52 % para tests)."""
    gap = group_vs_rest_gap(f)
    if not gap:
        return None
    top_name = _pretty_group(gap["top_name"])
    top_val, rest_avg, pct, var = (
        gap["top_val"],
        gap["rest_avg"],
        gap["pct"],
        gap["var"],
    )
    var_l = friendly_column(str(var), ctx)
    if pct is None:
        return (
            f"{top_name} tiene un promedio de {var_l} de ≈{_approx_num(top_val)} "
            f"(resto ≈{_approx_num(rest_avg)})"
        )
    direction = "más alto" if pct >= 0 else "más bajo"
    return (
        f"{top_name} tiene un promedio de {var_l} ~{abs(pct):.0f} % {direction} "
        f"que el resto (≈{_approx_num(top_val)} vs ≈{_approx_num(rest_avg)})"
    )


def _duplicate_n_pct(f: Finding, ctx: CopyContext | None = None) -> tuple[int | None, float | None]:
    ev = f.evidence or {}
    n = ev.get("duplicate_rows")
    pct = ev.get("duplicate_pct")
    if pct is None and ev.get("value") is not None and (
        "duplic" in str(ev.get("rule") or "").lower() or "duplic" in (f.title or "").lower()
    ):
        pct = ev.get("value")
    if pct is None and ctx and ctx.profile is not None and ctx.profile.duplicate_pct:
        pct = ctx.profile.duplicate_pct
    if n is None and pct is not None and ctx and ctx.profile is not None and ctx.profile.n_rows:
        n = int(round(float(pct) * ctx.profile.n_rows))
    if n is None and ctx and ctx.profile is not None and ctx.profile.duplicate_rows:
        n = int(ctx.profile.duplicate_rows)
    return (int(n) if n is not None else None, float(pct) if pct is not None else None)


def _duplicate_inflated_total(f: Finding, ctx: CopyContext | None = None) -> float | None:
    """Suma del valor target en filas duplicadas (keep='first' de duplicadas). Solo con df."""
    if not ctx or ctx.df is None or ctx.profile is None:
        return None
    df = ctx.df
    targets = list(ctx.profile.possible_targets or [])
    num_cols = [c for c in (targets or ctx.profile.numeric_columns or []) if c in df.columns]
    if not num_cols:
        return None
    col = next((c for c in num_cols if looks_like_money_column(c)), num_cols[0])
    try:
        extra = df[df.duplicated(keep="first")][col]
        return float(extra.sum())
    except Exception:
        return None


def _plain_bullet_for_finding(f: Finding, ctx: CopyContext | None = None) -> str | None:
    """Una frase 100% humana para el resumen."""
    ev = f.evidence or {}
    ctx = ctx or _DEFAULT_CTX
    if f.category == FindingCategory.RELATIONSHIP:
        a = ev.get("var_a") or (f.variables[0] if f.variables else "X")
        b = ev.get("var_b") or (f.variables[1] if len(f.variables) > 1 else "Y")
        r = ev.get("spearman_r")
        if r is None:
            r = ev.get("pearson_r")
        confirmed = relationship_confirmed(f)
        if not confirmed:
            if ev.get("nonlinear_hint"):
                return f"Posible relación no lineal entre '{a}' y '{b}' (no confirmada)"
            return f"Posible relación entre '{a}' y '{b}' (no confirmada)"
        if ev.get("nonlinear_hint") and abs(float(r or 0)) < 0.15:
            return (
                f"Entre '{a}' y '{b}' hay una relación que no es una línea simple "
                "(vale mirar el gráfico en Explorar)"
            )
        strength = _strength_word(float(r) if r is not None else None)
        if r is not None and float(r) < 0:
            return f"Cuando sube '{a}', tiende a bajar '{b}' (relación {strength})"
        return f"Cuando sube '{a}', tiende a subir '{b}' (relación {strength})"

    if f.category == FindingCategory.STATISTICAL:
        phrase = _group_vs_rest_phrase(f, ctx)
        if phrase:
            return phrase
        return _simplify_title(f, ctx)

    if f.category == FindingCategory.QUALITY:
        if "duplicate_pct" in ev or "duplic" in (f.title or "").lower() or ev.get("rule") == "max_duplicate_pct":
            n, pct = _duplicate_n_pct(f, ctx)
            if n is not None and pct is not None:
                return f"{int(n)} filas están repetidas ({format_pct(float(pct))})"
            if pct is not None:
                return f"Hay filas repetidas ({format_pct(float(pct))})"
            return "Hay filas repetidas en la planilla"
        if "missing_pct" in ev or "faltant" in (f.title or "").lower():
            col = ev.get("column") or (f.variables[0] if f.variables else "una columna")
            pct = float(ev.get("missing_pct") or 0)
            if pct >= 0.15:
                every = max(2, int(round(1 / pct))) if pct > 0 else 0
                return f"A '{col}' le falta el dato en 1 de cada {every} filas"
            return f"A '{col}' le faltan datos ({format_pct(pct)})"
        return None

    if f.category == FindingCategory.ANOMALY:
        n = ev.get("n_anomalies")
        pct = ev.get("pct")
        if n is not None:
            pct_txt = f" ({format_pct(float(pct) / 100 if float(pct) > 1 else float(pct))})" if pct is not None else ""
            if pct is not None and float(pct) > 1:
                pct_txt = f" ({format_number(pct, 0)} %)"
            return f"{int(n)} filas están fuera de lo común{pct_txt}"
        return "Hay filas fuera de lo común"

    if f.category == FindingCategory.DISTRIBUTION:
        var = f.variables[0] if f.variables else "una variable"
        return f"'{var}' tiene muchos valores extremos (cola larga); es frecuente en montos"

    if f.category == FindingCategory.SEGMENT:
        return "No aparecen grupos naturales claros entre las filas"

    return None


def _evidence_snippet(f: Finding) -> str:
    """Snippet solo para detalle técnico (puede tener números)."""
    ev = f.evidence or {}
    bits = []
    if "group_means" in ev and isinstance(ev["group_means"], dict):
        items = list(ev["group_means"].items())[:4]
        bits.append(" · ".join(f"{k}: {format_number(v)}" for k, v in items))
    if "spearman_r" in ev:
        bits.append(f"correlación={format_number(ev['spearman_r'])}")
    if "pearson_r" in ev:
        bits.append(f"r={format_number(ev['pearson_r'])}")
    if "missing_pct" in ev:
        bits.append(f"{format_pct(float(ev['missing_pct']))} vacíos")
    if "duplicate_pct" in ev:
        bits.append(f"{format_pct(float(ev['duplicate_pct']))} duplicados")
    if "effect_size" in ev:
        bits.append(f"efecto={format_number(ev['effect_size'])}")
    if "effect_size_cohens_d" in ev:
        bits.append(f"d={format_number(ev['effect_size_cohens_d'])}")
    return "; ".join(bits)


def _binned_means_phrase(df: pd.DataFrame, x_col: str, y_col: str, unit: str) -> str | None:
    """De tamaño 1 a 5, el promedio sube de A a B — solo con datos reales."""
    try:
        g = df[[x_col, y_col]].dropna()
        if g.empty or g[x_col].nunique() < 3:
            return None
        means = g.groupby(x_col, observed=True)[y_col].mean().sort_index()
        if len(means) < 3:
            return None
        first_k, first_v = means.index[0], float(means.iloc[0])
        last_k, last_v = means.index[-1], float(means.iloc[-1])
        if first_v == 0:
            return None
        pct = 100.0 * (last_v - first_v) / abs(first_v)
        return (
            f"De {friendly_column(x_col)} {first_k} a {last_k}, el promedio sube de "
            f"{format_amount(first_v, unit, digits=0)} a {format_amount(last_v, unit, digits=0)} "
            f"({'+' if pct >= 0 else ''}{pct:.0f} %)."
        )
    except Exception:
        return None


def _missing_uniform_across(df: pd.DataFrame, col: str, by: str) -> bool | None:
    try:
        if col not in df.columns or by not in df.columns:
            return None
        rates = df.groupby(by, observed=True)[col].apply(lambda s: float(s.isna().mean()))
        if len(rates) < 2:
            return None
        return float(rates.max() - rates.min()) < 0.08
    except Exception:
        return None


def _human_what(f: Finding, ctx: CopyContext | None = None) -> str:
    ev = f.evidence or {}
    ctx = ctx or _DEFAULT_CTX
    plain = _plain_bullet_for_finding(f, ctx)
    unit = ctx.money_unit or "unidades"

    if f.category == FindingCategory.RELATIONSHIP:
        a = ev.get("var_a") or (f.variables[0] if f.variables else "X")
        b = ev.get("var_b") or (f.variables[1] if len(f.variables) > 1 else "Y")
        a_l, b_l = friendly_column(str(a), ctx), friendly_column(str(b), ctx)
        if not relationship_confirmed(f):
            return f"Hay una posible relación entre {a_l} y {b_l}, todavía no confirmada."
        if ctx.df is not None and str(a) in ctx.df.columns and str(b) in ctx.df.columns:
            y_unit = unit if looks_like_money_column(str(b)) else "unidades"
            binned = _binned_means_phrase(ctx.df, str(a), str(b), y_unit)
            if binned:
                return binned + " La tendencia es clara, pero no perfecta."
        r = ev.get("spearman_r")
        strength = _strength_word(float(r) if r is not None else None)
        if r is not None and float(r) < 0:
            return f"Cuando sube {a_l}, {b_l} tiende a bajar (relación {strength})."
        return f"Cuando sube {a_l}, {b_l} tiende a subir (relación {strength})."

    if f.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(f)
        if gap and gap.get("pct") is not None:
            var = gap["var"]
            u = unit if looks_like_money_column(var) else "unidades"
            top = format_amount(gap["top_val"], u, digits=0)
            rest = format_amount(gap["rest_avg"], u, digits=0)
            raw_name = str(gap["top_name"])
            name = _pretty_group(raw_name)
            means = (f.evidence or {}).get("group_means") or {}
            others = (
                [_pretty_group(k) for k in means if str(k) != raw_name]
                if isinstance(means, dict)
                else []
            )
            others_txt = ", ".join(others) if others else "el resto"
            if looks_like_money_column(var):
                body = (
                    f"Un pedido promedio de {name} deja {top}; "
                    f"en {others_txt} ronda {rest}."
                )
            else:
                body = (
                    f"El promedio de {friendly_column(var, ctx)} en {name} es {top}; "
                    f"en {others_txt} ronda {rest}."
                )
            shares = _group_share_phrases(f, raw_name)
            if shares:
                # verbo explícito para no quedar "Además, 4 de cada 10 pedidos y…"
                if len(shares) == 1:
                    body += f" Además, concentra {shares[0]}."
                else:
                    body += f" Además, concentra {shares[0]} y {shares[1]}."
            return body
        phrase = _group_vs_rest_phrase(f, ctx)
        if phrase:
            return phrase + ("." if not phrase.endswith(".") else "")

    if f.category == FindingCategory.ANOMALY:
        n = ev.get("n_anomalies")
        pct = ev.get("pct")
        if n is not None and pct is not None:
            pct_v = float(pct)
            pct_txt = f"{pct_v:.0f} %" if pct_v > 1 else format_pct(pct_v)
            return f"Detectamos {int(n)} filas fuera de lo común ({pct_txt})."
        if n is not None:
            return f"Detectamos {int(n)} filas fuera de lo común."

    if f.category == FindingCategory.QUALITY:
        if "duplic" in (f.title or "").lower() or ev.get("rule") == "max_duplicate_pct" or "duplicate_pct" in ev:
            n, pct = _duplicate_n_pct(f, ctx)
            inflated = _duplicate_inflated_total(f, ctx)
            if n is not None and pct is not None:
                base = f"{format_int_es(n)} pedidos aparecen dos veces ({format_pct(pct)})."
                if inflated is not None and inflated > 0:
                    base += f" Inflan la facturación en unos {format_amount(inflated, unit, digits=0)}."
                return base
        if "faltant" in (f.title or "").lower() or "missing_pct" in ev:
            col = ev.get("column") or (f.variables[0] if f.variables else "una columna")
            col_l = friendly_column(str(col), ctx)
            pct = float(ev.get("missing_pct") or 0)
            every = max(2, int(round(1 / pct))) if pct >= 0.15 else None
            head = (
                f"Falta el {col_l} en 1 de cada {every} pedidos."
                if every
                else f"Faltan datos de {col_l} ({format_pct(pct)})."
            )
            if ctx.df is not None:
                by_plural = {"channel": "canales", "canal": "canales", "region": "regiones", "región": "regiones"}
                for by, plural in by_plural.items():
                    uni = _missing_uniform_across(ctx.df, str(col), by)
                    if uni is True:
                        head += (
                            f" Pasa parejo en todos los {plural}, así que probablemente "
                            "sea un olvido de carga y no algo escondido."
                        )
                        break
                    if uni is False:
                        head += f" No se reparte igual en todos los {plural}."
                        break
            return head
        if plain:
            return plain[0].upper() + plain[1:] + ("." if not plain.endswith(".") else "")

    if f.category == FindingCategory.DISTRIBUTION:
        var = f.variables[0] if f.variables else "la variable"
        var_l = friendly_column(var, ctx)
        mean = ev.get("mean")
        median = ev.get("median")
        u = unit if looks_like_money_column(var) else "unidades"
        if mean is not None and median is not None:
            return (
                f"{var_l.capitalize()} tiene cola larga: el promedio "
                f"(≈{format_amount(mean, u, digits=0)}) queda lejos de la mediana "
                f"(≈{format_amount(median, u, digits=0)}). En ventas suele ser normal."
            )
        return f"{var_l.capitalize()} tiene muchos valores extremos / cola larga."

    if f.category == FindingCategory.SEGMENT:
        return (
            "No encontramos grupos compactos y separados: "
            "las filas se parecen demasiado entre sí para armar 'tipos' claros."
        )
    obs = f.observation or _simplify_title(f, ctx)
    return _strip_jargon(obs)


def _human_why(f: Finding, ctx: CopyContext | None = None) -> str:
    track = finding_track(f)
    ev = f.evidence or {}
    ctx = ctx or _DEFAULT_CTX
    if f.category == FindingCategory.QUALITY:
        if "duplic" in (f.title or "").lower() or ev.get("rule") == "max_duplicate_pct" or "duplicate_pct" in ev:
            inflated = _duplicate_inflated_total(f, ctx)
            if inflated is not None and inflated > 0:
                return (
                    f"Si no los sacás, los totales quedan inflados "
                    f"(unos {format_amount(inflated, ctx.money_unit, digits=0)} de más)."
                )
            return "Las filas repetidas pueden inflar totales y distorsionar promedios."
        if "faltant" in (f.title or "").lower() or "missing_pct" in ev:
            col = friendly_column(
                str(ev.get("column") or (f.variables[0] if f.variables else "esa columna")), ctx
            )
            return f"Para analizar {col}, usá solo las filas que lo tienen; si no, sesgás las conclusiones."
        return "Un problema de calidad puede afectar cualquier conclusión que saques de estos datos."
    if f.category == FindingCategory.ANOMALY:
        return "Esas filas pueden ser errores de carga o casos especiales que merecen revisión."
    if f.category == FindingCategory.RELATIONSHIP:
        if not relationship_confirmed(f):
            return "Todavía no hay respaldo suficiente; tomalo como pista, no como hecho."
        a = ev.get("var_a") or (f.variables[0] if f.variables else "X")
        b = ev.get("var_b") or (f.variables[1] if len(f.variables) > 1 else "Y")
        if str(a) == "customer_size" and looks_like_money_column(str(b)):
            return (
                "Te marca dónde está el ticket alto: a mayor tamaño de cliente, "
                "más facturación por pedido. Ojo: no prueba que agrandar al cliente "
                "haga subir la venta; puede haber otra causa detrás."
            )
        return (
            f"Cuando {friendly_column(str(a), ctx)} y {friendly_column(str(b), ctx)} "
            "se mueven juntos, podés priorizar ese vínculo. "
            "Ojo: no prueba que una cause la otra."
        )
    if f.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(f)
        group = f.variables[1] if len(f.variables) >= 2 else (f.variables[0] if f.variables else "algunos grupos")
        group_l = friendly_column(str(group), ctx)
        plural = _group_plural(str(group), group_l)
        if gap:
            name = _pretty_group(gap["top_name"])
            # Textos distintos por tipo de agrupación (no reutilizar plantilla del Norte)
            if str(group).lower() in {"channel", "canal"}:
                return (
                    f"Si el margen en {name} es similar al resto, empujar ese canal "
                    "sube la facturación sin conseguir clientes nuevos. "
                    f"Ojo: la diferencia entre {plural} no prueba causa por sí sola."
                )
            if str(group).lower() in {"region", "región"}:
                return (
                    f"Si {name} rinde así por razones replicables (mix, precio, equipo), "
                    "es la mejor pista para crecer en las otras regiones. "
                    f"Ojo: la diferencia entre {plural} no prueba causa por sí sola."
                )
            return (
                f"Si {name} se destaca por razones replicables, conviene mirar qué lo hace distinto. "
                f"Ojo: la diferencia entre {plural} no prueba causa por sí sola."
            )
        return (
            f"La diferencia entre {plural} "
            "puede guiar foco comercial, stock o precios."
        )
    if f.category == FindingCategory.DISTRIBUTION:
        return "Con cola larga, el promedio engaña: conviene mirar también la mediana."
    if f.category == FindingCategory.SEGMENT:
        return "Forzar segmentos cuando no hay grupos naturales suele confundir más de lo que ayuda."
    if f.category == FindingCategory.TEMPORAL:
        return "Un patrón en el tiempo puede guiar pronósticos o alertas."
    if track == "problemas":
        return "Puede afectar la confiabilidad de cualquier conclusión que saques de estos datos."
    return "Vale la pena revisarlo antes de tomar decisiones."


def _human_actions(f: Finding, ctx: CopyContext | None = None) -> list[str]:
    ev = f.evidence or {}
    ctx = ctx or _DEFAULT_CTX
    if f.category == FindingCategory.RELATIONSHIP:
        a = ev.get("var_a") or (f.variables[0] if f.variables else None)
        b = ev.get("var_b") or (f.variables[1] if len(f.variables) > 1 else None)
        actions = [
            "Priorizá las cuentas o tramos donde el ticket es más alto.",
            "Ojo: chequeá si no hay una tercera columna que explique las dos.",
        ]
        if str(a) == "customer_size" and str(b).lower() in {"revenue", "facturacion", "facturación", "ventas"}:
            actions = [
                "Priorizá cuentas de tamaño alto en acciones comerciales.",
                "Revisá los casos que se salen de la regla antes de generalizar.",
            ]
        return actions
    if f.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(f)
        group = f.variables[1] if len(f.variables) >= 2 else "grupo"
        if gap:
            name = _pretty_group(gap["top_name"])
            if str(group).lower() in {"channel", "canal"}:
                return [
                    f"Revisá si la diferencia de {name} se mantiene descontando envíos o descuentos.",
                    "Compará el mix de productos entre canales antes de empujar presupuesto.",
                ]
            if str(group).lower() in {"region", "región"}:
                return [
                    f"Compará qué clientes y productos compran en {name} vs. el resto.",
                    "Antes de mover presupuesto, chequeá que no sea un par de clientes grandes.",
                ]
            return [
                f"Compará el perfil de {name} contra el resto.",
                "Revisá si la diferencia importa en plata o unidades reales.",
            ]
        return [
            f"Compará los niveles de {friendly_column(str(group), ctx)} entre sí.",
            "Revisá si la diferencia importa en plata o unidades reales.",
        ]
    if f.category == FindingCategory.ANOMALY:
        return [
            "Revisá esas filas a mano y anotá si son errores o casos especiales.",
            "Contrastalas con reglas de negocio antes de borrarlas o ajustar totales.",
        ]
    if f.category == FindingCategory.QUALITY:
        if "duplic" in (f.title or "").lower() or "duplicate_pct" in ev or ev.get("rule") == "max_duplicate_pct":
            return [
                "Sacalos antes de sumar totales.",
                "Si dudás, buscá pedidos con el mismo número o la misma clave de negocio.",
            ]
        if "faltant" in (f.title or "").lower() or "missing_pct" in ev:
            col = friendly_column(
                str(ev.get("column") or (f.variables[0] if f.variables else "esa columna")), ctx
            )
            return [
                f"Para análisis de {col}, usá solo las filas que lo tienen.",
                "No completes los vacíos sin entender por qué faltan.",
            ]
        return ["Revisá la columna y el umbral según tu negocio."]
    if f.category == FindingCategory.DISTRIBUTION:
        return ["Usá la mediana además del promedio.", "Separá los casos extremos antes de decidir."]
    if f.category == FindingCategory.SEGMENT:
        return ["No fuerces grupos por costumbre.", "Si necesitás cortes, usá reglas de negocio."]
    cleaned = [_strip_jargon(a) for a in (f.follow_up or [])[:2]]
    return cleaned or ["Revisá el hallazgo con tu equipo antes de actuar."]


def method_plain_es(method: str | None) -> str:
    """Método de hallazgo en español claro; nombre técnico entre paréntesis si aporta."""
    if method is None or str(method).strip() == "":
        return "—"
    t = str(method).strip()
    exact = {
        "Spearman + FDR-BH": (
            "Correlación por rangos con corrección por muchas pruebas (Spearman + FDR-BH)"
        ),
        "Spearman + FDR-BH + distance correlation": (
            "Correlación por rangos y dependencia no lineal, "
            "con corrección por muchas pruebas (Spearman + FDR-BH + distance correlation)"
        ),
        "IsolationForest": "Detección de filas raras (IsolationForest)",
        "pandas.DataFrame.duplicated": "Detección de filas repetidas",
        "conteo de NA": "Conteo de valores vacíos",
        "Asimetría (skew) y percentiles": "Asimetría y percentiles",
        "K-means grid + silhouette": "Agrupamiento automático (K-means)",
        "K-means + silhouette; contraste jerárquico/DBSCAN": (
            "Agrupamiento automático con contraste de métodos (K-means / DBSCAN)"
        ),
        "IQR 1.5": "Valores extremos por rango intercuartílico (IQR)",
        "value_counts": "Conteo de categorías",
        "suma agrupada / participación relativa": "Suma agrupada y participación relativa",
        "búsqueda de corte + t-test": "Búsqueda de corte y comparación de medias",
        "PELT L2": "Detección de cambios de nivel (PELT)",
        "PELT_VAR": "Detección de cambios de varianza (PELT)",
        "comparación no paramétrica entre grupos": "Comparación no paramétrica entre grupos",
        "Little MCAR (aproximado)": "Prueba de faltantes al azar (Little MCAR, aproximado)",
    }
    if t in exact:
        return exact[t]
    if t.startswith("quality_rules"):
        return "Reglas de calidad"
    if t.startswith("regresión lineal sobre serie agregada"):
        return "Tendencia por regresión lineal sobre la serie agregada"
    # Sustituciones parciales de jerga frecuente en métodos compuestos
    out = t
    out = re.sub(r"\bFDR-BH\b", "corrección por muchas pruebas (FDR-BH)", out)
    out = re.sub(r"\bFDR\b", "corrección por muchas pruebas", out)
    out = re.sub(r"\bSpearman\b", "correlación por rangos (Spearman)", out, flags=re.I)
    out = re.sub(r"\bresiduales\b", "diferencias del modelo", out, flags=re.I)
    out = re.sub(r"\bcovariables\b", "otras columnas", out, flags=re.I)
    out = re.sub(r"\bIsolationForest\b", "detección de filas raras (IsolationForest)", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def _strip_jargon(text: str) -> str:
    if not text:
        return text
    t = text
    replacements = [
        (r"IsolationForest\s*marcó\s*", "Detectamos "),
        (r"IsolationForest", "detección de filas raras"),
        (r"Spearman\s*ρ\s*=\s*([0-9.eE+-]+)\s*\(p FDR=[^)]+,\s*n=\d+\)\.?", r"relación de fuerza \1."),
        (r"Spearman\s*ρ\s*=\s*([0-9.eE+-]+)", r"relación ≈ \1"),
        (r"Spearman\s*ρ", "relación"),
        (r"Spearman", "relación"),
        (r"Pearson\s*r\s*=\s*([0-9.eE+-]+)", r"relación lineal ≈ \1"),
        (r"Pearson", "relación lineal"),
        (r"Distance correlation\s*=\s*([0-9.eE+-]+)", r"dependencia ≈ \1"),
        (r"Distance correlation", "dependencia"),
        (r"p FDR\s*=\s*[0-9.eE+-]+", ""),
        (r"p FDR", ""),
        (r"FDR-BH", "corrección por muchas pruebas"),
        (r"\bρ\b", "relación"),
        (r"silhouette", "calidad del agrupamiento"),
        (r"K-means|K-Means", "agrupamiento"),
        (r"DBSCAN", "agrupamiento por densidad"),
        (r"Scatterplot y residuales\.?", "Mirá el gráfico en Explorar."),
        (r"scatterplot", "gráfico"),
        (r"residuales", "diferencias del modelo"),
        (r"covariables candidatas", "otras columnas que puedan influir"),
        (r"covariables", "otras columnas"),
        (r"contamination alternativo\s*\([^)]*\)", "otra sensibilidad"),
        (r"contamination", "sensibilidad"),
        (r"MCAR/MAR/MNAR", "si faltan al azar o con patrón"),
        (r"\bMCAR\b", "faltantes completamente al azar"),
        (r"\bMNAR\b", "faltantes no al azar"),
        (r"\bMAR\b", "faltantes con patrón"),
        (r"Ajuste parcial por covariables candidatas\.?", "Revisá si otra columna explica las dos."),
        (r"Repetir con contamination alternativo \(sensibilidad\)\.?", "Probar con otra sensibilidad."),
        (r"Analizar missingness vs otras variables \(MCAR/MAR/MNAR\)\.?", "Revisá si el vacío tiene patrón."),
    ]
    for pat, repl in replacements:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+\.", ".", t)
    return t


def humanize_finding(f: Finding, ctx: CopyContext | None = None) -> dict[str, Any]:
    ctx = ctx or _DEFAULT_CTX
    label, color = severity_label_es(f)
    track = finding_track(f)
    return {
        "track": track,
        "label": label,
        "color": color,
        "pill": category_pill_es(f),
        "what": _human_what(f, ctx),
        "why": _human_why(f, ctx),
        "do": _human_actions(f, ctx),
        "snippet": _evidence_snippet(f) or format_magnitude(f.magnitude, finding=f),
        "p_txt": format_p_value(f.p_value_fdr if f.p_value_fdr is not None else f.p_value),
        "title_simple": _simplify_title(f, ctx),
    }


def _simplify_title(f: Finding, ctx: CopyContext | None = None) -> str:
    t = f.title or ""
    ev = f.evidence or {}
    ctx = ctx or _DEFAULT_CTX
    for junk in (" · pandas.Series.skew / percentiles", " · IQR 1.5"):
        t = t.replace(junk, "")

    if f.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(f)
        if gap and gap.get("pct") is not None:
            var = gap["var"]
            name = str(gap["top_name"])
            direction = "más" if gap["pct"] >= 0 else "menos"
            if looks_like_money_column(var):
                pct_i = abs(gap["pct"])
                if name.lower() in {"norte", "sur", "este", "oeste"}:
                    return f"El {name} vende {pct_i:.0f} % {direction} por pedido que el resto"
                label = name[:1].upper() + name[1:] if name else name
                return f"{label} deja {pct_i:.0f} % {direction} por pedido que el resto"
            return (
                f"{name} está {abs(gap['pct']):.0f} % "
                f"{'más alto' if gap['pct'] >= 0 else 'más bajo'} que el resto"
            )

    if f.category == FindingCategory.RELATIONSHIP:
        vars_ = f.variables or []
        a = ev.get("var_a") or (vars_[0] if vars_ else None)
        b = ev.get("var_b") or (vars_[1] if len(vars_) > 1 else None)
        if a and b:
            if not relationship_confirmed(f):
                return f"Posible relación (no confirmada): '{a}' y '{b}'"
            if str(a) == "customer_size" and str(b).lower() in {"revenue", "facturacion", "facturación", "ventas"}:
                return "Los clientes más grandes gastan más por pedido"
            if looks_like_money_column(str(b)):
                return (
                    f"Cuando sube {friendly_column(str(a), ctx)}, "
                    f"también sube {friendly_column(str(b), ctx)}"
                )
            return f"'{a}' y '{b}' se mueven juntos"

    if "max_duplicate_pct" in t or ev.get("rule") == "max_duplicate_pct" or t.startswith("Filas duplicadas"):
        n, pct = _duplicate_n_pct(f, ctx)
        if n is not None and pct is not None:
            return f"{format_int_es(n)} pedidos aparecen dos veces ({format_pct(pct)})"
        return "Hay filas repetidas"

    if "Regla de calidad" in t:
        rule = str(ev.get("rule") or "")
        if "duplicate" in rule:
            n, pct = _duplicate_n_pct(f, ctx)
            if n is not None and pct is not None:
                return f"{format_int_es(n)} pedidos aparecen dos veces ({format_pct(pct)})"
            return "Hay filas repetidas"
        if "missing" in rule:
            var = f.variables[0] if f.variables else "una columna"
            return f"Faltan datos en '{var}'"
        return "Hay un problema de calidad en los datos"

    if t.startswith("Distribución muy asimétrica"):
        var = f.variables[0] if f.variables else "variable"
        return f"La variable '{var}' tiene muchos valores extremos / cola larga"
    if "faltantes elevados" in t or "Valores faltantes" in t:
        var = ev.get("column") or (f.variables[0] if f.variables else "una columna")
        pct = float(ev.get("missing_pct") or 0)
        if pct >= 0.15:
            every = max(2, int(round(1 / pct)))
            return f"Falta el {friendly_column(str(var), ctx)} en 1 de cada {every} pedidos"
        return f"Faltan datos en '{var}'"
    if t.startswith("Concentración de valores extremos"):
        var = f.variables[0] if f.variables else "variable"
        return f"Hay valores atípicos en '{var}'"
    if t.startswith("Anomalías"):
        return "Hay filas fuera de lo común"
    if t.startswith("Correlación:"):
        vars_ = f.variables or []
        if len(vars_) >= 2:
            if not relationship_confirmed(f):
                return f"Posible relación (no confirmada): '{vars_[0]}' y '{vars_[1]}'"
            return f"'{vars_[0]}' y '{vars_[1]}' se mueven juntos"
        return "Hay una relación entre dos columnas"
    if "cambia el nivel" in t:
        return t
    if "No hay evidencia fuerte de grupos" in t:
        return "No hay grupos naturales claros"
    if t.startswith("Posible relación no lineal"):
        vars_ = f.variables or []
        if len(vars_) >= 2:
            if not relationship_confirmed(f):
                return f"Posible relación (no confirmada): '{vars_[0]}' y '{vars_[1]}'"
            return f"Relación no lineal posible entre '{vars_[0]}' y '{vars_[1]}'"
    return t


def plain_finding_line(f: Finding, ctx: CopyContext | None = None) -> str:
    """Línea corta para 'Hallazgos principales' del informe."""
    h = humanize_finding(f, ctx)
    mag = format_magnitude(f.magnitude, finding=f)
    if f.category in {FindingCategory.STATISTICAL, FindingCategory.RELATIONSHIP}:
        base = _plain_bullet_for_finding(f, ctx) or h["title_simple"]
        return f"{base} ({h['label']})"
    return f"{h['title_simple']} — {mag} ({h['label']})"


def format_bullet_list(items: list[str] | None) -> list[str]:
    """Normaliza listas para UI: sin None, sin vacíos, strings limpios."""
    if not items:
        return []
    out = []
    for x in items:
        s = str(x).strip()
        if s and s.lower() not in {"none", "nan", "[]"}:
            out.append(s)
    return out


EMPTY_FINDINGS_MESSAGE = (
    "Esta vez no apareció nada que se destaque, y eso también es información: "
    "los grupos se comportan parecido. Si esperabas una diferencia, probá con más filas "
    "(ideal: 300 o más) o mirá los gráficos en Explorar para ver los datos por tu cuenta."
)


def build_summary_headline(
    findings: list[Finding],
    profile: DatasetProfile,
    artifacts: dict[str, Any] | None = None,
    ctx: CopyContext | None = None,
) -> tuple[str, str]:
    """Titular estilo diario (Direction C) + bajada con números reales."""
    artifacts = artifacts or {}
    ctx = ctx or _DEFAULT_CTX
    ordered = dedupe_findings(findings)
    disc = [f for f in ordered if finding_track(f) == "descubrimientos"]
    problems = [f for f in ordered if finding_track(f) == "problemas"]
    label = ctx.dataset_label or friendly_dataset_name(profile.name)

    headline = f"{label}: aún no hay un titular fuerte"
    # Preferir hallazgo de grupos con peso en facturación
    top = None
    for f in disc:
        if f.category == FindingCategory.STATISTICAL and group_vs_rest_gap(f):
            top = f
            break
    if top is None and disc:
        top = disc[0]

    if top and top.category == FindingCategory.STATISTICAL:
        gap = group_vs_rest_gap(top)
        shares = _group_share_phrases(top, gap["top_name"]) if gap else []
        if gap and any("mitad de la facturación" in s for s in shares):
            headline = f"{label}: el {gap['top_name']} explica la mitad de lo que facturás"
        elif gap and gap.get("pct") is not None:
            headline = (
                f"{label}: '{gap['top_name']}' rinde ~{abs(gap['pct']):.0f} % "
                f"{'más' if gap['pct'] >= 0 else 'menos'} que el resto"
            )
        else:
            headline = f"{label}: { _simplify_title(top, ctx) }"
    elif top and top.category == FindingCategory.RELATIONSHIP:
        headline = f"{label}: { _simplify_title(top, ctx) }"
    elif top:
        headline = f"{label}: { _simplify_title(top, ctx) }"

    n_strong = sum(
        1
        for f in disc
        if f.severity in {FindingSeverity.CRITICAL, FindingSeverity.NOTABLE}
        or f.practically_important
    )
    n_prob = len(problems)
    row_word = "pedidos" if "venta" in label.lower() or profile.name.lower() in {"commercial", "ventas"} else "filas"
    sub = (
        f"Analizamos {format_int_es(profile.n_rows)} {row_word} y {format_int_es(profile.n_cols)} columnas. "
    )
    if n_strong or n_prob:
        bits = []
        if n_strong:
            bits.append(f"{n_strong} cosa{'s' if n_strong != 1 else ''} que conviene mirar ya")
        if n_prob:
            bits.append(
                f"{n_prob} problema{'s' if n_prob != 1 else ''} de datos que podrían torcer los números"
            )
        sub += "Hay " + " y ".join(bits) + "."
    else:
        sub += "No hay alertas fuertes; igual conviene revisar Explorar."
    return headline, sub


def build_business_kpis(
    findings: list[Finding],
    profile: DatasetProfile,
    df: pd.DataFrame | None = None,
    ctx: CopyContext | None = None,
) -> list[dict[str, str]]:
    """4 métricas de negocio para el resumen (maqueta A)."""
    ctx = ctx or _DEFAULT_CTX
    ordered = dedupe_findings(findings)
    disc = [f for f in ordered if finding_track(f) == "descubrimientos"]
    problems = [f for f in ordered if finding_track(f) == "problemas"]
    n_strong = sum(
        1
        for f in disc
        if f.severity in {FindingSeverity.CRITICAL, FindingSeverity.NOTABLE}
        or (f.practically_important is True)
    )
    unit = ctx.money_unit or "unidades"
    row_label = "Pedidos analizados" if profile.name.lower() in {"commercial", "ventas"} else "Filas analizadas"

    mean_txt, median_txt = "—", ""
    target = None
    if profile.possible_targets:
        target = profile.possible_targets[0]
    elif profile.numeric_columns:
        target = profile.numeric_columns[0]
    if df is not None and target and target in df.columns:
        s = pd.to_numeric(df[target], errors="coerce").dropna()
        if len(s):
            u = unit if looks_like_money_column(target) else "unidades"
            mean_txt = format_amount(float(s.mean()), u, digits=0)
            median_txt = f"mediana {format_amount(float(s.median()), u, digits=0)}"

    # cobertura de problemas
    touch = ""
    if problems:
        miss = [f for f in problems if "missing" in str((f.evidence or {}).keys()) or "faltant" in (f.title or "").lower()]
        if miss:
            pct = float((miss[0].evidence or {}).get("missing_pct") or 0)
            if pct >= 0.15:
                every = max(2, int(round(1 / pct)))
                touch = f"tocan 1 de cada {every} filas"
        if not touch and profile.duplicate_pct >= 0.005:
            touch = f"{format_pct(profile.duplicate_pct)} de filas tocadas"
        if not touch:
            touch = f"{len(problems)} a revisar"

    return [
        {
            "label": row_label,
            "value": format_int_es(profile.n_rows),
            "delta": f"{format_int_es(profile.n_cols)} columnas",
        },
        {
            "label": "Ticket promedio" if target and looks_like_money_column(target) else "Promedio clave",
            "value": mean_txt,
            "delta": median_txt,
        },
        {
            "label": "Hallazgos fuertes",
            "value": format_int_es(n_strong),
            "delta": f"de {format_int_es(len(ordered))} revisados",
        },
        {
            "label": "Problemas de datos",
            "value": format_int_es(len(problems)),
            "delta": touch or "sin alertas graves",
        },
    ]


def suggested_next_steps(
    findings: list[Finding],
    ctx: CopyContext | None = None,
    max_steps: int = 3,
) -> list[str]:
    """Próximos pasos de negocio a partir de hallazgos (sin inventar)."""
    ctx = ctx or _DEFAULT_CTX
    steps: list[str] = []
    ordered = dedupe_findings(findings)
    # Problemas primero (acciones urgentes), después oportunidades
    ordered = [f for f in ordered if finding_track(f) == "problemas"] + [
        f for f in ordered if finding_track(f) != "problemas"
    ]
    for f in ordered:
        if len(steps) >= max_steps:
            break
        if finding_track(f) == "problemas" and (
            "duplic" in (f.title or "").lower()
            or (f.evidence or {}).get("rule") == "max_duplicate_pct"
            or "duplicate_pct" in (f.evidence or {})
        ):
            n, _ = _duplicate_n_pct(f, ctx)
            steps.append(
                f"Sacá los {format_int_es(n)} pedidos repetidos." if n else "Sacá las filas repetidas."
            )
            continue
        if f.category == FindingCategory.STATISTICAL:
            gap = group_vs_rest_gap(f)
            if gap:
                steps.append(f"Investigá qué hace distinto a '{gap['top_name']}'.")
                continue
        if f.category == FindingCategory.RELATIONSHIP and relationship_confirmed(f):
            steps.append("Priorizá el tramo donde el ticket es más alto.")
            continue
    if len(steps) < max_steps:
        steps.append("Mandá el PDF a tu equipo.")
    # unique preserve order
    seen: set[str] = set()
    out = []
    for s in steps:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:max_steps]


def sort_findings_for_humans(findings: list[Finding]) -> list[Finding]:
    """Descubrimientos útiles primero; después problemas por gravedad."""

    def key(f: Finding):
        track = 0 if finding_track(f) == "descubrimientos" else 1
        prac = 0 if f.practically_important else 1
        sev_rank = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.NOTABLE: 1,
            FindingSeverity.INTERESTING: 2,
            FindingSeverity.INFO: 3,
        }.get(f.severity, 4)
        return (track, sev_rank, prac, -f.score)

    return sorted(findings, key=key)


def discoveries_for_summary(findings: list[Finding]) -> list[Finding]:
    """Descubrimientos con el hallazgo del titular (grupos / facturación) primero."""
    disc = [f for f in dedupe_findings(findings) if finding_track(f) == "descubrimientos"]
    best = None
    best_score = -1.0
    for f in disc:
        if f.category != FindingCategory.STATISTICAL:
            continue
        gap = group_vs_rest_gap(f)
        if not gap:
            continue
        shares = _group_share_phrases(f, gap["top_name"])
        # Preferir el que explica facturación / tiene mayor |pct|
        score = abs(float(gap.get("pct") or 0))
        if any("facturación" in s for s in shares):
            score += 100
        if score > best_score:
            best_score = score
            best = f
    if best is None:
        return disc
    return [best] + [f for f in disc if f.id != best.id]


def _plain_limitation_text(text: str) -> str:
    """Limpia jerga típica del motor en limitaciones / alternativas."""
    t = _strip_jargon(str(text or ""))
    t = re.sub(r"outliers?\s*o\s*ties", "valores extremos o empates", t, flags=re.I)
    t = re.sub(r"\boutliers?\b", "valores extremos", t, flags=re.I)
    t = re.sub(r"\bties\b", "empates", t, flags=re.I)
    t = re.sub(r"correlación espuria", "correlación engañosa", t, flags=re.I)
    t = re.sub(r"\bcovariables?\b", "otras variables", t, flags=re.I)
    t = re.sub(r"\bresiduales\b", "restos del ajuste", t, flags=re.I)
    t = re.sub(r"\bcontamination\b", "umbral de rareza", t, flags=re.I)
    return t


def plain_alts_and_followups(f: Finding) -> tuple[list[str], list[str]]:
    """Alternativas y follow-ups sin jerga para informes."""
    cleaned_alts = [_plain_limitation_text(a) for a in format_bullet_list(f.alternative_explanations)]
    follows = [_plain_limitation_text(a) for a in format_bullet_list(f.follow_up)]
    return cleaned_alts, follows


def plain_limitations(f: Finding) -> list[str]:
    """Limitaciones del hallazgo sin jerga."""
    return [_plain_limitation_text(a) for a in format_bullet_list(f.limitations)]


def counted_findings_n(findings: list[Finding]) -> int:
    """Cantidad coherente con la UI (tras dedupe)."""
    return len(dedupe_findings(findings))


def build_plain_summary(
    findings: list[Finding],
    profile: DatasetProfile,
    artifacts: dict[str, Any] | None = None,
    max_bullets: int = 5,
    ctx: CopyContext | None = None,
) -> list[str]:
    """3–5 frases en lenguaje simple a partir de hallazgos reales."""
    artifacts = artifacts or {}
    ctx = ctx or CopyContext(profile=profile)
    if ctx.profile is None:
        ctx = CopyContext(
            df=ctx.df,
            profile=profile,
            money_unit=ctx.money_unit,
            column_labels=ctx.column_labels,
            dataset_label=ctx.dataset_label,
        )
    bullets: list[str] = []

    ordered = dedupe_findings(findings)
    disc = [
        f
        for f in ordered
        if finding_track(f) == "descubrimientos"
        and f.severity
        in {FindingSeverity.CRITICAL, FindingSeverity.NOTABLE, FindingSeverity.INTERESTING}
    ]
    for f in disc:
        phrase = _plain_bullet_for_finding(f, ctx)
        if phrase:
            bullets.append(phrase)
        if len(bullets) >= 3:
            break

    # calidad: duplicados + faltantes con plantillas humanas
    dup = profile.duplicate_pct
    if dup >= 0.005:
        n_dup = getattr(profile, "duplicate_rows", None)
        if n_dup is None and profile.n_rows:
            n_dup = int(round(dup * profile.n_rows))
        if n_dup:
            bullets.append(f"{int(n_dup)} filas están repetidas ({format_pct(dup)})")
        else:
            bullets.append(f"Hay filas repetidas ({format_pct(dup)})")

    miss_vars = [v for v in profile.variables if v.missing_pct >= 0.05]
    if miss_vars and len(bullets) < max_bullets:
        top = sorted(miss_vars, key=lambda v: -v.missing_pct)[0]
        pct = top.missing_pct
        if pct >= 0.15:
            every = max(2, int(round(1 / pct)))
            bullets.append(f"A '{top.name}' le falta el dato en 1 de cada {every} filas")
        else:
            bullets.append(f"A '{top.name}' le faltan datos ({format_pct(pct)})")

    problems = [f for f in ordered if finding_track(f) == "problemas"]
    for f in problems:
        if len(bullets) >= max_bullets:
            break
        if f.category == FindingCategory.ANOMALY:
            phrase = _plain_bullet_for_finding(f, ctx)
            if phrase and phrase not in bullets:
                bullets.append(phrase)

    if not bullets:
        bullets.append("No vimos señales fuertes; igual conviene mirar Explorar.")

    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        low = b.lower()
        if any(tok.lower() in low for tok in SUMMARY_FORBIDDEN if tok != "default"):
            b = _strip_jargon(b)
        if "max_" in b or "(default)" in b.lower() or re.search(r"\bdefault\b", b.lower()):
            continue
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:max_bullets]


def default_explore_columns(profile: DatasetProfile) -> dict[str, str | None]:
    """Elige columnas sensatas (no IDs) para histograma/scatter."""
    ids = set(profile.identifier_columns)
    nums = [c for c in profile.numeric_columns if c not in ids]
    cats = [c for c in profile.categorical_columns if c not in ids]
    targets = [c for c in profile.possible_targets if c in nums] or nums
    hist = targets[0] if targets else (nums[0] if nums else None)
    scatter_x = nums[0] if nums else None
    scatter_y = None
    if len(nums) >= 2:
        # prefer target as y
        scatter_y = targets[0] if targets and targets[0] != scatter_x else nums[1]
        if scatter_y == scatter_x and len(nums) > 2:
            scatter_y = nums[2] if nums[2] != scatter_x else nums[1]
    return {"hist": hist, "scatter_x": scatter_x, "scatter_y": scatter_y, "color_cat": cats[0] if cats else None}


def default_causal_columns(profile: DatasetProfile, df: pd.DataFrame) -> dict[str, Any]:
    """Defaults que no rompen: treatment binario/categórico, outcome numérico."""
    ids = set(profile.identifier_columns)
    nums = [c for c in profile.numeric_columns if c not in ids]
    cats = [c for c in profile.categorical_columns if c not in ids]
    binaries = []
    for c in list(df.columns):
        if c in ids:
            continue
        nunique = df[c].nunique(dropna=True)
        if nunique == 2:
            binaries.append(c)

    treatment = binaries[0] if binaries else (cats[0] if cats else None)
    outcome_candidates = [c for c in (profile.possible_targets or nums) if c != treatment]
    outcome = outcome_candidates[0] if outcome_candidates else (nums[0] if nums else None)
    confounders = [
        c
        for c in (cats + nums)[:5]
        if c not in {treatment, outcome} and c not in ids
    ][:3]
    return {
        "treatment": treatment,
        "outcome": outcome,
        "confounders": confounders,
        "treatment_options": [c for c in (binaries + cats + nums) if c not in ids],
        "outcome_options": [c for c in nums if c not in ids],
    }
