"""Traducción a lenguaje humano de hallazgos y defaults de UI."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from estacion.core.types import DatasetProfile, Finding, FindingCategory, FindingSeverity


QUALITY_LIKE = {
    FindingCategory.QUALITY,
    FindingCategory.ANOMALY,
}

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
    Definición única de brecha entre grupos: extremo vs promedio del resto.
    (No usar relative_mean_gap del motor: ese compara max−min / media global.)
    """
    ev = f.evidence or {}
    means = ev.get("group_means")
    if not isinstance(means, dict) or len(means) < 2:
        return None
    items = [(str(k), float(v)) for k, v in means.items()]
    items.sort(key=lambda kv: kv[1], reverse=True)
    top_name, top_val = items[0]
    rest = [v for _, v in items[1:]]
    rest_avg = sum(rest) / len(rest)
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
    Descubrimientos: Muy relevante / Relevante / Para mirar / Info
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
    # descubrimientos
    if sev == FindingSeverity.CRITICAL or (
        sev == FindingSeverity.NOTABLE and f.practically_important
    ):
        return "Muy relevante", "teal"
    if sev == FindingSeverity.NOTABLE:
        return "Relevante", "teal"
    if sev == FindingSeverity.INTERESTING:
        return "Para mirar", "blue"
    return "Info", "gray"


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


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Quita tarjetas repetidas (misma clase de problema / mismas variables)."""
    seen: set[tuple] = set()
    out: list[Finding] = []
    # Preferir el de mayor score dentro del mismo grupo
    ranked = sorted(findings, key=lambda f: -f.score)
    for f in ranked:
        if finding_track(f) == "problemas":
            key = _problem_dedupe_key(f)
        else:
            key = ("disc", f.category.value, tuple(sorted(f.variables or [])), f.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    # restaurar orden humano
    return sort_findings_for_humans(out)


def _group_vs_rest_phrase(f: Finding) -> str | None:
    gap = group_vs_rest_gap(f)
    if not gap:
        return None

    def _approx(v: float) -> str:
        if abs(v) >= 20:
            return f"{v:.0f}"
        return format_number(v)

    top_name, top_val, rest_avg, pct, var = (
        gap["top_name"],
        gap["top_val"],
        gap["rest_avg"],
        gap["pct"],
        gap["var"],
    )
    if pct is None:
        return (
            f"'{top_name}' tiene un promedio de '{var}' de ≈{_approx(top_val)} "
            f"(resto ≈{_approx(rest_avg)})"
        )
    direction = "más alto" if pct >= 0 else "más bajo"
    return (
        f"'{top_name}' tiene un promedio de '{var}' ~{abs(pct):.0f} % {direction} "
        f"que el resto (≈{_approx(top_val)} vs ≈{_approx(rest_avg)})"
    )


def _plain_bullet_for_finding(f: Finding) -> str | None:
    """Una frase 100% humana para el resumen."""
    ev = f.evidence or {}
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
        phrase = _group_vs_rest_phrase(f)
        if phrase:
            return phrase
        return _simplify_title(f)

    if f.category == FindingCategory.QUALITY:
        if "duplicate_pct" in ev or "duplic" in (f.title or "").lower() or ev.get("rule") == "max_duplicate_pct":
            n = ev.get("duplicate_rows")
            pct = ev.get("duplicate_pct")
            if pct is None and ev.get("value") is not None:
                pct = ev.get("value")
            if n is not None and pct is not None:
                return f"{int(n)} filas están repetidas ({format_pct(float(pct))})"
            if pct is not None:
                return f"Hay filas repetidas ({format_pct(float(pct))})"
            return "Hay filas repetidas en la planilla"
        if "missing_pct" in ev or "faltant" in (f.title or "").lower():
            col = ev.get("column") or (f.variables[0] if f.variables else "una columna")
            pct = float(ev.get("missing_pct") or 0)
            if pct >= 0.15:
                # "1 de cada N"
                every = max(2, int(round(1 / pct))) if pct > 0 else 0
                return f"A '{col}' le falta el dato en 1 de cada {every} filas"
            return f"A '{col}' le faltan datos ({format_pct(pct)})"
        return None

    if f.category == FindingCategory.ANOMALY:
        n = ev.get("n_anomalies")
        pct = ev.get("pct")
        if n is not None:
            pct_txt = f" ({format_pct(float(pct) / 100 if float(pct) > 1 else float(pct))})" if pct is not None else ""
            # pct en evidence suele venir ya en % (2.0)
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


def _human_what(f: Finding) -> str:
    ev = f.evidence or {}
    plain = _plain_bullet_for_finding(f)
    if f.category == FindingCategory.RELATIONSHIP:
        a = ev.get("var_a") or (f.variables[0] if f.variables else "X")
        b = ev.get("var_b") or (f.variables[1] if len(f.variables) > 1 else "Y")
        if not relationship_confirmed(f):
            return f"Hay una posible relación entre '{a}' y '{b}', todavía no confirmada."
        r = ev.get("spearman_r")
        strength = _strength_word(float(r) if r is not None else None)
        if r is not None and float(r) < 0:
            return f"Cuando sube '{a}', '{b}' tiende a bajar (relación {strength})."
        return f"Cuando sube '{a}', '{b}' tiende a subir (relación {strength})."
    if f.category == FindingCategory.STATISTICAL:
        phrase = _group_vs_rest_phrase(f)
        if phrase:
            return phrase + "."
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
        if plain:
            return plain[0].upper() + plain[1:] + ("." if not plain.endswith(".") else "")
    if f.category == FindingCategory.DISTRIBUTION:
        var = f.variables[0] if f.variables else "la variable"
        mean = ev.get("mean")
        median = ev.get("median")
        if mean is not None and median is not None:
            return (
                f"'{var}' tiene cola larga: el promedio (≈{format_number(mean)}) "
                f"queda lejos de la mediana (≈{format_number(median)}). En ventas suele ser normal."
            )
        return f"'{var}' tiene muchos valores extremos / cola larga."
    if f.category == FindingCategory.SEGMENT:
        return (
            "No encontramos grupos compactos y separados: "
            "las filas se parecen demasiado entre sí para armar 'tipos' claros."
        )
    # fallback: limpiar observación cruda de jerga
    obs = f.observation or _simplify_title(f)
    return _strip_jargon(obs)


def _human_why(f: Finding) -> str:
    track = finding_track(f)
    ev = f.evidence or {}
    if f.category == FindingCategory.QUALITY:
        if "duplic" in (f.title or "").lower() or ev.get("rule") == "max_duplicate_pct" or "duplicate_pct" in ev:
            return "Las filas repetidas pueden inflar totales y distorsionar promedios."
        if "faltant" in (f.title or "").lower() or "missing_pct" in ev:
            col = ev.get("column") or (f.variables[0] if f.variables else "esa columna")
            return f"Sin el dato de '{col}' es fácil llegar a conclusiones sesgadas."
        return "Un problema de calidad puede afectar cualquier conclusión que saques de estos datos."
    if f.category == FindingCategory.ANOMALY:
        return "Esas filas pueden ser errores de carga o casos especiales que merecen revisión."
    if f.category == FindingCategory.RELATIONSHIP:
        if not relationship_confirmed(f):
            return "Todavía no hay respaldo estadístico suficiente; tomalo como pista, no como hecho."
        return "Si dos columnas se mueven juntas, puede ayudar a priorizar acciones o hipótesis de negocio."
    if f.category == FindingCategory.STATISTICAL:
        group = f.variables[1] if len(f.variables) >= 2 else (f.variables[0] if f.variables else "algunos grupos")
        return f"La diferencia entre niveles de '{group}' puede guiar foco comercial, stock o precios."
    if f.category == FindingCategory.DISTRIBUTION:
        return "Con cola larga, el promedio engaña: conviene mirar también la mediana."
    if f.category == FindingCategory.SEGMENT:
        return "Forzar segmentos cuando no hay grupos naturales suele confundir más de lo que ayuda."
    if f.category == FindingCategory.TEMPORAL:
        return "Un patrón en el tiempo puede guiar pronósticos o alertas."
    if track == "problemas":
        return "Puede afectar la confiabilidad de cualquier conclusión que saques de estos datos."
    return "Vale la pena revisarlo antes de tomar decisiones."


def _human_actions(f: Finding) -> list[str]:
    ev = f.evidence or {}
    if f.category == FindingCategory.RELATIONSHIP:
        return ["Mirá el gráfico en Explorar.", "Pensá si hay una tercera variable que explique las dos."]
    if f.category == FindingCategory.STATISTICAL:
        return ["Mirá el gráfico de medias en Explorar.", "Revisá si la diferencia importa en plata o unidades reales."]
    if f.category == FindingCategory.ANOMALY:
        return ["Revisá esas filas a mano.", "Contrastalas con reglas de negocio antes de borrarlas."]
    if f.category == FindingCategory.QUALITY:
        if "duplic" in (f.title or "").lower() or "duplicate_pct" in ev or ev.get("rule") == "max_duplicate_pct":
            return ["Definí una clave de negocio.", "Mirálas antes de eliminarlas."]
        if "faltant" in (f.title or "").lower() or "missing_pct" in ev:
            return [
                "Revisá si el vacío tiene un patrón (p.ej. un canal).",
                "No completes los vacíos sin entender por qué faltan.",
            ]
        return ["Revisá la columna y el umbral según tu negocio."]
    if f.category == FindingCategory.DISTRIBUTION:
        return ["Mirá el histograma en Explorar.", "Usá la mediana además del promedio."]
    if f.category == FindingCategory.SEGMENT:
        return ["No fuerces grupos por costumbre.", "Si necesitás cortes, usá reglas de negocio."]
    # limpiar follow_ups del motor
    cleaned = []
    for a in (f.follow_up or [])[:2]:
        cleaned.append(_strip_jargon(a))
    return cleaned or ["Revisar en la pestaña Explorar."]


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


def humanize_finding(f: Finding) -> dict[str, Any]:
    label, color = severity_label_es(f)
    track = finding_track(f)
    return {
        "track": track,
        "label": label,
        "color": color,
        "what": _human_what(f),
        "why": _human_why(f),
        "do": _human_actions(f),
        "snippet": _evidence_snippet(f) or format_magnitude(f.magnitude),
        "p_txt": format_p_value(f.p_value_fdr if f.p_value_fdr is not None else f.p_value),
        "title_simple": _simplify_title(f),
    }


def _simplify_title(f: Finding) -> str:
    t = f.title or ""
    ev = f.evidence or {}
    for junk in (" · pandas.Series.skew / percentiles", " · IQR 1.5"):
        t = t.replace(junk, "")
    if "max_duplicate_pct" in t or ev.get("rule") == "max_duplicate_pct":
        return "Hay filas repetidas"
    if "Regla de calidad" in t:
        rule = str(ev.get("rule") or "")
        if "duplicate" in rule:
            return "Hay filas repetidas"
        if "missing" in rule:
            var = f.variables[0] if f.variables else "una columna"
            return f"Faltan datos en '{var}'"
        return "Hay un problema de calidad en los datos"
    if t.startswith("Distribución muy asimétrica"):
        var = f.variables[0] if f.variables else "variable"
        return f"La variable '{var}' tiene muchos valores extremos / cola larga"
    if t.startswith("Filas duplicadas"):
        return "Hay filas repetidas en la planilla"
    if "faltantes elevados" in t or "Valores faltantes" in t:
        var = f.variables[0] if f.variables else "una columna"
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
        return t  # ya es bastante humano
    if "No hay evidencia fuerte de grupos" in t:
        return "No hay grupos naturales claros"
    if t.startswith("Posible relación no lineal"):
        vars_ = f.variables or []
        if len(vars_) >= 2:
            if not relationship_confirmed(f):
                return f"Posible relación (no confirmada): '{vars_[0]}' y '{vars_[1]}'"
            return f"Relación no lineal posible entre '{vars_[0]}' y '{vars_[1]}'"
    return t


def plain_finding_line(f: Finding) -> str:
    """Línea corta para 'Hallazgos principales' del informe."""
    h = humanize_finding(f)
    mag = format_magnitude(f.magnitude, finding=f)
    # Preferir la frase humana del what si es de grupos/relaciones
    if f.category in {FindingCategory.STATISTICAL, FindingCategory.RELATIONSHIP}:
        base = _plain_bullet_for_finding(f) or h["title_simple"]
        return f"{base} ({h['label']})"
    return f"{h['title_simple']} — {mag} ({h['label']})"


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


def build_plain_summary(
    findings: list[Finding],
    profile: DatasetProfile,
    artifacts: dict[str, Any] | None = None,
    max_bullets: int = 5,
) -> list[str]:
    """3–5 frases en lenguaje simple a partir de hallazgos reales."""
    artifacts = artifacts or {}
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
        phrase = _plain_bullet_for_finding(f)
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

    # anomalías u otros problemas graves si queda cupo
    problems = [f for f in ordered if finding_track(f) == "problemas"]
    for f in problems:
        if len(bullets) >= max_bullets:
            break
        if f.category == FindingCategory.ANOMALY:
            phrase = _plain_bullet_for_finding(f)
            if phrase and phrase not in bullets:
                bullets.append(phrase)

    if not bullets:
        bullets.append("No vimos señales fuertes; igual conviene mirar Explorar.")

    seen: set[str] = set()
    out: list[str] = []
    for b in bullets:
        # garantizar sin jerga de resumen
        low = b.lower()
        if any(tok.lower() in low for tok in SUMMARY_FORBIDDEN if tok != "default"):
            # reintentar limpiando
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
