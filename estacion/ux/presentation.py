"""Traducción a lenguaje humano de hallazgos y defaults de UI."""

from __future__ import annotations

from typing import Any

import pandas as pd

from estacion.core.types import DatasetProfile, Finding, FindingCategory, FindingSeverity


QUALITY_LIKE = {
    FindingCategory.QUALITY,
    FindingCategory.ANOMALY,
}


def format_number(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if abs(v) >= 1000:
        return f"{v:,.1f}".replace(",", " ")
    if abs(v) >= 10:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}g}"


def format_p_value(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0,001"
    return f"{p:.3f}".replace(".", ",")


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


def _evidence_snippet(f: Finding) -> str:
    ev = f.evidence or {}
    bits = []
    if "group_means" in ev and isinstance(ev["group_means"], dict):
        items = list(ev["group_means"].items())[:4]
        bits.append(
            " · ".join(f"{k}: {format_number(v)}" for k, v in items)
        )
    if "spearman_r" in ev:
        bits.append(f"correlación={format_number(ev['spearman_r'])}")
    if "pearson_r" in ev:
        bits.append(f"r={format_number(ev['pearson_r'])}")
    if "missing_pct" in ev:
        bits.append(f"{format_number(100 * float(ev['missing_pct']))}% vacíos")
    if "duplicate_pct" in ev:
        bits.append(f"{format_number(100 * float(ev['duplicate_pct']))}% duplicados")
    if "effect_size" in ev:
        bits.append(f"efecto={format_number(ev['effect_size'])}")
    if "effect_size_cohens_d" in ev:
        bits.append(f"d={format_number(ev['effect_size_cohens_d'])}")
    return "; ".join(bits)


def humanize_finding(f: Finding) -> dict[str, Any]:
    label, color = severity_label_es(f)
    track = finding_track(f)
    why = ""
    if track == "problemas":
        why = "Puede afectar la confiabilidad de cualquier conclusión que saques de estos datos."
    elif f.category == FindingCategory.RELATIONSHIP:
        why = "Sugiere una relación entre variables que puede importar para negocio o análisis."
    elif f.category == FindingCategory.STATISTICAL:
        why = "Hay diferencias entre grupos; conviene ver si son relevantes en unidades reales."
    elif f.category == FindingCategory.DISTRIBUTION:
        why = "La forma de la variable puede influir en promedios, modelos y detección de outliers."
    elif f.category == FindingCategory.TEMPORAL:
        why = "Hay un patrón en el tiempo que puede guiar pronósticos o alertas."
    elif f.category == FindingCategory.SEGMENT:
        why = "Puede haber grupos de comportamiento distintos."
    else:
        why = "Vale la pena revisarlo antes de tomar decisiones."

    actions = list(f.follow_up[:2]) if f.follow_up else ["Revisar en la pestaña Explorar."]
    return {
        "track": track,
        "label": label,
        "color": color,
        "what": f.observation,
        "why": why,
        "do": actions,
        "snippet": _evidence_snippet(f) or f.magnitude,
        "p_txt": format_p_value(f.p_value_fdr if f.p_value_fdr is not None else f.p_value),
        "title_simple": _simplify_title(f),
    }


def _simplify_title(f: Finding) -> str:
    t = f.title
    # quitar ruido técnico frecuente
    for junk in (" · pandas.Series.skew / percentiles", " · IQR 1.5"):
        t = t.replace(junk, "")
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
    return t


def sort_findings_for_humans(findings: list[Finding]) -> list[Finding]:
    """Problemas graves primero; luego descubrimientos por relevancia práctica/score."""

    def key(f: Finding):
        track = 0 if finding_track(f) == "problemas" else 1
        # boost practical
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

    # calidad
    dup = profile.duplicate_pct
    miss_vars = [v for v in profile.variables if v.missing_pct >= 0.05]
    if dup >= 0.01 or miss_vars:
        miss_txt = ""
        if miss_vars:
            top = sorted(miss_vars, key=lambda v: -v.missing_pct)[0]
            miss_txt = (
                f" '{top.name}' tiene ~{100 * top.missing_pct:.0f}% de valores vacíos."
            )
        dup_txt = f"~{100 * dup:.1f}% de filas repetidas." if dup >= 0.005 else "pocas filas repetidas."
        bullets.append(f"Calidad: {dup_txt}{miss_txt}")
    else:
        bullets.append("Calidad: no vimos problemas graves de faltantes o duplicados.")

    ordered = sort_findings_for_humans(findings)
    # top discovery with practical importance or statistical
    disc = [
        f
        for f in ordered
        if finding_track(f) == "descubrimientos"
        and f.severity in {FindingSeverity.CRITICAL, FindingSeverity.NOTABLE, FindingSeverity.INTERESTING}
    ]
    for f in disc[:3]:
        h = humanize_finding(f)
        bullets.append(f"Hallazgo: {h['title_simple']}. {h['snippet']}".strip())
        if len(bullets) >= max_bullets:
            break

    problems = [f for f in ordered if finding_track(f) == "problemas"]
    for f in problems[:2]:
        if len(bullets) >= max_bullets:
            break
        if f.severity in {FindingSeverity.CRITICAL, FindingSeverity.NOTABLE}:
            h = humanize_finding(f)
            bullets.append(f"Atención: {h['title_simple']}.")

    fdr = (artifacts.get("fdr_global") or {}) if artifacts else {}
    if fdr.get("n_tested"):
        bullets.append(
            f"Hallazgos con respaldo estadístico (tras corregir por muchas pruebas): "
            f"{fdr.get('n_significant', 0)} de {fdr.get('n_tested')}."
        )

    # unique preserve order
    seen = set()
    out = []
    for b in bullets:
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
