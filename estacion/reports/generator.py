"""Generador de informes exportables (Markdown / JSON)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from estacion.core.types import DatasetProfile, Finding


def export_findings_json(findings: list[Finding], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [f.to_dict() for f in findings]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def generate_markdown_report(
    profile: DatasetProfile,
    findings: list[Finding],
    artifacts: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> str:
    artifacts = artifacts or {}
    lines: list[str] = []
    lines.append(f"# Informe de análisis — {profile.name}")
    lines.append("")
    lines.append(f"_Generado: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Resumen ejecutivo")
    lines.append("")
    lines.append(
        f"Dataset con **{profile.n_rows:,}** filas y **{profile.n_cols}** columnas "
        f"({profile.memory_mb} MB). "
        f"Se priorizaron **{len(findings)}** hallazgos automáticos."
    )
    try:
        from estacion.ux.presentation import (
            build_plain_summary,
            humanize_finding,
            format_magnitude,
            dedupe_findings,
            plain_finding_line,
            evidence_pairs_plain,
        )

        bullets = build_plain_summary(findings, profile, artifacts)
        if bullets:
            lines.append("")
            lines.append("### En lenguaje simple")
            for b in bullets:
                lines.append(f"- {b}")

        top = dedupe_findings(findings)[:5]
        if top:
            lines.append("")
            lines.append("Hallazgos principales:")
            for i, f in enumerate(top, 1):
                lines.append(f"{i}. **{plain_finding_line(f)}**")
    except Exception:
        top = findings[:5]
        if top:
            lines.append("")
            lines.append("Hallazgos principales:")
            for i, f in enumerate(top, 1):
                mag = f.magnitude
                if mag is None or str(mag).lower() in {"none", "nan"}:
                    mag = "—"
                lines.append(f"{i}. **{f.title}** — {mag} ({f.severity.value})")
    lines.append("")
    lines.append("## Características de los datos")
    lines.append("")
    lines.append(f"- Filas duplicadas: {profile.duplicate_rows} ({profile.duplicate_pct:.1%})")
    lines.append(f"- Numéricas: {', '.join(profile.numeric_columns) or '—'}")
    lines.append(f"- Categóricas: {', '.join(profile.categorical_columns) or '—'}")
    lines.append(f"- Fechas: {', '.join(profile.datetime_columns) or '—'}")
    lines.append(f"- Identificadores: {', '.join(profile.identifier_columns) or '—'}")
    lines.append(f"- Posibles objetivos: {', '.join(profile.possible_targets) or '—'}")
    if profile.granularity_notes:
        lines.append("")
        lines.append("Notas de granularidad:")
        for g in profile.granularity_notes:
            lines.append(f"- {_humanize_granularity(g)}")
    if profile.quality_issues:
        lines.append("")
        lines.append("Problemas de calidad:")
        for q in profile.quality_issues:
            lines.append(f"- {q}")

    lines.append("")
    lines.append("## Hallazgos con evidencia")
    lines.append("")
    for f in findings:
        try:
            from estacion.ux.presentation import humanize_finding, format_magnitude, evidence_pairs_plain

            h = humanize_finding(f)
            title = h["title_simple"]
            mag = format_magnitude(f.magnitude, finding=f)
            obs = h["what"]
            pairs = evidence_pairs_plain(f.evidence)
        except Exception:
            title = f.title
            mag = f.magnitude
            obs = f.observation
            pairs = []
        lines.append(f"### {title}")
        lines.append("")
        from estacion.ux.presentation import (
            CATEGORY_LABELS_ES,
            SEVERITY_LABELS_ES,
            bool_es,
        )

        cat = CATEGORY_LABELS_ES.get(f.category.value, f.category.value)
        sev = SEVERITY_LABELS_ES.get(f.severity.value, f.severity.value)
        lines.append(f"- **Qué se observó:** {obs}")
        lines.append(f"- **Magnitud:** {mag}")
        lines.append(f"- **Método:** {f.method}")
        lines.append(f"- **Categoría / severidad:** {cat} / {sev}")
        if f.statistically_significant is not None:
            lines.append(f"- **Estadísticamente significativo:** {bool_es(f.statistically_significant)}")
        if f.practically_important is not None:
            lines.append(f"- **Prácticamente importante:** {bool_es(f.practically_important)}")
        if pairs:
            lines.append("- **Evidencia (clave):**")
            for k, v in pairs:
                lines.append(f"  - {k}: {v}")
        lines.append("- **Explicaciones alternativas:**")
        for a in f.alternative_explanations:
            lines.append(f"  - {a}")
        lines.append("- **Análisis adicional sugerido:**")
        for a in f.follow_up:
            lines.append(f"  - {a}")
        if f.limitations:
            lines.append("- **Limitaciones:**")
            for a in f.limitations:
                lines.append(f"  - {a}")
        lines.append("")

    lines.append("## Limitaciones generales del informe")
    lines.append("")
    lines.append("- Los hallazgos son exploratorios y están sujetos a múltiples comparaciones.")
    lines.append("- Correlación o asociación ajustada **no** implica causalidad.")
    lines.append("- La relevancia de negocio requiere juicio humano y conocimiento de dominio.")
    lines.append("- Muestras pequeñas o desbalanceadas pueden producir falsos positivos/negativos.")
    lines.append("")
    lines.append("## Próximas investigaciones recomendadas")
    lines.append("")
    follow = []
    for f in findings[:10]:
        follow.extend(f.follow_up[:1])
    for item in list(dict.fromkeys(follow))[:12]:
        lines.append(f"- {item}")

    md = "\n".join(lines)
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
    return md


def _humanize_granularity(note: str) -> str:
    s = str(note)
    # Timedelta crudo → texto corto
    s = s.replace("0 days ", "").replace("0 day ", "")
    if "ratio único=" in s:
        s = s.replace("ratio único=", "proporción de valores únicos ≈ ")
    return s
