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
    top = findings[:5]
    if top:
        lines.append("")
        lines.append("Hallazgos principales:")
        for i, f in enumerate(top, 1):
            lines.append(f"{i}. **{f.title}** — {f.magnitude} ({f.severity.value})")
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
            lines.append(f"- {g}")
    if profile.quality_issues:
        lines.append("")
        lines.append("Problemas de calidad:")
        for q in profile.quality_issues:
            lines.append(f"- {q}")

    lines.append("")
    lines.append("## Hallazgos con evidencia")
    lines.append("")
    for f in findings:
        lines.append(f"### {f.title}")
        lines.append("")
        lines.append(f"- **Qué se observó:** {f.observation}")
        lines.append(f"- **Magnitud:** {f.magnitude}")
        lines.append(f"- **Método:** {f.method}")
        lines.append(f"- **Categoría / severidad:** {f.category.value} / {f.severity.value}")
        if f.statistically_significant is not None:
            lines.append(f"- **Estadísticamente significativo:** {f.statistically_significant}")
        if f.practically_important is not None:
            lines.append(f"- **Prácticamente importante:** {f.practically_important}")
        lines.append(f"- **Evidencia (clave):** `{json.dumps(f.evidence, ensure_ascii=False, default=str)[:500]}`")
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
