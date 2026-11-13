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
    ctx=None,
) -> str:
    artifacts = artifacts or {}
    lines: list[str] = []

    from estacion.ux.presentation import (
        bool_es,
        build_plain_summary,
        build_summary_headline,
        counted_findings_n,
        dedupe_findings,
        evidence_pairs_plain,
        format_int_es,
        format_memory_mb,
        format_pct,
        friendly_column,
        friendly_dataset_name,
        humanize_finding,
        humanize_quality_issue,
        method_plain_es,
        plain_alts_and_followups,
        plain_finding_line,
        plain_limitations,
    )

    name_txt = (getattr(ctx, "dataset_label", None) if ctx else None) or friendly_dataset_name(profile.name)
    lines.append(f"# Informe de análisis — {name_txt}")
    lines.append("")
    lines.append(f"_Generado: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")

    headline, sub = build_summary_headline(findings, profile, artifacts, ctx)
    lines.append("## Resumen ejecutivo")
    lines.append("")
    lines.append(f"**{headline}**")
    lines.append("")
    lines.append(sub)
    lines.append("")
    n_rows_txt = format_int_es(profile.n_rows)
    n_cols_txt = format_int_es(profile.n_cols)
    n_find = counted_findings_n(findings)
    n_find_txt = format_int_es(n_find)
    mem_txt = format_memory_mb(profile.memory_mb)
    lines.append(
        f"Dataset con **{n_rows_txt}** filas y **{n_cols_txt}** columnas "
        f"({mem_txt}). "
        f"Se priorizaron **{n_find_txt}** hallazgos (tras deduplicar)."
    )

    bullets = build_plain_summary(findings, profile, artifacts, ctx=ctx)
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
            lines.append(f"{i}. **{plain_finding_line(f, ctx)}**")

    lines.append("")
    lines.append("## Características de los datos")
    lines.append("")
    lines.append(
        f"- Filas duplicadas: {format_int_es(profile.duplicate_rows)} "
        f"({format_pct(profile.duplicate_pct, digits=1)})"
    )
    lines.append(
        "- Numéricas: "
        + (", ".join(friendly_column(c, ctx) for c in profile.numeric_columns) or "—")
    )
    lines.append(
        "- Categóricas: "
        + (", ".join(friendly_column(c, ctx) for c in profile.categorical_columns) or "—")
    )
    lines.append(f"- Fechas: {', '.join(profile.datetime_columns) or '—'}")
    lines.append(f"- Identificadores: {', '.join(profile.identifier_columns) or '—'}")
    lines.append(
        "- Posibles objetivos: "
        + (", ".join(friendly_column(c, ctx) for c in profile.possible_targets) or "—")
    )
    if profile.granularity_notes:
        lines.append("")
        lines.append("Notas de granularidad:")
        for g in profile.granularity_notes:
            lines.append(f"- {_humanize_granularity(g)}")
    if profile.quality_issues:
        # Evitar repetir "Filas duplicadas" si ya está arriba
        qi = []
        for q in profile.quality_issues:
            hq = humanize_quality_issue(q, ctx)
            if "duplicad" in hq.lower() and profile.duplicate_rows:
                continue
            qi.append(hq)
        if qi:
            lines.append("")
            lines.append("Problemas de calidad:")
            for q in qi:
                lines.append(f"- {q}")

    lines.append("")
    lines.append("## Hallazgos con evidencia")
    lines.append("")
    for f in dedupe_findings(findings):
        h = humanize_finding(f, ctx)
        title = h["title_simple"]
        from estacion.ux.presentation import format_magnitude, group_vs_rest_gap, format_amount, looks_like_money_column

        gap = group_vs_rest_gap(f)
        if gap and gap.get("pct") is not None:
            u = (getattr(ctx, "money_unit", "unidades") if ctx else "unidades")
            if not looks_like_money_column(gap["var"]):
                u = "unidades"
            mag = (
                f"~{abs(gap['pct']):.0f} % vs el resto "
                f"({format_amount(gap['top_val'], u, digits=0)} vs {format_amount(gap['rest_avg'], u, digits=0)})"
            )
        else:
            mag = format_magnitude(f.magnitude, finding=f)
        obs = h["what"]
        pairs = evidence_pairs_plain(f.evidence)
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"- **Qué se observó:** {obs}")
        lines.append(f"- **Por qué importa:** {h['why']}")
        lines.append(f"- **Qué podés hacer:** {' · '.join(h['do'])}")
        lines.append(f"- **Magnitud:** {mag}")
        lines.append(f"- **Método:** {method_plain_es(f.method)}")
        if f.statistically_significant is not None:
            lines.append(f"- **Estadísticamente significativo:** {bool_es(f.statistically_significant)}")
        if f.practically_important is not None:
            lines.append(f"- **Prácticamente importante:** {bool_es(f.practically_important)}")
        if pairs:
            lines.append("- **Evidencia (clave):**")
            for k, v in pairs:
                lines.append(f"  - {k}: {v}")
        alts, follows = plain_alts_and_followups(f)
        if alts:
            lines.append("- **Explicaciones alternativas:**")
            for a in alts:
                lines.append(f"  - {a}")
        if follows:
            lines.append("- **Análisis adicional sugerido:**")
            for a in follows:
                lines.append(f"  - {a}")
        lims = plain_limitations(f)
        if lims:
            lines.append("- **Limitaciones:**")
            for a in lims:
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
    for f in dedupe_findings(findings)[:10]:
        _, follows = plain_alts_and_followups(f)
        follow.extend(follows[:1])
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
    s = s.replace("0 days ", "").replace("0 day ", "")
    if "ratio único=" in s:
        s = s.replace("ratio único=", "proporción de valores únicos ≈ ")
    return s
