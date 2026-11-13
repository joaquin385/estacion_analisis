"""Capa de presentación: lenguaje simple, sin cambiar los números del motor."""

from .presentation import (
    build_plain_summary,
    format_p_value,
    format_number,
    finding_track,
    severity_label_es,
    humanize_finding,
    sort_findings_for_humans,
    default_explore_columns,
    default_causal_columns,
)

__all__ = [
    "build_plain_summary",
    "format_p_value",
    "format_number",
    "finding_track",
    "severity_label_es",
    "humanize_finding",
    "sort_findings_for_humans",
    "default_explore_columns",
    "default_causal_columns",
]
