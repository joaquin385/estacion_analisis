"""Export PDF con figuras estáticas (matplotlib) — tema claro Claro SaaS."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

from estacion.core.types import DatasetProfile, Finding


def _wrap(ax, title: str):
    ax.set_title(title, fontsize=11, loc="left", color="#0F172A")
    ax.grid(True, alpha=0.25)


def _fmt_int(n: Any) -> str:
    try:
        from estacion.ux.presentation import format_int_es

        return format_int_es(n)
    except Exception:
        return str(n)


def _fmt_mem(mb: Any) -> str:
    try:
        from estacion.ux.presentation import format_memory_mb

        return format_memory_mb(mb)
    except Exception:
        return f"{mb} MB"


def _es_formatter(x, _pos=None):
    try:
        from estacion.ux.presentation import format_int_es, format_number

        if abs(float(x) - round(float(x))) < 1e-9:
            return format_int_es(int(round(float(x))))
        return format_number(float(x), digits=2)
    except Exception:
        return str(x)


def _apply_es_axes(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_es_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_es_formatter))


def generate_pdf_report(
    df: pd.DataFrame,
    profile: DatasetProfile,
    findings: list[Finding],
    output_path: str | Path,
    artifacts: dict[str, Any] | None = None,
    max_findings: int = 12,
    ctx=None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts or {}

    from estacion.ux.presentation import (
        bool_es,
        build_plain_summary,
        build_summary_headline,
        counted_findings_n,
        dedupe_findings,
        evidence_pairs_plain,
        format_amount,
        format_magnitude,
        format_pct,
        friendly_column,
        friendly_dataset_name,
        group_vs_rest_gap,
        humanize_finding,
        looks_like_money_column,
        method_plain_es,
        plain_alts_and_followups,
        plain_finding_line,
        plain_limitations,
    )

    name_txt = (getattr(ctx, "dataset_label", None) if ctx else None) or friendly_dataset_name(profile.name)
    unit = (getattr(ctx, "money_unit", None) if ctx else None) or "unidades"

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#F6F8FB")
        ax = fig.add_subplot(111)
        ax.axis("off")
        headline, sub = build_summary_headline(findings, profile, artifacts, ctx)
        lines = [
            "Estación de Análisis",
            f"Informe: {name_txt}",
            f"Generado: {datetime.now().isoformat(timespec='seconds')}",
            "",
            headline,
            sub,
            "",
            f"Filas: {_fmt_int(profile.n_rows)}   Columnas: {_fmt_int(profile.n_cols)}   Memoria: {_fmt_mem(profile.memory_mb)}",
            f"Duplicados: {_fmt_int(profile.duplicate_rows)} ({format_pct(profile.duplicate_pct, digits=1)})",
            f"Hallazgos priorizados: {_fmt_int(counted_findings_n(findings))}",
            "",
            "Resumen en lenguaje simple:",
        ]
        for b in build_plain_summary(findings, profile, artifacts, ctx=ctx):
            lines.append(f"• {b}")
        lines.append("")
        lines.append("Hallazgos principales:")
        for i, f in enumerate(dedupe_findings(findings)[:8], 1):
            lines.append(f"{i}. {plain_finding_line(f, ctx)}")
        lines += [
            "",
            "Nota: los cálculos son estadísticos; correlación ≠ causalidad.",
            "Este PDF es exploratorio y debe interpretarse con conocimiento de dominio.",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="DejaVu Sans", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        miss = pd.Series({v.name: v.missing_pct for v in profile.variables}).sort_values(ascending=False)
        miss = miss[miss > 0].head(20)
        if len(miss):
            fig, ax = plt.subplots(figsize=(11, 6))
            miss.index = [friendly_column(str(i), ctx) for i in miss.index]
            miss.plot(kind="bar", ax=ax, color="#4F46E5")
            _wrap(ax, "Proporción de valores faltantes (top)")
            ax.set_ylabel("% vacíos")
            _apply_es_axes(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        nums = profile.numeric_columns[:4]
        if nums:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes = axes.ravel()
            for i, col in enumerate(nums):
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                axes[i].hist(s, bins=30, color="#818CF8", alpha=0.85, edgecolor="white")
                _wrap(axes[i], f"Distribución: {friendly_column(col, ctx)}")
                _apply_es_axes(axes[i])
            for j in range(len(nums), 4):
                axes[j].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        corr_art = artifacts.get("correlation") or {}
        mat = corr_art.get("spearman_matrix")
        if isinstance(mat, pd.DataFrame) and mat.shape[0] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            labels = [friendly_column(str(c), ctx) for c in mat.columns]
            im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels([friendly_column(str(i), ctx) for i in mat.index], fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
            _wrap(ax, "Mapa de relaciones")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        seg = artifacts.get("segmentation") or {}
        coords = seg.get("pca_coords")
        labels = (seg.get("kmeans") or {}).get("labels")
        if coords is not None and labels is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            labs = np.asarray(labels)
            for lab in sorted(set(labs.tolist())):
                mask = labs == lab
                ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, label=str(lab))
            ax.legend(title="grupo", fontsize=8)
            _wrap(ax, "Segmentos (PCA 2D)")
            _apply_es_axes(ax)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        for f in dedupe_findings(findings)[:max_findings]:
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.patch.set_facecolor("#F6F8FB")
            ax = fig.add_subplot(111)
            ax.axis("off")
            h = humanize_finding(f, ctx)
            title = h["title_simple"]
            gap = group_vs_rest_gap(f)
            if gap and gap.get("pct") is not None:
                u = unit if looks_like_money_column(gap["var"]) else "unidades"
                mag = (
                    f"~{abs(gap['pct']):.0f} % vs el resto "
                    f"({format_amount(gap['top_val'], u, digits=0)} vs {format_amount(gap['rest_avg'], u, digits=0)})"
                )
            else:
                mag = format_magnitude(f.magnitude, finding=f)
            obs = h["what"]
            ev_lines = [f"- {k}: {v}" for k, v in evidence_pairs_plain(f.evidence)]
            method_txt = method_plain_es(f.method)
            alts, follows = plain_alts_and_followups(f)
            text = [
                title,
                f"Magnitud: {mag}",
                f"Método: {method_txt}",
                "",
                "Qué vimos:",
                obs,
                "",
                "Por qué importa:",
                h["why"],
                "",
                "Qué podés hacer:",
                *[f"- {a}" for a in h["do"]],
                "",
                "Explicaciones alternativas:",
                *[f"- {a}" for a in (alts or ["—"])],
                "",
                "Seguimiento sugerido:",
                *[f"- {a}" for a in (follows or ["—"])],
                "",
                "Limitaciones:",
                *[f"- {a}" for a in (plain_limitations(f) or ["—"])],
                "",
                "Evidencia (clave):",
                *(ev_lines or ["- —"]),
            ]
            if f.statistically_significant is not None:
                text.insert(4, f"Estadísticamente significativo: {bool_es(f.statistically_significant)}")
            if f.practically_important is not None:
                text.insert(5, f"Prácticamente importante: {bool_es(f.practically_important)}")
            ax.text(0.02, 0.98, "\n".join(text), va="top", ha="left", family="DejaVu Sans", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    "Limitaciones generales",
                    "",
                    "- Hallazgos exploratorios sujetos a comparaciones múltiples.",
                    "- Asociación ≠ causalidad.",
                    "- Significancia estadística ≠ importancia práctica.",
                    "- Validar siempre con conocimiento de dominio.",
                    "- Revisar muestras pequeñas y calidad de datos antes de decidir.",
                ]
            ),
            va="top",
            fontsize=12,
            family="DejaVu Sans",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path
