"""Export PDF con figuras estáticas (matplotlib)."""

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

from estacion.core.types import DatasetProfile, Finding


def _wrap(ax, title: str):
    ax.set_title(title, fontsize=11, loc="left", color="#0F4C5C")
    ax.grid(True, alpha=0.25)


def _fmt_int(n: Any) -> str:
    try:
        from estacion.ux.presentation import format_int_es

        return format_int_es(n)
    except Exception:
        return str(n)


def generate_pdf_report(
    df: pd.DataFrame,
    profile: DatasetProfile,
    findings: list[Finding],
    output_path: str | Path,
    artifacts: dict[str, Any] | None = None,
    max_findings: int = 12,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts or {}

    with PdfPages(output_path) as pdf:
        # Portada / resumen
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape-ish
        fig.patch.set_facecolor("#F7F4EF")
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "Estación Universal de Análisis de Datos",
            f"Informe: {profile.name}",
            f"Generado: {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"Filas: {_fmt_int(profile.n_rows)}   Columnas: {_fmt_int(profile.n_cols)}   Memoria: {profile.memory_mb} MB",
            f"Duplicados: {_fmt_int(profile.duplicate_rows)} ({profile.duplicate_pct:.1%})",
            f"Hallazgos priorizados: {_fmt_int(len(findings))}",
            "",
            "Resumen en lenguaje simple:",
        ]
        try:
            from estacion.ux.presentation import (
                build_plain_summary,
                dedupe_findings,
                plain_finding_line,
            )

            for b in build_plain_summary(findings, profile, artifacts):
                lines.append(f"• {b}")
            lines.append("")
            lines.append("Hallazgos principales:")
            for i, f in enumerate(dedupe_findings(findings)[:8], 1):
                lines.append(f"{i}. {plain_finding_line(f)}")
        except Exception:
            lines.append("Hallazgos principales:")
            for i, f in enumerate(findings[:8], 1):
                lines.append(f"{i}. [{f.severity.value}] {f.title} — {f.magnitude}")
        lines += [
            "",
            "Nota: los cálculos son estadísticos; correlación ≠ causalidad.",
            "Este PDF es exploratorio y debe interpretarse con conocimiento de dominio.",
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="DejaVu Sans", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Missingness bar
        miss = pd.Series({v.name: v.missing_pct for v in profile.variables}).sort_values(ascending=False)
        miss = miss[miss > 0].head(20)
        if len(miss):
            fig, ax = plt.subplots(figsize=(11, 6))
            miss.plot(kind="bar", ax=ax, color="#0F4C5C")
            _wrap(ax, "Proporción de valores faltantes (top)")
            ax.set_ylabel("missing %")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Histograms de numéricas top
        nums = profile.numeric_columns[:4]
        if nums:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axes = axes.ravel()
            for i, col in enumerate(nums):
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                axes[i].hist(s, bins=30, color="#E36414", alpha=0.85, edgecolor="white")
                _wrap(axes[i], f"Distribución: {col}")
            for j in range(len(nums), 4):
                axes[j].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Correlación
        corr_art = artifacts.get("correlation") or {}
        mat = corr_art.get("spearman_matrix")
        if isinstance(mat, pd.DataFrame) and mat.shape[0] >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mat.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(mat.columns)))
            ax.set_yticks(range(len(mat.index)))
            ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
            ax.set_yticklabels(mat.index, fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
            _wrap(ax, "Matriz Spearman")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Clusters PCA si hay
        seg = artifacts.get("segmentation") or {}
        coords = seg.get("pca_coords")
        labels = (seg.get("kmeans") or {}).get("labels")
        if coords is not None and labels is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            labs = np.asarray(labels)
            for lab in sorted(set(labs.tolist())):
                mask = labs == lab
                ax.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.7, label=str(lab))
            ax.legend(title="cluster", fontsize=8)
            _wrap(ax, "Segmentos (PCA 2D)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Página por hallazgos
        for f in findings[:max_findings]:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.axis("off")
            try:
                from estacion.ux.presentation import (
                    humanize_finding,
                    format_magnitude,
                    evidence_pairs_plain,
                    CATEGORY_LABELS_ES,
                    SEVERITY_LABELS_ES,
                    bool_es,
                    method_plain_es,
                )

                h = humanize_finding(f)
                title = h["title_simple"]
                mag = format_magnitude(f.magnitude, finding=f)
                obs = h["what"]
                ev_lines = [f"- {k}: {v}" for k, v in evidence_pairs_plain(f.evidence)]
                cat = CATEGORY_LABELS_ES.get(f.category.value, f.category.value)
                sev = SEVERITY_LABELS_ES.get(f.severity.value, f.severity.value)
                method_txt = method_plain_es(f.method)
                sig_line = None
                if f.statistically_significant is not None:
                    sig_line = f"Estadísticamente significativo: {bool_es(f.statistically_significant)}"
                prac_line = None
                if f.practically_important is not None:
                    prac_line = f"Prácticamente importante: {bool_es(f.practically_important)}"
            except Exception:
                title = f.title
                mag = str(f.magnitude)
                obs = f.observation
                ev_lines = []
                cat = f.category.value
                sev = f.severity.value
                method_txt = str(f.method)
                sig_line = None
                prac_line = None
            text = [
                title,
                f"Categoría: {cat} | Severidad: {sev} | Score: {f.score:.0f}",
                f"Magnitud: {mag}",
                f"Método: {method_txt}",
                *([sig_line] if sig_line else []),
                *([prac_line] if prac_line else []),
                "",
                "Observación:",
                obs,
                "",
                "Explicaciones alternativas:",
                *[f"- {a}" for a in f.alternative_explanations],
                "",
                "Seguimiento sugerido:",
                *[f"- {a}" for a in f.follow_up],
                "",
                "Limitaciones:",
                *[f"- {a}" for a in (f.limitations or ["—"])],
                "",
                "Evidencia (clave):",
                *(ev_lines or ["- —"]),
            ]
            ax.text(0.02, 0.98, "\n".join(text), va="top", ha="left", family="DejaVu Sans", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Limitaciones generales
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
