"""
Motor de descubrimiento automático.

Examina los datos y genera hallazgos priorizados con evidencia,
limitaciones y análisis de seguimiento. No inventa cifras.
"""

from __future__ import annotations

from typing import Any
import hashlib

import numpy as np
import pandas as pd

from estacion.core.profiler import profile_dataframe
from estacion.core.types import (
    DatasetProfile,
    Finding,
    FindingCategory,
    FindingSeverity,
)
from statsmodels.stats.multitest import multipletests

from estacion.stats.correlation import correlation_analysis
from estacion.stats.hypothesis import group_comparison
from estacion.patterns.anomalies import detect_anomalies
from estacion.patterns.temporal import temporal_analysis
from estacion.patterns.relationships import categorical_numeric_effects
from estacion.patterns.missingness import analyze_missingness
from estacion.clustering.segmenter import run_segmentation
from estacion.core.quality_rules import load_rules, evaluate_quality_rules


def _fid(*parts: Any) -> str:
    raw = "|".join(map(str, parts))
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _severity_from_score(score: float) -> FindingSeverity:
    if score >= 80:
        return FindingSeverity.CRITICAL
    if score >= 60:
        return FindingSeverity.NOTABLE
    if score >= 40:
        return FindingSeverity.INTERESTING
    return FindingSeverity.INFO


class DiscoveryEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        name: str = "dataset",
        profile: DatasetProfile | None = None,
        max_findings: int = 40,
        quality_domain: str = "default",
        fdr_alpha: float = 0.05,
    ):
        self.df = df
        self.name = name
        self.profile = profile or profile_dataframe(df, name=name)
        self.max_findings = max_findings
        self.quality_domain = quality_domain
        self.fdr_alpha = fdr_alpha
        self.findings: list[Finding] = []
        self.artifacts: dict[str, Any] = {}

    def run(self) -> list[Finding]:
        self.findings = []
        self._quality_findings()
        self._missingness_findings()
        self._distribution_findings()
        self._correlation_findings()
        self._group_effect_findings()
        self._anomaly_findings()
        self._temporal_findings()
        self._segmentation_findings()
        self._apply_global_fdr()
        self.findings.sort(key=lambda f: -f.score)
        self.findings = self.findings[: self.max_findings]
        return self.findings

    def _apply_global_fdr(self) -> None:
        """FDR Benjamini-Hochberg sobre todos los hallazgos con p_value."""
        indexed = [(i, f) for i, f in enumerate(self.findings) if f.p_value is not None and np.isfinite(f.p_value)]
        if not indexed:
            self.artifacts["fdr_global"] = {"n_tested": 0, "alpha": self.fdr_alpha}
            return
        pvals = [f.p_value for _, f in indexed]
        reject, p_adj, _, _ = multipletests(pvals, alpha=self.fdr_alpha, method="fdr_bh")
        for j, (i, _) in enumerate(indexed):
            self.findings[i].p_value_fdr = float(p_adj[j])
            self.findings[i].significant_fdr_global = bool(reject[j])
            # Recalibrar score: penalizar si no sobrevive FDR global
            if not reject[j] and self.findings[i].statistically_significant:
                self.findings[i].score = max(10, self.findings[i].score * 0.7)
                self.findings[i].limitations = list(self.findings[i].limitations) + [
                    f"No sobrevive FDR global (α={self.fdr_alpha}) entre hallazgos con p-valor."
                ]
            elif reject[j]:
                self.findings[i].score = min(100, self.findings[i].score + 5)
        self.artifacts["fdr_global"] = {
            "n_tested": len(indexed),
            "n_significant": int(sum(reject)),
            "alpha": self.fdr_alpha,
            "method": "fdr_bh",
        }

    def _add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def _missingness_findings(self) -> None:
        miss = analyze_missingness(self.df)
        self.artifacts["missingness"] = {
            "overall_missing_cell_rate": miss["overall_missing_cell_rate"],
            "little_mcar": miss["little_mcar"],
            "mar_screen": {
                k: v for k, v in miss["mar_screen"].items() if k != "tests"
            },
            "summary_top": miss["summary"].head(15).to_dict(orient="records"),
        }
        little = miss.get("little_mcar") or {}
        if little.get("ok") and little.get("reject_mcar_alpha05"):
            self._add(
                Finding(
                    id=_fid("mcar"),
                    title="Evidencia contra MCAR (test de Little)",
                    observation=little.get("interpretation", ""),
                    evidence=little,
                    magnitude=f"p={little.get('p_value'):.4g}",
                    method="Little MCAR (aproximado)",
                    alternative_explanations=[
                        "Missingness depende de variables observadas (MAR).",
                        "Missingness depende de no observados (MNAR).",
                        "Violación de normalidad que infla el estadístico.",
                    ],
                    follow_up=[
                        "Revisar pantalla MAR (missing indicator vs observados).",
                        "No imputar a ciegas; modelar el mecanismo si es posible.",
                    ],
                    category=FindingCategory.QUALITY,
                    severity=FindingSeverity.NOTABLE,
                    score=72,
                    p_value=float(little["p_value"]),
                    statistically_significant=True,
                    limitations=little.get("limitations", []),
                )
            )
        mar = miss.get("mar_screen") or {}
        for t in (mar.get("tests") or [])[:5]:
            if not t.get("significant_fdr"):
                continue
            score = min(80, 45 + (1 - t.get("p_fdr", 1)) * 30)
            self._add(
                Finding(
                    id=_fid("mar", t["missing_col"], t["predictor"]),
                    title=(
                        f"Missingness de '{t['missing_col']}' asociada a '{t['predictor']}'"
                    ),
                    observation=(
                        f"El indicador de faltantes de '{t['missing_col']}' se asocia con "
                        f"'{t['predictor']}' ({t['method']}, p FDR={t['p_fdr']:.4g}). "
                        f"Compatible con MAR (o MNAR con proxy observado)."
                    ),
                    evidence=t,
                    magnitude=f"p_fdr={t['p_fdr']:.4g}",
                    method=t["method"] + " + FDR",
                    alternative_explanations=[
                        "Relación espuria por otra variable.",
                        "MNAR donde el predictor es proxy del valor faltante.",
                    ],
                    follow_up=[
                        "Inspeccionar tasas de missing por niveles del predictor.",
                        "Considerar imputación condicional / modelos de selección.",
                    ],
                    category=FindingCategory.QUALITY,
                    severity=_severity_from_score(score),
                    score=score,
                    variables=[t["missing_col"], t["predictor"]],
                    p_value=float(t["p_value"]),
                    statistically_significant=True,
                    practically_important=True,
                )
            )

    def _quality_findings(self) -> None:
        rules = load_rules(domain=self.quality_domain)
        qres = evaluate_quality_rules(self.df, rules)
        self.artifacts["quality_rules"] = qres
        for v in qres.get("violations", [])[:12]:
            sev = {
                "critical": FindingSeverity.CRITICAL,
                "notable": FindingSeverity.NOTABLE,
                "interesting": FindingSeverity.INTERESTING,
            }.get(v.get("severity", "interesting"), FindingSeverity.INFO)
            score = {"critical": 85, "notable": 65, "interesting": 45}.get(v.get("severity", ""), 30)
            self._add(
                Finding(
                    id=_fid("rule", v.get("rule"), v.get("column"), v.get("detail")),
                    title=f"Regla de calidad ({rules.name}): {v.get('rule')}",
                    observation=v.get("detail", ""),
                    evidence=v,
                    magnitude=str(v.get("value", "—")),
                    method=f"quality_rules[{rules.name}]",
                    alternative_explanations=[
                        "La regla del dominio es demasiado estricta para este dataset.",
                        "Error real de carga/proceso.",
                    ],
                    follow_up=["Revisar la columna/regla y ajustar umbrales si el dominio lo justifica."],
                    category=FindingCategory.QUALITY,
                    severity=sev,
                    score=score,
                    variables=[v["column"]] if v.get("column") else [],
                )
            )

        p = self.profile
        if p.duplicate_pct >= 0.01:
            score = min(90, 40 + p.duplicate_pct * 100)
            self._add(
                Finding(
                    id=_fid("dup", p.duplicate_rows),
                    title="Filas duplicadas detectadas",
                    observation=(
                        f"Se observaron {p.duplicate_rows} filas duplicadas "
                        f"({p.duplicate_pct:.1%} del total)."
                    ),
                    evidence={
                        "duplicate_rows": p.duplicate_rows,
                        "duplicate_pct": p.duplicate_pct,
                        "n_rows": p.n_rows,
                    },
                    magnitude=f"{p.duplicate_pct:.1%} duplicadas",
                    method="pandas.DataFrame.duplicated",
                    alternative_explanations=[
                        "Duplicados legítimos (eventos repetidos).",
                        "Error de carga/concatenación.",
                        "Clave de negocio incompleta (no son duplicados semánticos).",
                    ],
                    follow_up=[
                        "Definir clave de negocio y medir duplicados semánticos.",
                        "Inspeccionar muestra de duplicados antes de eliminar.",
                    ],
                    category=FindingCategory.QUALITY,
                    severity=_severity_from_score(score),
                    score=score,
                    limitations=["duplicated() usa igualdad exacta de todas las columnas."],
                )
            )

        for v in p.variables:
            if v.missing_pct >= 0.2:
                score = min(85, 30 + v.missing_pct * 80)
                self._add(
                    Finding(
                        id=_fid("miss", v.name),
                        title=f"Valores faltantes elevados en '{v.name}'",
                        observation=(
                            f"La variable '{v.name}' tiene {v.missing_pct:.1%} de valores faltantes "
                            f"({v.n_missing} de {p.n_rows})."
                        ),
                        evidence={
                            "column": v.name,
                            "missing_pct": v.missing_pct,
                            "n_missing": v.n_missing,
                            "role": v.role.value,
                        },
                        magnitude=f"{v.missing_pct:.1%} faltantes",
                        method="conteo de NA",
                        alternative_explanations=[
                            "Missingness estructural (no aplica a ciertos casos).",
                            "Fallo de recolección / integración.",
                            "Campo opcional en el proceso origen.",
                        ],
                        follow_up=[
                            "Analizar missingness vs otras variables (MCAR/MAR/MNAR).",
                            "No imputar automáticamente sin entender el mecanismo.",
                        ],
                        category=FindingCategory.QUALITY,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[v.name],
                    )
                )

    def _distribution_findings(self) -> None:
        for v in self.profile.variables:
            if v.role.value not in {"numeric", "binary"}:
                continue
            skew = v.stats.get("skewness")
            if skew is not None and abs(skew) >= 2:
                score = min(88, 45 + abs(skew) * 8)
                self._add(
                    Finding(
                        id=_fid("skew", v.name),
                        title=f"Distribución muy asimétrica: '{v.name}'",
                        observation=(
                            f"'{v.name}' presenta asimetría {skew:.2f}. "
                            f"Media={v.stats.get('mean'):.4g}, mediana={v.stats.get('p50'):.4g}."
                        ),
                        evidence={
                            "skewness": skew,
                            "mean": v.stats.get("mean"),
                            "median": v.stats.get("p50"),
                            "p99": v.stats.get("p99"),
                            "max": v.stats.get("max"),
                        },
                        magnitude=f"|skew|={abs(skew):.2f}",
                        method="pandas.Series.skew / percentiles",
                        alternative_explanations=[
                            "Proceso generador naturally heavy-tailed (p.ej. montos).",
                            "Outliers / errores de carga (ceros extra, unidades mezcladas).",
                            "Mezcla de subpoblaciones.",
                        ],
                        follow_up=[
                            "Inspeccionar cola superior e inferior.",
                            "Considerar transformaciones (log) solo para modelado, no para ocultar.",
                            "Segmentar y ver si la asimetría desaparece dentro de grupos.",
                        ],
                        category=FindingCategory.DISTRIBUTION,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[v.name],
                        practically_important=True,
                    )
                )
            out_pct = v.stats.get("iqr_outlier_pct")
            if out_pct is not None and out_pct >= 0.05:
                score = min(75, 35 + out_pct * 200)
                self._add(
                    Finding(
                        id=_fid("iqr", v.name),
                        title=f"Concentración de valores extremos en '{v.name}'",
                        observation=(
                            f"Según regla IQR 1.5, {out_pct:.1%} de los valores de '{v.name}' "
                            f"son extremos."
                        ),
                        evidence={
                            "iqr_outlier_pct": out_pct,
                            "iqr_outlier_count": v.stats.get("iqr_outlier_count"),
                            "lower": v.stats.get("iqr_lower"),
                            "upper": v.stats.get("iqr_upper"),
                        },
                        magnitude=f"{out_pct:.1%} extremos IQR",
                        method="IQR 1.5",
                        alternative_explanations=[
                            "Distribución legítimamente leptocúrtica.",
                            "Errores de digitación / unidades inconsistentes.",
                            "La regla IQR no es óptima para distribuciones asimétricas.",
                        ],
                        follow_up=[
                            "Contrastar con IsolationForest multivariado.",
                            "Revisar dominio permitido de la variable.",
                        ],
                        category=FindingCategory.ANOMALY,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[v.name],
                        limitations=["Outlier ≠ error; depende del método y del dominio."],
                    )
                )
            top_share = v.stats.get("top_share")
            if top_share and top_share >= 0.85 and v.role.value == "categorical":
                pass  # handled below via categorical branch

        for v in self.profile.variables:
            if v.role.value not in {"categorical", "binary"}:
                continue
            top_share = v.stats.get("top_share")
            mode = v.stats.get("mode")
            if top_share and top_share >= 0.85:
                score = min(70, 30 + top_share * 40)
                self._add(
                    Finding(
                        id=_fid("dom", v.name),
                        title=f"Categoría dominante en '{v.name}'",
                        observation=(
                            f"El nivel '{mode}' concentra {top_share:.1%} de los valores no nulos."
                        ),
                        evidence={"top_share": top_share, "mode": mode, "n_levels": v.stats.get("n_levels")},
                        magnitude=f"{top_share:.1%} en '{mode}'",
                        method="value_counts",
                        alternative_explanations=[
                            "Desbalance real de la población.",
                            "Default de sistema / valor sentinela.",
                        ],
                        follow_up=[
                            "Ver si el desbalance se asocia a la variable objetivo.",
                            "Evitar métricas engañosas en clases raras.",
                        ],
                        category=FindingCategory.DISTRIBUTION,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[v.name],
                    )
                )

    def _correlation_findings(self) -> None:
        nums = self.profile.numeric_columns[:25]
        if len(nums) < 2:
            return
        corr = correlation_analysis(self.df, columns=nums, include_distance=True)
        self.artifacts["correlation"] = corr
        for pair in corr["pairs"][:15]:
            if not pair.get("significant_fdr") and not pair.get("nonlinear_hint"):
                if not (pair.get("practically_important") and abs(pair.get("spearman_r") or 0) >= 0.5):
                    continue
            r = pair.get("spearman_r") or 0
            score = 0
            if pair.get("significant_fdr"):
                score += 35
            if pair.get("practically_important"):
                score += 30
            score += min(30, abs(r) * 40)
            if pair.get("nonlinear_hint"):
                score += 15
                title = (
                    f"Posible relación no lineal: '{pair['var_a']}' ↔ '{pair['var_b']}'"
                )
                obs = (
                    f"Distance correlation={pair.get('distance_corr'):.3f} con "
                    f"Pearson r={pair.get('pearson_r'):.3f}, sugiriendo dependencia no lineal."
                )
            else:
                title = f"Correlación: '{pair['var_a']}' ↔ '{pair['var_b']}'"
                obs = (
                    f"Spearman ρ={r:.3f} (p FDR={pair.get('spearman_p_fdr'):.4g}, n={pair['n']})."
                )
            self._add(
                Finding(
                    id=_fid("corr", pair["var_a"], pair["var_b"]),
                    title=title,
                    observation=obs,
                    evidence=pair,
                    magnitude=f"|ρ|={abs(r):.3f}",
                    method="Spearman + FDR-BH" + (" + distance correlation" if pair.get("nonlinear_hint") else ""),
                    alternative_explanations=[
                        "Confusor común (correlación espuria).",
                        "Causalidad en cualquiera de los dos sentidos.",
                        "Artefacto por outliers o ties.",
                    ],
                    follow_up=[
                        "Scatterplot y residuales.",
                        "Ajuste parcial por covariables candidatas.",
                        "No interpretar como causalidad.",
                    ],
                    category=FindingCategory.RELATIONSHIP,
                    severity=_severity_from_score(score),
                    score=min(95, score),
                    variables=[pair["var_a"], pair["var_b"]],
                    statistically_significant=bool(pair.get("significant_fdr")),
                    practically_important=bool(pair.get("practically_important")),
                    p_value=float(pair["spearman_p"]) if pair.get("spearman_p") is not None else None,
                    limitations=[
                        "Correlación ≠ causalidad.",
                        "FDR controla tasa de falsos descubrimientos entre pares testeados.",
                    ],
                )
            )

    def _group_effect_findings(self) -> None:
        cats = [c for c in self.profile.categorical_columns if self.df[c].nunique() <= 12][:8]
        nums = self.profile.numeric_columns[:8]
        effects = categorical_numeric_effects(self.df, cats, nums, max_tests=30)
        self.artifacts["group_effects"] = effects
        for res in effects[:10]:
            if not res.get("significant") and not res.get("practically_important"):
                continue
            effect = res.get("effect_size_cohens_d") or res.get("effect_size_eta_sq")
            score = res.get("priority_score", 40)
            self._add(
                Finding(
                    id=_fid("grp", res["value_col"], res["group_col"]),
                    title=(
                        f"Diferencias entre grupos de '{res['group_col']}' "
                        f"en '{res['value_col']}'"
                    ),
                    observation=(
                        f"Prueba {res['method']}: p={res['p_value']:.4g}. "
                        f"Medias por grupo: {res.get('group_means')}. "
                        f"Tamaño del efecto={effect}."
                    ),
                    evidence=res,
                    magnitude=f"effect={effect}",
                    method=res["method"],
                    alternative_explanations=[
                        "Confusión por otras variables.",
                        "Selección / sesgo de muestreo.",
                        "Heterogeneidad de varianzas / n desbalanceado.",
                    ],
                    follow_up=[
                        "Comparaciones pairwise con FDR.",
                        "Regresión ajustada por covariables.",
                        "Verificar significancia práctica además de p-valor.",
                    ],
                    category=FindingCategory.STATISTICAL,
                    severity=_severity_from_score(score),
                    score=min(92, score),
                    variables=[res["value_col"], res["group_col"]],
                    statistically_significant=bool(res.get("significant")),
                    practically_important=bool(res.get("practically_important")),
                    p_value=float(res["p_value"]) if res.get("p_value") is not None else None,
                    limitations=res.get("warnings", []),
                )
            )

        # Concentración: una categoría explica proporción desproporcionada del total
        for cat in cats[:5]:
            for num in nums[:5]:
                sub = self.df[[cat, num]].dropna()
                sub[num] = pd.to_numeric(sub[num], errors="coerce")
                sub = sub.dropna()
                if len(sub) < 30 or (sub[num] < 0).any():
                    continue
                total = sub[num].sum()
                if total <= 0:
                    continue
                shares = sub.groupby(cat)[num].sum() / total
                top_cat = shares.idxmax()
                top_share = float(shares.max())
                count_share = float((sub[cat] == top_cat).mean())
                if top_share >= 0.4 and top_share >= count_share * 1.5:
                    score = min(90, 40 + top_share * 50)
                    self._add(
                        Finding(
                            id=_fid("conc", cat, num, top_cat),
                            title=f"Concentración: '{top_cat}' domina '{num}'",
                            observation=(
                                f"La categoría '{top_cat}' de '{cat}' concentra {top_share:.1%} de "
                                f"'{num}', pese a representar {count_share:.1%} de las filas."
                            ),
                            evidence={
                                "category_col": cat,
                                "value_col": num,
                                "top_category": str(top_cat),
                                "value_share": top_share,
                                "row_share": count_share,
                                "lift": top_share / max(count_share, 1e-9),
                            },
                            magnitude=f"{top_share:.1%} del total",
                            method="suma agrupada / participación relativa",
                            alternative_explanations=[
                                "Pocos registros de alto valor (Pareto).",
                                "Error de unidades en esa categoría.",
                            ],
                            follow_up=[
                                "Análisis Pareto / Lorenz.",
                                "Robustecer con mediana además de suma.",
                            ],
                            category=FindingCategory.RELATIONSHIP,
                            severity=_severity_from_score(score),
                            score=score,
                            variables=[cat, num],
                            practically_important=True,
                        )
                    )

    def _anomaly_findings(self) -> None:
        nums = self.profile.numeric_columns[:15]
        if len(nums) < 2:
            return
        anom = detect_anomalies(self.df, numeric_cols=nums)
        self.artifacts["anomalies"] = anom
        mv = anom.get("multivariate") or {}
        if mv.get("ok") and mv.get("pct", 0) > 0:
            score = min(80, 40 + mv["pct"] * 400)
            self._add(
                Finding(
                    id=_fid("iforest"),
                    title="Anomalías multivariadas detectadas",
                    observation=(
                        f"IsolationForest marcó {mv['n_anomalies']} filas "
                        f"({mv['pct']:.1%}) como anómalas."
                    ),
                    evidence=mv,
                    magnitude=f"{mv['pct']:.1%} anomalías",
                    method="IsolationForest",
                    alternative_explanations=[
                        "Cola natural de la distribución conjunta.",
                        "contamination inadecuado (hiperparámetro).",
                        "Columnas con escalas/roles inadecuados incluidos.",
                    ],
                    follow_up=[
                        "Inspeccionar filas anómalas y contrastar con reglas de negocio.",
                        "Repetir con contamination alternativo (sensibilidad).",
                        "Anomalías contextuales por segmento.",
                    ],
                    category=FindingCategory.ANOMALY,
                    severity=_severity_from_score(score),
                    score=score,
                    variables=nums[:10],
                    limitations=mv.get("limitations", []),
                )
            )

    def _temporal_findings(self) -> None:
        if not self.profile.datetime_columns or not self.profile.numeric_columns:
            return
        date_col = self.profile.datetime_columns[0]
        for value_col in self.profile.numeric_columns[:3]:
            res = temporal_analysis(self.df, date_col, value_col)
            if not res.get("ok"):
                continue
            self.artifacts.setdefault("temporal", {})[value_col] = res
            trend = res["trend"]
            if trend.get("significant") and abs(trend.get("r_value") or 0) >= 0.3:
                score = min(88, 50 + abs(trend["r_value"]) * 40)
                self._add(
                    Finding(
                        id=_fid("trend", date_col, value_col),
                        title=f"Tendencia temporal en '{value_col}'",
                        observation=(
                            f"Tendencia {trend['direction']} significativa "
                            f"(pendiente={trend['slope_per_step']:.4g}, "
                            f"r={trend['r_value']:.3f}, p={trend['p_value']:.4g})."
                        ),
                        evidence=res,
                        magnitude=f"r={trend['r_value']:.3f}",
                        method=f"regresión lineal sobre serie agregada ({res['freq']})",
                        alternative_explanations=[
                            "Cambio de composición de la muestra en el tiempo.",
                            "Efecto de calendario / estacionalidad no modelada.",
                            "Break puntual interpretado como tendencia.",
                        ],
                        follow_up=[
                            "Descomponer estacionalidad.",
                            "Evaluar change-point reportado.",
                        ],
                        category=FindingCategory.TEMPORAL,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[date_col, value_col],
                        statistically_significant=True,
                        practically_important=abs(trend["r_value"]) >= 0.5,
                        p_value=float(trend["p_value"]),
                        limitations=res.get("limitations", []),
                    )
                )
            cp = res.get("change_point")
            if cp and cp.get("p_value", 1) < 0.01:
                delta = cp["mean_after"] - cp["mean_before"]
                score = 70
                self._add(
                    Finding(
                        id=_fid("cp", date_col, value_col, cp["timestamp"]),
                        title=f"Posible cambio estructural en '{value_col}'",
                        observation=(
                            f"Alrededor de {cp['timestamp']}, la media pasa de "
                            f"{cp['mean_before']:.4g} a {cp['mean_after']:.4g} "
                            f"(Δ={delta:.4g}, p={cp['p_value']:.4g})."
                        ),
                        evidence=cp,
                        magnitude=f"Δ={delta:.4g}",
                        method="búsqueda de corte + t-test",
                        alternative_explanations=[
                            "Cambio operativo real.",
                            "Cambio de definición de la métrica / ETL.",
                            "Outlier temporal que desplaza el óptimo.",
                        ],
                        follow_up=[
                            "Validar con conocimiento de negocio en esa fecha.",
                            "Contrastar con PELT multi-cambio del mismo artefacto.",
                        ],
                        category=FindingCategory.TEMPORAL,
                        severity=FindingSeverity.NOTABLE,
                        score=score,
                        variables=[date_col, value_col],
                        statistically_significant=True,
                        p_value=float(cp["p_value"]),
                    )
                )
            pelt = res.get("changepoints_pelt") or {}
            mean_block = pelt.get("mean") if pelt.get("method") == "PELT_mean_and_variance" else pelt
            var_block = pelt.get("variance") if pelt.get("method") == "PELT_mean_and_variance" else None
            ncp = (
                pelt.get("n_changepoints_mean")
                or (mean_block or {}).get("n_changepoints")
                or 0
            )
            if pelt.get("ok") and ncp >= 1:
                score = min(82, 50 + ncp * 8)
                examples = (mean_block or {}).get("changepoints", pelt.get("changepoints", []))[:3]
                self._add(
                    Finding(
                        id=_fid("pelt", date_col, value_col, ncp),
                        title=f"Change-points de media (PELT) en '{value_col}'",
                        observation=(
                            f"PELT L2 detectó {ncp} cambio(s) de media. "
                            f"Ejemplos: {examples}."
                        ),
                        evidence=pelt,
                        magnitude=f"{ncp} cortes",
                        method="PELT L2",
                        alternative_explanations=[
                            "Cambios reales de nivel.",
                            "Outliers que inducen cortes espurios.",
                            "Penalización automática inadecuada.",
                        ],
                        follow_up=[
                            "Revisar segmentos y validar fechas con dominio.",
                            "Probar sensibilidad a la penalización.",
                        ],
                        category=FindingCategory.TEMPORAL,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[date_col, value_col],
                        practically_important=ncp >= 1,
                        limitations=pelt.get("limitations", []),
                    )
                )
            ncv = pelt.get("n_changepoints_variance") or (var_block or {}).get("n_changepoints") or 0
            if pelt.get("ok") and ncv >= 1:
                score = min(78, 48 + ncv * 8)
                self._add(
                    Finding(
                        id=_fid("peltvar", date_col, value_col, ncv),
                        title=f"Change-points de varianza (PELT) en '{value_col}'",
                        observation=(
                            f"PELT de varianza detectó {ncv} cambio(s) de volatilidad. "
                            f"Segmentos: {(var_block or {}).get('segments', [])[:3]}."
                        ),
                        evidence=var_block or pelt,
                        magnitude=f"{ncv} cortes de σ",
                        method="PELT_VAR",
                        alternative_explanations=[
                            "Cambio real de volatilidad / régimen.",
                            "Heterocedasticidad por composición de muestra.",
                            "Penalización inadecuada.",
                        ],
                        follow_up=[
                            "Comparar con cortes de media.",
                            "Modelar GARCH/heterocedasticidad si aplica.",
                        ],
                        category=FindingCategory.TEMPORAL,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[date_col, value_col],
                        practically_important=True,
                        limitations=[
                            "Detecta cambios de varianza; interpretar con contexto.",
                        ],
                    )
                )
            stl = res.get("stl") or {}
            if stl.get("ok") and stl.get("detected_seasonality"):
                strength = stl.get("seasonal_strength") or 0
                score = min(80, 45 + strength * 40)
                self._add(
                    Finding(
                        id=_fid("stl", date_col, value_col),
                        title=f"Estacionalidad (STL) en '{value_col}'",
                        observation=(
                            f"Descomposición {stl.get('method')} con periodo={stl.get('period')}: "
                            f"fuerza estacional≈{strength:.2f}, "
                            f"acf@periodo={stl.get('seasonal_acf_period')}."
                        ),
                        evidence=stl,
                        magnitude=f"strength={strength:.2f}",
                        method=str(stl.get("method")),
                        alternative_explanations=[
                            "Periodo mal especificado.",
                            "Tendencia residual confundida con estacionalidad.",
                        ],
                        follow_up=[
                            "Graficar componentes trend/seasonal/resid.",
                            "Modelar con estacionalidad explícita si se pronostica.",
                        ],
                        category=FindingCategory.TEMPORAL,
                        severity=_severity_from_score(score),
                        score=score,
                        variables=[date_col, value_col],
                        practically_important=strength >= 0.4,
                        limitations=stl.get("limitations", []),
                    )
                )

    def _segmentation_findings(self) -> None:
        nums = self.profile.numeric_columns[:10]
        cats = [c for c in self.profile.categorical_columns if self.df[c].nunique() <= 10][:4]
        if len(nums) < 2:
            return
        seg = run_segmentation(self.df, nums, cats)
        self.artifacts["segmentation"] = seg
        if not seg.get("ok"):
            return
        best = seg.get("kmeans", {}).get("best") or {}
        sil = best.get("silhouette")
        if seg.get("natural_groups_likely") and sil is not None:
            score = min(85, 40 + sil * 80)
            self._add(
                Finding(
                    id=_fid("seg", best.get("k")),
                    title=f"Estructura de segmentos sugerida (k={best.get('k')})",
                    observation=(
                        f"K-means con k={best.get('k')} obtiene silhouette={sil:.3f}. "
                        f"{seg.get('caveat')}"
                    ),
                    evidence={
                        "best": best,
                        "stability": seg.get("stability"),
                        "hierarchical": seg.get("hierarchical"),
                        "dbscan_best": seg.get("dbscan", {}).get("best"),
                    },
                    magnitude=f"silhouette={sil:.3f}",
                    method="K-means + silhouette; contraste jerárquico/DBSCAN",
                    alternative_explanations=[
                        "Artefacto de escala/encoding.",
                        "Clusters no interpretables de negocio.",
                        "Sobresegmentación.",
                    ],
                    follow_up=[
                        "Perfilar centroides / medianas por cluster.",
                        "Evaluar estabilidad (ARI entre semillas).",
                        "Validar con variables externas no usadas en el clustering.",
                    ],
                    category=FindingCategory.SEGMENT,
                    severity=_severity_from_score(score),
                    score=score,
                    variables=nums[:8],
                    limitations=seg.get("limitations", []),
                    practically_important=sil >= 0.45,
                )
            )
        elif sil is not None and sil < 0.25:
            self._add(
                Finding(
                    id=_fid("noseg"),
                    title="No hay evidencia fuerte de grupos naturales",
                    observation=(
                        f"El mejor silhouette de K-means fue {sil:.3f} (<0.25). "
                        "Es posible que no existan segmentos compactos en el espacio analizado."
                    ),
                    evidence={"best": best, "caveat": seg.get("caveat")},
                    magnitude=f"silhouette={sil:.3f}",
                    method="K-means grid + silhouette",
                    alternative_explanations=[
                        "Features inadecuadas / falta de variables relevantes.",
                        "Estructura no esférica (probar DBSCAN/density).",
                    ],
                    follow_up=[
                        "Revisar resultados DBSCAN.",
                        "No forzar k por convención de negocio sin validación.",
                    ],
                    category=FindingCategory.SEGMENT,
                    severity=FindingSeverity.INFO,
                    score=25,
                    variables=nums[:8],
                    limitations=seg.get("limitations", []),
                )
            )


def run_discovery(
    df: pd.DataFrame,
    name: str = "dataset",
    quality_domain: str = "default",
    fdr_alpha: float = 0.05,
) -> tuple[list[Finding], DatasetProfile, dict]:
    engine = DiscoveryEngine(
        df, name=name, quality_domain=quality_domain, fdr_alpha=fdr_alpha
    )
    findings = engine.run()
    return findings, engine.profile, engine.artifacts
