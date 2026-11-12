"""
Análisis de sensibilidad causal exploratorio.

- E-value (VanderWeele / Ding) para asociaciones observadas
- Cotas de Rosenbaum (Γ) para estudios pareados / signos de diferencias

Ninguno demuestra o refuta causalidad; cuantifican cuán fuerte debería ser
un confusor no medido para explicar el resultado.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def evalue_from_risk_ratio(rr: float, rare_outcome: bool = True) -> dict[str, Any]:
    """
    E-value para un risk ratio (o approx OR si outcome raro).

    E-value = RR + sqrt(RR * (RR - 1)) para RR >= 1.
    """
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return {"ok": False, "reason": "RR inválido"}
    rr_use = float(rr) if rr >= 1 else float(1 / rr)
    if rr_use < 1:
        return {"ok": False, "reason": "RR inválido"}
    if abs(rr_use - 1) < 1e-12:
        evalue = 1.0
    else:
        evalue = rr_use + np.sqrt(rr_use * (rr_use - 1))
    return {
        "ok": True,
        "method": "e_value_risk_ratio",
        "rr_point": float(rr),
        "rr_used_ge1": rr_use,
        "e_value": float(evalue),
        "rare_outcome_assumption": rare_outcome,
        "interpretation": (
            f"Un confusor no medido debería asociarse al treatment y al outcome con "
            f"RR≥{evalue:.2f} cada uno (más allá de lo ajustado) para explicar por completo "
            f"la asociación observada."
        ),
        "limitations": [
            "E-value no identifica el confusor; solo acota su fuerza mínima hipotética.",
            "Si se usa OR como RR, la aproximación es mejor con outcome raro.",
            "No reemplaza diseño ni medición de confusores.",
        ],
    }


def evalue_from_continuous_effect(
    effect: float,
    se: float | None = None,
    sd_outcome: float | None = None,
) -> dict[str, Any]:
    """
    Aproximación: convierte efecto continuo a RR vía d de Cohen / fórmula de Ding.

    RR ≈ exp(0.91 * d) con d = effect / sd.
    """
    if sd_outcome is None or sd_outcome <= 0:
        return {"ok": False, "reason": "Se necesita sd_outcome > 0"}
    d = abs(float(effect) / float(sd_outcome))
    rr_approx = float(np.exp(0.91 * d))
    base = evalue_from_risk_ratio(rr_approx, rare_outcome=True)
    if not base.get("ok"):
        return base
    base["method"] = "e_value_from_standardized_mean_diff"
    base["cohens_d_approx"] = d
    base["rr_approx"] = rr_approx
    if se is not None and se > 0:
        # E-value para límite inferior del IC 95% (si se aleja de 0)
        lo = abs(effect) - 1.96 * se
        if lo > 0:
            d_lo = lo / sd_outcome
            rr_lo = float(np.exp(0.91 * d_lo))
            ev_lo = evalue_from_risk_ratio(rr_lo)
            base["e_value_ci_lower"] = ev_lo.get("e_value")
        else:
            base["e_value_ci_lower"] = 1.0
            base["ci_crosses_null"] = True
    return base


def rosenbaum_bounds_signed(
    treated_outcomes: np.ndarray,
    control_outcomes: np.ndarray,
    gamma_grid: list[float] | None = None,
) -> dict[str, Any]:
    """
    Cotas de Rosenbaum para test de signos sobre diferencias pareadas.

    Γ=1 → sin sesgo oculto. Se busca el Γ donde el p-value máximo supera 0.05.
    """
    y_t = np.asarray(treated_outcomes, dtype=float)
    y_c = np.asarray(control_outcomes, dtype=float)
    if len(y_t) != len(y_c) or len(y_t) < 10:
        return {"ok": False, "reason": "Se requieren ≥10 pares tratados/controles"}
    diffs = y_t - y_c
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n < 10:
        return {"ok": False, "reason": "Demasiados empates; n efectivo < 10"}
    n_pos = int((diffs > 0).sum())
    # p bajo sesgo (Γ=1): binomial bilateral approx con p=0.5
    p_unbiased = float(stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue)

    if gamma_grid is None:
        gamma_grid = [1.0, 1.1, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

    rows = []
    critical_gamma = None
    for g in gamma_grid:
        # Bajo sesgo Γ, la probabilidad de signo positivo está entre 1/(1+Γ) y Γ/(1+Γ)
        p_max = g / (1 + g)
        p_min = 1 / (1 + g)
        # p-value máximo (peor caso) usando la cola más conservadora
        p_hi = float(stats.binomtest(n_pos, n, p_max, alternative="greater").pvalue)
        p_lo = float(stats.binomtest(n_pos, n, p_min, alternative="less").pvalue)
        # bilateral grosero
        p_worst = min(1.0, 2 * min(p_hi, p_lo)) if False else float(max(p_hi, p_lo))
        # Mejor: p-value del test de signos bajo el sesgo más adverso
        # Usamos el p de la hipótesis que más infla el p hacia no-significancia
        if n_pos >= n / 2:
            p_worst = float(stats.binomtest(n_pos, n, p_max, alternative="greater").pvalue)
        else:
            p_worst = float(stats.binomtest(n_pos, n, p_min, alternative="less").pvalue)
        rows.append({"gamma": g, "p_value_upper_bound": p_worst, "significant_at_05": p_worst < 0.05})
        if p_worst >= 0.05 and critical_gamma is None and g > 1:
            critical_gamma = g

    return {
        "ok": True,
        "method": "rosenbaum_sign_bounds",
        "n_pairs_nonzero": n,
        "n_positive_diffs": n_pos,
        "p_value_gamma1": p_unbiased,
        "grid": rows,
        "critical_gamma_approx": critical_gamma,
        "interpretation": (
            f"El resultado permanece significativo hasta Γ≈{critical_gamma} "
            f"(aprox.); un sesgo oculto de esa magnitud podría anular la inferencia."
            if critical_gamma
            else "En la grilla evaluada el p-value acotado sigue <0.05, o el efecto no era significativo en Γ=1."
        ),
        "limitations": [
            "Aplica a outcomes pareados (p.ej. post-matching).",
            "Cotas basadas en test de signos; ignora magnitud de las diferencias.",
            "Γ crítico es aproximado sobre una grilla discreta.",
        ],
    }


def sensitivity_report(
    effect: float | None = None,
    se: float | None = None,
    sd_outcome: float | None = None,
    odds_ratio: float | None = None,
    matched_treated: np.ndarray | None = None,
    matched_control: np.ndarray | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "disclaimer": (
            "Sensibilidad exploratoria. No demuestra ni refuta causalidad; "
            "cuantifica robustez ante confusión no medida."
        )
    }
    if odds_ratio is not None:
        out["evalue_or"] = evalue_from_risk_ratio(odds_ratio, rare_outcome=True)
    if effect is not None and sd_outcome is not None:
        out["evalue_continuous"] = evalue_from_continuous_effect(effect, se, sd_outcome)
    if matched_treated is not None and matched_control is not None:
        out["rosenbaum"] = rosenbaum_bounds_signed(matched_treated, matched_control)
    return out
