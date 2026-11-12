"""
Análisis causal exploratorio.

IMPORTANTE: Nada aquí demuestra causalidad. Se exploran asociaciones
ajustadas y se explicitan supuestos faltantes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import networkx as nx


def build_suggested_dag(
    treatment: str,
    outcome: str,
    confounders: list[str],
    mediators: list[str] | None = None,
) -> dict[str, Any]:
    """DAG sugerido (hipótesis), no inferido de los datos."""
    mediators = mediators or []
    g = nx.DiGraph()
    nodes = [treatment, outcome] + confounders + mediators
    g.add_nodes_from(nodes)
    for c in confounders:
        g.add_edge(c, treatment)
        g.add_edge(c, outcome)
    g.add_edge(treatment, outcome)
    for m in mediators:
        g.add_edge(treatment, m)
        g.add_edge(m, outcome)
    return {
        "nodes": list(g.nodes),
        "edges": list(g.edges),
        "note": (
            "DAG hipotético construido a partir de roles asignados por el usuario/heurística. "
            "No está identificado automáticamente a partir de los datos."
        ),
        "assumptions_required": [
            "No confusores no medidos (ignorabilidad condicional).",
            "Positividad: overlap en propensión al tratamiento.",
            "Consistencia / SUTVA.",
            "Especificación correcta del DAG (sin flechas omitidas relevantes).",
        ],
    }


def adjusted_association(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any]:
    """Regresión del outcome ~ treatment + confounders (asociación ajustada)."""
    cols = [treatment, outcome] + confounders
    data = df[cols].dropna().copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(data[c]):
            if data[c].nunique() <= 10:
                dummies = pd.get_dummies(data[c], prefix=c, drop_first=True, dtype=float)
                data = pd.concat([data.drop(columns=[c]), dummies], axis=1)
                if c == treatment:
                    # primer dummy como tratamiento
                    tcols = [x for x in dummies.columns]
                    if not tcols:
                        return {"ok": False, "reason": "Tratamiento no codificable"}
                    treatment_use = tcols[0]
                if c == outcome:
                    return {"ok": False, "reason": "Outcome categórico no soportado aquí"}
            else:
                data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna()
    if treatment not in data.columns:
        # encoding case
        t_candidates = [c for c in data.columns if c.startswith(treatment + "_")]
        if not t_candidates:
            return {"ok": False, "reason": "No se pudo preparar treatment"}
        treatment_use = t_candidates[0]
    else:
        treatment_use = treatment

    y = pd.to_numeric(data[outcome], errors="coerce")
    X = data.drop(columns=[outcome])
    X = X.select_dtypes(include=[np.number])
    if treatment_use not in X.columns or len(X) < 40:
        return {"ok": False, "reason": "Datos insuficientes tras preparación"}

    Xc = sm.add_constant(X)
    # Binario vs continuo
    if y.nunique() == 2:
        model = sm.Logit((y == y.max()).astype(int), Xc).fit(disp=False, maxiter=200)
        effect = {
            "coef": float(model.params[treatment_use]),
            "odds_ratio": float(np.exp(model.params[treatment_use])),
            "pvalue": float(model.pvalues[treatment_use]),
            "model": "logit_adjusted",
        }
    else:
        model = sm.OLS(y, Xc).fit()
        effect = {
            "coef": float(model.params[treatment_use]),
            "pvalue": float(model.pvalues[treatment_use]),
            "ci": [float(x) for x in model.conf_int().loc[treatment_use].tolist()],
            "model": "ols_adjusted",
            "r_squared": float(model.rsquared),
        }

    crude = sm.OLS(y, sm.add_constant(X[[treatment_use]])).fit() if y.nunique() > 2 else None
    return {
        "ok": True,
        "treatment": treatment_use,
        "outcome": outcome,
        "confounders": confounders,
        "n": int(len(X)),
        "adjusted": effect,
        "crude_coef": float(crude.params[treatment_use]) if crude is not None else None,
        "interpretation_guardrail": (
            "Esto es una asociación ajustada, NO un efecto causal demostrado. "
            "Solo sería causal bajo supuestos fuertes (ver DAG)."
        ),
        "alternative_explanations": [
            "Confusores no medidos residuales.",
            "Causalidad inversa (outcome afecta treatment).",
            "Error de medición.",
            "Selección muestral / missingness informativo.",
            "Sobreajuste de mediadores (si se condicionó en mediadores).",
        ],
    }


def propensity_score_matching(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    caliper: float = 0.05,
) -> dict[str, Any]:
    """Matching 1:1 por propensity score (nearest neighbor + caliper)."""
    cols = [treatment, outcome] + confounders
    data = df[cols].dropna().copy()
    # treatment binario
    t = data[treatment]
    if t.nunique() != 2:
        return {"ok": False, "reason": "El treatment debe ser binario para PSM"}
    levels = sorted(t.unique())
    t_bin = (t == levels[-1]).astype(int).to_numpy()
    X = data[confounders].copy()
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]):
            dummies = pd.get_dummies(X[c], prefix=c, drop_first=True, dtype=float)
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = pd.to_numeric(data[outcome], errors="coerce").to_numpy()
    if len(X) < 50:
        return {"ok": False, "reason": "n insuficiente"}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(Xs, t_bin)
    ps = clf.predict_proba(Xs)[:, 1]

    treated_idx = np.where(t_bin == 1)[0]
    control_idx = np.where(t_bin == 0)[0]
    if len(treated_idx) < 10 or len(control_idx) < 10:
        return {"ok": False, "reason": "Pocos tratados o controles"}

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ps[control_idx].reshape(-1, 1))
    dist, ind = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    pairs = []
    used_controls = set()
    for ti, d, ci_local in zip(treated_idx, dist.ravel(), ind.ravel()):
        if d > caliper:
            continue
        ci = control_idx[ci_local]
        if ci in used_controls:
            continue
        used_controls.add(ci)
        pairs.append((ti, ci, float(d)))

    if len(pairs) < 10:
        return {
            "ok": False,
            "reason": "Pocos pares dentro del caliper; posible falta de overlap",
            "n_pairs": len(pairs),
        }

    y_t = np.array([y[i] for i, _, _ in pairs])
    y_c = np.array([y[j] for _, j, _ in pairs])
    ate = float(y_t.mean() - y_c.mean())
    # SE naive pareado
    diffs = y_t - y_c
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    # Balance: diferencia estandarizada media en PS y covariables
    ps_t = np.array([ps[i] for i, _, _ in pairs])
    ps_c = np.array([ps[j] for _, j, _ in pairs])
    smd_ps = float((ps_t.mean() - ps_c.mean()) / np.sqrt((ps_t.var() + ps_c.var()) / 2 + 1e-9))

    cov_balance = []
    Xv = X.to_numpy(dtype=float)
    for ci, cname in enumerate(X.columns):
        before_t = Xv[t_bin == 1, ci]
        before_c = Xv[t_bin == 0, ci]
        after_t = np.array([Xv[i, ci] for i, _, _ in pairs])
        after_c = np.array([Xv[j, ci] for _, j, _ in pairs])

        def _smd(a, b):
            denom = np.sqrt((a.var() + b.var()) / 2 + 1e-9)
            return float((a.mean() - b.mean()) / denom)

        cov_balance.append(
            {
                "covariate": cname,
                "smd_before": _smd(before_t, before_c),
                "smd_after": _smd(after_t, after_c),
            }
        )

    from estacion.causal.sensitivity import sensitivity_report

    sens = sensitivity_report(
        effect=ate,
        se=se,
        sd_outcome=float(np.std(np.concatenate([y_t, y_c]), ddof=1)),
        matched_treated=y_t,
        matched_control=y_c,
    )

    return {
        "ok": True,
        "method": "propensity_score_matching_1to1",
        "n_pairs": len(pairs),
        "caliper": caliper,
        "ate_estimate": ate,
        "se_naive": se,
        "ci95_naive": [ate - 1.96 * se, ate + 1.96 * se],
        "ps_smd_after": smd_ps,
        "covariate_balance": cov_balance,
        "balance_ok": all(abs(r["smd_after"]) < 0.1 for r in cov_balance) if cov_balance else False,
        "sensitivity": sens,
        "matched_outcomes": {"treated": y_t.tolist()[:500], "control": y_c.tolist()[:500]},
        "treatment_level_positive": levels[-1],
        "warning": (
            "Estimación exploratoria. IC naive; no robusto a dependencia del matching. "
            "NO interpretar como causalidad demostrada."
        ),
        "assumptions": [
            "Ignorabilidad condicional dados los confounders incluidos.",
            "Overlap / positividad.",
            "Sin mediadores incorrectamente tratados como confusores.",
            "Outcome bien medido; treatment bien definido.",
        ],
        "missing_for_causal_claim": [
            "Justificación teórica del DAG.",
            "Confusores no medidos (ver e-value / Rosenbaum en sensitivity).",
            "Balance post-matching en todas las covariables relevantes.",
            "Diseño previo (no fishing sobre múltiples treatments/outcomes).",
        ],
    }


def difference_in_differences(
    df: pd.DataFrame,
    outcome: str,
    treat_group_col: str,
    time_col: str,
    post_period_value: Any,
) -> dict[str, Any]:
    from estacion.causal.did import difference_in_differences_2x2

    return difference_in_differences_2x2(
        df, outcome, treat_group_col, time_col, post_period_value
    )


def exploratory_causal_screen(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str] | None = None,
) -> dict[str, Any]:
    from estacion.causal.sensitivity import evalue_from_risk_ratio, sensitivity_report

    confounders = confounders or []
    dag = build_suggested_dag(treatment, outcome, confounders)
    adj = adjusted_association(df, treatment, outcome, confounders)
    psm = None
    t = df[treatment]
    if t.nunique(dropna=True) == 2 and confounders:
        psm = propensity_score_matching(df, treatment, outcome, confounders)

    sensitivity = None
    if adj.get("ok"):
        adjusted = adj.get("adjusted") or {}
        if adjusted.get("model") == "logit_adjusted" and "odds_ratio" in adjusted:
            sensitivity = {
                "evalue_from_adjusted_or": evalue_from_risk_ratio(adjusted["odds_ratio"]),
                "note": "E-value usando OR ajustado como aproximación de RR (mejor si outcome raro).",
            }
        elif adjusted.get("model") == "ols_adjusted" and "coef" in adjusted:
            # sd outcome
            y = pd.to_numeric(df[outcome], errors="coerce").dropna()
            sensitivity = sensitivity_report(
                effect=adjusted["coef"],
                sd_outcome=float(y.std(ddof=1)) if len(y) else None,
            )

    return {
        "dag": dag,
        "adjusted_association": adj,
        "propensity_matching": psm,
        "sensitivity_adjusted": sensitivity,
        "disclaimer": (
            "Análisis causal EXPLORATORIO. Ningún resultado debe presentarse "
            "como efecto causal demostrado."
        ),
    }
