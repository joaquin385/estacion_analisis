"""Difference-in-differences multi-período con test de pre-trends."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def difference_in_differences_2x2(
    df: pd.DataFrame,
    outcome: str,
    treat_group_col: str,
    time_col: str,
    post_period_value: Any,
) -> dict[str, Any]:
    data = df[[outcome, treat_group_col, time_col]].dropna().copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    levels = sorted(data[treat_group_col].unique())
    if len(levels) != 2:
        return {"ok": False, "reason": "treat_group_col debe tener 2 niveles"}
    treat = (data[treat_group_col] == levels[-1]).astype(int)
    post = (data[time_col] == post_period_value).astype(int)
    if post.sum() == 0 or post.sum() == len(post):
        return {"ok": False, "reason": "post_period_value no separa períodos"}

    X = pd.DataFrame({"treat": treat, "post": post, "did": treat * post})
    Xc = sm.add_constant(X)
    model = sm.OLS(data[outcome], Xc).fit()
    return {
        "ok": True,
        "method": "did_2x2",
        "did_coef": float(model.params["did"]),
        "pvalue": float(model.pvalues["did"]),
        "ci": [float(x) for x in model.conf_int().loc["did"].tolist()],
        "n": int(model.nobs),
        "assumptions": [
            "Tendencias paralelas (no verificable solo con 2 períodos).",
            "No anticipación / composición estable de grupos.",
            "Sin shocks diferenciales concurrentes.",
        ],
        "warning": "DiD exploratorio; tendencias paralelas no validadas aquí.",
    }


def parallel_trends_test(
    df: pd.DataFrame,
    outcome: str,
    treat_group_col: str,
    time_col: str,
    treated_level: Any,
    post_periods: list[Any],
) -> dict[str, Any]:
    """
    En períodos PRE, estima outcome ~ treat + C(time) + treat×C(time).
    Si las interacciones pre son conjuntamente ~0, no se rechazan tendencias paralelas.
    """
    data = df[[outcome, treat_group_col, time_col]].dropna().copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    data["_treat"] = (data[treat_group_col] == treated_level).astype(int)
    pre = data[~data[time_col].isin(post_periods)].copy()
    times = sorted(pre[time_col].unique(), key=lambda x: str(x))
    if len(times) < 3:
        return {
            "ok": False,
            "reason": "Se necesitan ≥3 períodos pre para test de pre-trends",
            "n_pre_periods": len(times),
        }
    # baseline = primer período
    base = times[0]
    dummies = []
    for t in times[1:]:
        dummies.append(f"t_{t}")
        pre[f"t_{t}"] = (pre[time_col] == t).astype(int)
        pre[f"treat_x_{t}"] = pre["_treat"] * pre[f"t_{t}"]

    X = pre[["_treat"] + [f"t_{t}" for t in times[1:]] + [f"treat_x_{t}" for t in times[1:]]].astype(float)
    Xc = sm.add_constant(X)
    model = sm.OLS(pre[outcome], Xc).fit()
    interact = [f"treat_x_{t}" for t in times[1:]]
    # Wald conjunto
    try:
        wald = model.wald_test(",".join([f"{c}=0" for c in interact]), scalar=True)
        wald_p = float(wald.pvalue)
        wald_stat = float(wald.statistic)
    except Exception:
        # fallback: media de p individuales
        ps = [float(model.pvalues[c]) for c in interact]
        wald_p = float(np.min(ps))
        wald_stat = np.nan

    coefs = {c: float(model.params[c]) for c in interact}
    return {
        "ok": True,
        "method": "event_study_pretrends",
        "baseline_period": base,
        "pre_periods": [str(t) for t in times],
        "interaction_coefs": coefs,
        "wald_stat": wald_stat,
        "wald_pvalue": wald_p,
        "parallel_trends_rejected_alpha05": bool(wald_p < 0.05),
        "interpretation": (
            "Se rechazan tendencias paralelas en pre (p<0.05): DiD post es dudoso."
            if wald_p < 0.05
            else "No se rechazan pre-trends a α=0.05; no prueba el supuesto, solo no lo contradice."
        ),
        "limitations": [
            "Requiere múltiples períodos pre-tratamiento.",
            "No detecta violaciones solo en el post.",
            "Sensibles a especificación y composición de paneles.",
        ],
    }


def difference_in_differences_multipperiod(
    df: pd.DataFrame,
    outcome: str,
    treat_group_col: str,
    time_col: str,
    treated_level: Any,
    post_periods: list[Any],
) -> dict[str, Any]:
    """
    DiD con dummies temporales + treat×post (post agregado) y event-study opcional.
    """
    data = df[[outcome, treat_group_col, time_col]].dropna().copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    data["_treat"] = (data[treat_group_col] == treated_level).astype(int)
    data["_post"] = data[time_col].isin(post_periods).astype(int)
    data["_did"] = data["_treat"] * data["_post"]

    times = sorted(data[time_col].unique(), key=lambda x: str(x))
    if len(times) < 2:
        return {"ok": False, "reason": "Se necesitan ≥2 períodos"}

    # TWFE simple: outcome ~ treat + C(time) + did
    base_t = times[0]
    for t in times[1:]:
        data[f"t_{t}"] = (data[time_col] == t).astype(int)
    X = data[["_treat", "_did"] + [f"t_{t}" for t in times[1:]]].astype(float)
    Xc = sm.add_constant(X)
    model = sm.OLS(data[outcome], Xc).fit()

    # Event study: treat × cada período (omitir un pre)
    pre_times = [t for t in times if t not in post_periods]
    omit = pre_times[-1] if pre_times else times[0]
    es_cols = []
    for t in times:
        if t == omit:
            continue
        col = f"es_{t}"
        data[col] = data["_treat"] * (data[time_col] == t).astype(int)
        es_cols.append(col)
    es = None
    if es_cols:
        Xe = data[["_treat"] + [f"t_{t}" for t in times[1:] if t != times[0]] + es_cols]
        # time dummies already; simplify
        time_d = [f"t_{t}" for t in times[1:]]
        Xe = data[["_treat"] + time_d + es_cols].astype(float)
        Xe = sm.add_constant(Xe)
        try:
            es_model = sm.OLS(data[outcome], Xe).fit()
            es = {
                "omitted_period": str(omit),
                "coefs": {c: float(es_model.params[c]) for c in es_cols},
                "pvalues": {c: float(es_model.pvalues[c]) for c in es_cols},
            }
        except Exception as exc:
            es = {"ok": False, "reason": str(exc)}

    pre = parallel_trends_test(
        df, outcome, treat_group_col, time_col, treated_level, post_periods
    )

    return {
        "ok": True,
        "method": "did_multipperiod_twfe",
        "did_coef": float(model.params["_did"]),
        "pvalue": float(model.pvalues["_did"]),
        "ci": [float(x) for x in model.conf_int().loc["_did"].tolist()],
        "n": int(model.nobs),
        "n_periods": len(times),
        "post_periods": [str(p) for p in post_periods],
        "treated_level": treated_level,
        "event_study": es,
        "pretrends": pre,
        "assumptions": [
            "Tendencias paralelas (evaluadas en pre si hay ≥3 períodos pre).",
            "No anticipación; composición estable.",
            "Efectos homogéneos (TWFE puede sesgar con adopción escalonada).",
        ],
        "warning": (
            "DiD exploratorio. Si pretrends rechazan paralelismo, no interpretar el DID post."
        ),
        "disclaimer": "No demuestra causalidad.",
    }
