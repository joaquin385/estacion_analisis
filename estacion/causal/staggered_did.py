"""
DiD con adopción escalonada — ATT por cohorte×tiempo (estilo Callaway–Sant'Anna simplificado).

Comparador: never-treated si existe; si no, not-yet-treated.
Incluye placebos pre-tratamiento para chequear anticipación/paralelismo.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _safe_lt(a, b) -> bool:
    try:
        return a < b
    except TypeError:
        return str(a) < str(b)


def _safe_le(a, b) -> bool:
    try:
        return a <= b
    except TypeError:
        return str(a) <= str(b)


def _safe_gt(a, b) -> bool:
    try:
        return a > b
    except TypeError:
        return str(a) > str(b)


def _event_time(t, g) -> int:
    try:
        if isinstance(t, (int, np.integer)) and isinstance(g, (int, np.integer)):
            return int(t - g)
        if isinstance(t, (float, np.floating)) and isinstance(g, (float, np.floating, int, np.integer)):
            return int(t - g)
    except Exception:
        pass
    if str(t) == str(g):
        return 0
    return 1 if str(t) > str(g) else -1


def staggered_att(
    df: pd.DataFrame,
    outcome: str,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
    never_treated_value: Any = None,
    include_placebos: bool = True,
) -> dict[str, Any]:
    need = [outcome, unit_col, time_col, treat_time_col]
    data = df[need].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome, unit_col, time_col])

    unit_cohort = (
        data.groupby(unit_col)[treat_time_col].min().rename("cohort").reset_index()
    )
    data = data.merge(unit_cohort, on=unit_col, how="left")

    def _is_never(v) -> bool:
        if never_treated_value is None:
            return pd.isna(v)
        return v == never_treated_value or pd.isna(v)

    data["never"] = data["cohort"].map(_is_never)
    times = sorted(data[time_col].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
    # better numeric sort if possible
    try:
        times = sorted(data[time_col].dropna().unique())
    except TypeError:
        times = sorted(data[time_col].dropna().unique(), key=lambda x: str(x))

    cohorts = sorted(
        [c for c in data.loc[~data["never"], "cohort"].dropna().unique()],
        key=lambda x: str(x),
    )
    try:
        cohorts = sorted([c for c in data.loc[~data["never"], "cohort"].dropna().unique()])
    except TypeError:
        pass

    if not cohorts:
        return {"ok": False, "reason": "No hay cohortes tratadas"}
    if len(times) < 2:
        return {"ok": False, "reason": "Se necesitan ≥2 períodos"}

    has_never = bool(data["never"].any())
    control_mode = "never_treated" if has_never else "not_yet_treated"

    data["cohort_label"] = np.where(data["never"], "never", data["cohort"].astype(str))
    means = data.groupby(["cohort_label", time_col])[outcome].mean().rename("ybar").reset_index()
    sizes = data.groupby(["cohort_label", time_col])[outcome].count().rename("n").reset_index()
    panel = means.merge(sizes, on=["cohort_label", time_col])

    def ybar(label: str, t) -> float | None:
        row = panel[(panel["cohort_label"] == label) & (panel[time_col] == t)]
        return float(row["ybar"].iloc[0]) if len(row) else None

    def n_obs(label: str, t) -> int:
        row = panel[(panel["cohort_label"] == label) & (panel[time_col] == t)]
        return int(row["n"].iloc[0]) if len(row) else 0

    def control_means(t, g_pre, t_eval):
        if control_mode == "never_treated":
            return ybar("never", t_eval), ybar("never", g_pre), n_obs("never", t_eval)
        # not-yet-treated at t_eval: cohort > t_eval
        mask_units = (~data["never"]) & data["cohort"].map(lambda c: _safe_gt(c, t_eval))
        nyt = data.loc[mask_units]
        if nyt.empty:
            return None, None, 0
        yt = nyt.loc[nyt[time_col] == t_eval, outcome]
        yp = nyt.loc[nyt[time_col] == g_pre, outcome]
        if yt.empty or yp.empty:
            return None, None, 0
        return float(yt.mean()), float(yp.mean()), int(len(yt))

    att_rows = []
    for g in cohorts:
        g_label = str(g)
        pre_times = [t for t in times if _safe_lt(t, g)]
        if len(pre_times) < 1:
            continue
        # base canónico: último pre
        g_pre_default = pre_times[-1]

        target_times = list(times)
        for t in target_times:
            is_post = _safe_le(g, t)
            if not is_post and not include_placebos:
                continue
            if not is_post:
                # placebo: base = período anterior a t dentro de pre
                earlier = [u for u in pre_times if _safe_lt(u, t)]
                if not earlier:
                    continue
                g_pre = earlier[-1]
            else:
                g_pre = g_pre_default

            y_gt = ybar(g_label, t)
            y_gpre = ybar(g_label, g_pre)
            if y_gt is None or y_gpre is None:
                continue
            y_ct, y_cpre, ctrl_n = control_means(t, g_pre, t)
            if y_ct is None or y_cpre is None:
                continue
            att = (y_gt - y_gpre) - (y_ct - y_cpre)
            att_rows.append(
                {
                    "cohort": g,
                    "time": t,
                    "base_pre": g_pre,
                    "att": float(att),
                    "placebo": (not is_post),
                    "n_treated_cells": min(n_obs(g_label, t), n_obs(g_label, g_pre)),
                    "n_control_cells": ctrl_n,
                    "event_time": _event_time(t, g),
                }
            )

    if not att_rows:
        return {
            "ok": False,
            "reason": "No se pudieron estimar ATT(g,t); revisá panel y never-treated",
            "control_mode": control_mode,
        }

    att_df = pd.DataFrame(att_rows)
    post_df = att_df[~att_df["placebo"]].copy()
    overall = float(post_df["att"].mean()) if len(post_df) else np.nan

    # Peso por tamaño de celda tratada (aprox. agregación CS por tamaño de cohorte×tiempo)
    if len(post_df):
        w = post_df["n_treated_cells"].clip(lower=1).to_numpy(dtype=float)
        w = w / w.sum()
        overall_weighted = float(np.sum(post_df["att"].to_numpy(dtype=float) * w))
        post_df["weight_share"] = w
    else:
        overall_weighted = np.nan

    by_event = (
        att_df.groupby(["event_time", "placebo"])["att"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # event-time weighted (solo post)
    by_event_w = []
    if len(post_df):
        for et, gdf in post_df.groupby("event_time"):
            ww = gdf["n_treated_cells"].clip(lower=1).to_numpy(dtype=float)
            ww = ww / ww.sum()
            by_event_w.append(
                {
                    "event_time": int(et) if not isinstance(et, (float, np.floating)) or float(et).is_integer() else et,
                    "att_weighted": float(np.sum(gdf["att"].to_numpy(dtype=float) * ww)),
                    "att_simple": float(gdf["att"].mean()),
                    "n_cells": int(len(gdf)),
                    "n_treated_total": int(gdf["n_treated_cells"].sum()),
                }
            )

    pre_events = att_df[att_df["placebo"]]["att"]
    pretrend = None
    if len(pre_events) >= 2:
        tstat, p = stats.ttest_1samp(pre_events.to_numpy(), 0.0)
        pretrend = {
            "mean_pre_att": float(pre_events.mean()),
            "t_stat": float(tstat),
            "p_value": float(p),
            "reject_zero_alpha05": bool(p < 0.05),
            "n_placebos": int(len(pre_events)),
            "note": "Placebos ATT con event_time<0; si ≠0, sospechar violación de paralelismo/anticipación.",
        }

    # Outcome-regression DR simplificado a nivel unidad (una sola agregación post)
    dr = _outcome_regression_att(data, outcome, unit_col, time_col, control_mode)

    return {
        "ok": True,
        "method": "staggered_att_weighted_cs_style",
        "control_mode": control_mode,
        "n_cohorts": len(cohorts),
        "n_att_cells": int(len(post_df)),
        "n_placebo_cells": int(att_df["placebo"].sum()),
        "overall_att_simple_avg": overall,
        "overall_att_weighted": overall_weighted,
        "att_by_cohort_time": att_df.to_dict(orient="records"),
        "att_by_event_time": by_event.to_dict(orient="records"),
        "att_by_event_time_weighted": by_event_w,
        "doubly_robust_lite": dr,
        "pretrend_check": pretrend,
        "assumptions": [
            "Tendencias paralelas condicionales al grupo de adopción.",
            "No anticipación.",
            "Overlap / soporte común de cohortes.",
            "Sin efectos spillover entre unidades.",
        ],
        "limitations": [
            "Pesos por n tratado en cada celda (g,t); no son los pesos exactos de Callaway–Sant'Anna.",
            "doubly_robust_lite es una regresión de outcome simple (no AIPW formal completo).",
            "Sin errores estándar clusterizados formales.",
            "Resultado exploratorio; no demuestra causalidad.",
        ],
        "disclaimer": "No demuestra causalidad. Resultado exploratorio.",
    }


def _outcome_regression_att(
    data: pd.DataFrame,
    outcome: str,
    unit_col: str,
    time_col: str,
    control_mode: str,
) -> dict[str, Any]:
    """
    Estimador lite: para cada unidad tratada, ΔY post-pre menos ΔY predicho
    con regresión ajustada en controles (never / not-yet) sobre time FE.
    """
    d = data.copy()
    # pre = primer período, post = último
    try:
        times = sorted(d[time_col].dropna().unique())
    except TypeError:
        times = sorted(d[time_col].dropna().unique(), key=lambda x: str(x))
    if len(times) < 2:
        return {"ok": False, "reason": "pocos períodos"}
    t0, t1 = times[0], times[-1]

    wide = d.pivot_table(index=unit_col, columns=time_col, values=outcome, aggfunc="mean")
    if t0 not in wide.columns or t1 not in wide.columns:
        return {"ok": False, "reason": "no se pudo pivotear pre/post"}
    meta = d.groupby(unit_col).agg(cohort=("cohort", "first"), never=("never", "first"))
    panel = wide.join(meta)
    panel = panel.dropna(subset=[t0, t1])
    panel["dy"] = panel[t1] - panel[t0]

    controls = panel[panel["never"]] if control_mode == "never_treated" else panel[panel["never"] | panel["cohort"].isna()]
    treated = panel[~panel["never"] & panel["cohort"].notna()]
    if len(controls) < 10 or len(treated) < 5:
        return {"ok": False, "reason": "pocos tratados/controles para DR-lite"}

    # En controles, dy ~ const (media); residual treated - mean_control
    mu_c = float(controls["dy"].mean())
    att_or = float((treated["dy"] - mu_c).mean())
    # bootstrap simple de unidades
    rng = np.random.default_rng(0)
    boots = []
    tc = treated["dy"].to_numpy()
    cc = controls["dy"].to_numpy()
    for _ in range(300):
        tb = rng.choice(tc, size=len(tc), replace=True)
        cb = rng.choice(cc, size=len(cc), replace=True)
        boots.append(tb.mean() - cb.mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "ok": True,
        "method": "outcome_reg_delta_y_control_mean",
        "att": att_or,
        "ci95_bootstrap": [float(lo), float(hi)],
        "n_treated_units": int(len(treated)),
        "n_control_units": int(len(controls)),
        "pre_period": t0,
        "post_period": t1,
        "note": "Comparación ΔY (último−primero) tratados vs controles; agregación panel simple.",
    }


def prepare_treat_time_from_binary(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    treat_col: str,
) -> pd.Series:
    """Deriva primer período con treat>0; NaN si never-treated."""
    data = df[[unit_col, time_col, treat_col]].copy()
    on = data[data[treat_col].astype(float) > 0]
    first = on.groupby(unit_col)[time_col].min()
    return df[unit_col].map(first)
