"""
AIPW / doubly-robust para adopción escalonada (versión formal-lite).

Para un horizonte post fijo (o último período), estima ATT entre unidades
con treat_time <= t* versus never-treated, usando:
  - modelo de outcome en controles
  - propensity de llegar a tratarse (vs never)
  - fórmula AIPW clásica

No sustituye el estimador completo Callaway–Sant'Anna multi (g,t), pero
sí aporta un DR formal con IC bootstrap sobre un contraste agregado.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


def aipw_staggered_att(
    df: pd.DataFrame,
    outcome: str,
    unit_col: str,
    time_col: str,
    treat_time_col: str,
    post_time: Any | None = None,
    pre_time: Any | None = None,
    covariate_cols: list[str] | None = None,
    n_boot: int = 400,
    random_state: int = 42,
) -> dict[str, Any]:
    data = df.copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[outcome, unit_col, time_col])

    times = sorted(data[time_col].dropna().unique())
    try:
        times = sorted(data[time_col].dropna().unique())
    except TypeError:
        times = sorted(data[time_col].dropna().unique(), key=lambda x: str(x))
    if len(times) < 2:
        return {"ok": False, "reason": "Se necesitan ≥2 períodos"}

    pre_time = pre_time if pre_time is not None else times[0]
    post_time = post_time if post_time is not None else times[-1]

    # cohort por unidad
    cohort = data.groupby(unit_col)[treat_time_col].min().rename("cohort")
    # wide outcome
    wide = data.pivot_table(index=unit_col, columns=time_col, values=outcome, aggfunc="mean")
    if pre_time not in wide.columns or post_time not in wide.columns:
        return {"ok": False, "reason": "pre_time/post_time no presentes en el panel"}

    panel = pd.DataFrame(
        {
            "y_pre": wide[pre_time],
            "y_post": wide[post_time],
        }
    ).join(cohort)
    panel["dy"] = panel["y_post"] - panel["y_pre"]
    panel["never"] = panel["cohort"].isna()
    # tratado si cohort <= post_time
    def _le(a, b):
        try:
            return a <= b
        except TypeError:
            return str(a) <= str(b)

    panel["treated"] = (~panel["never"]) & panel["cohort"].map(
        lambda c: False if pd.isna(c) else _le(c, post_time)
    )
    panel = panel.dropna(subset=["dy"])

    # covariables a nivel unidad (promedio temporal o primer valor)
    covars = covariate_cols or []
    X_cols = []
    for c in covars:
        if c not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[c]):
            panel[c] = data.groupby(unit_col)[c].mean()
            X_cols.append(c)
        else:
            # one-hot moda
            mode = data.groupby(unit_col)[c].agg(
                lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else "NA"
            )
            dummies = pd.get_dummies(mode, prefix=c, drop_first=True, dtype=float)
            panel = panel.join(dummies)
            X_cols.extend(list(dummies.columns))

    # baseline: también usar y_pre como covariable del propensity/outcome
    X_cols = list(dict.fromkeys(X_cols + ["y_pre"]))
    for c in X_cols:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    panel = panel.dropna(subset=X_cols + ["dy", "treated"])

    treated = panel["treated"].astype(int).to_numpy()
    if treated.sum() < 8 or (1 - treated).sum() < 8:
        return {"ok": False, "reason": "Pocos tratados o controles para AIPW"}

    X = panel[X_cols].to_numpy(dtype=float)
    y = panel["dy"].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Propensity e(X) = P(treated|X)
    clf = LogisticRegression(max_iter=500, random_state=random_state)
    clf.fit(Xs, treated)
    e = np.clip(clf.predict_proba(Xs)[:, 1], 0.05, 0.95)

    # Outcome models μ1, μ0
    mu1 = LinearRegression().fit(Xs[treated == 1], y[treated == 1]).predict(Xs)
    mu0 = LinearRegression().fit(Xs[treated == 0], y[treated == 0]).predict(Xs)

    # AIPW ATT: E[ μ1-μ0 + T*(Y-μ1)/e - (1-T)*(Y-μ0)/(1-e) | ] among treated weighting
    # Standard AIPW for ATE:
    aipw_ate_i = (mu1 - mu0) + treated * (y - mu1) / e - (1 - treated) * (y - mu0) / (1 - e)
    # ATT AIPW (Hirano/Imbens style approx):
    # E[(T/e) * ((Y - μ0) - (1-e)/e wait)
    # Usamos fórmula ATT:
    # ψ = (T - e)/((1-e)*p_t) * (Y - μ0)  ... alternate
    # Implementación práctica estable:
    # ATT_hat = mean_{T=1}[ Y - μ0(X) ]  + corrección IPW residual en controles proyectada
    # Mejor: Cao/Zheng style
    p_t = treated.mean()
    psi_att = (
        treated * (y - mu0) / p_t
        - e * (1 - treated) * (y - mu0) / ((1 - e) * p_t)
        - treated * (mu1 - mu0) * 0  # keep OR component via μ0 only for ATT
    )
    # Include outcome regression for treated path:
    psi_att = treated / p_t * ((y - mu1) + (mu1 - mu0)) - (1 - treated) * e / (
        (1 - e) * p_t
    ) * (y - mu0)

    att = float(np.mean(psi_att))
    ate = float(np.mean(aipw_ate_i))

    # bootstrap units
    rng = np.random.default_rng(random_state)
    boots = []
    n = len(panel)
    idx_all = np.arange(n)
    for _ in range(n_boot):
        b = rng.choice(idx_all, size=n, replace=True)
        tb = treated[b]
        if tb.sum() < 5 or (1 - tb).sum() < 5:
            continue
        Xb, yb = Xs[b], y[b]
        try:
            clf_b = LogisticRegression(max_iter=300, random_state=0)
            clf_b.fit(Xb, tb)
            eb = np.clip(clf_b.predict_proba(Xb)[:, 1], 0.05, 0.95)
            mu1b = LinearRegression().fit(Xb[tb == 1], yb[tb == 1]).predict(Xb)
            mu0b = LinearRegression().fit(Xb[tb == 0], yb[tb == 0]).predict(Xb)
            ptb = tb.mean()
            psi = tb / ptb * ((yb - mu1b) + (mu1b - mu0b)) - (1 - tb) * eb / (
                (1 - eb) * ptb
            ) * (yb - mu0b)
            boots.append(float(np.mean(psi)))
        except Exception:
            continue
    if len(boots) >= 50:
        lo, hi = np.quantile(boots, [0.025, 0.975])
        ci = [float(lo), float(hi)]
    else:
        ci = [np.nan, np.nan]

    return {
        "ok": True,
        "method": "aipw_staggered_att_aggregated",
        "pre_time": pre_time,
        "post_time": post_time,
        "n_units": int(n),
        "n_treated": int(treated.sum()),
        "n_control": int((1 - treated).sum()),
        "covariates": X_cols,
        "att_aipw": att,
        "ate_aipw": ate,
        "ci95_att_bootstrap": ci,
        "propensity_overlap": {
            "e_min": float(e.min()),
            "e_max": float(e.max()),
            "e_mean_treated": float(e[treated == 1].mean()),
            "e_mean_control": float(e[treated == 0].mean()),
        },
        "assumptions": [
            "Ignorabilidad condicional de la adopción dados X (y y_pre).",
            "Positividad / overlap de propensity.",
            "Consistencia; sin interferencia entre unidades.",
            "Contraste agregado pre→post; no desagrega todos los ATT(g,t).",
        ],
        "limitations": [
            "AIPW sobre ΔY agregado, no el estimador CS completo celda a celda.",
            "Modelos lineales/logit; malespecificación posible.",
            "IC bootstrap de unidades; no cluster adicional.",
        ],
        "disclaimer": "No demuestra causalidad. Estimación exploratoria doubly-robust.",
    }
