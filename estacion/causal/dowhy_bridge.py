"""
Wrapper opcional a DoWhy.

Si DoWhy no está instalado, degrada con mensaje claro. Nunca es dependencia dura.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def dowhy_available() -> bool:
    try:
        import dowhy  # noqa: F401

        return True
    except Exception:
        return False


def estimate_ate_dowhy(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    method: str = "backdoor.linear_regression",
) -> dict[str, Any]:
    if not dowhy_available():
        return {
            "ok": False,
            "reason": "DoWhy no instalado. Opcional: pip install dowhy",
            "optional": True,
        }
    try:
        from dowhy import CausalModel
    except Exception as exc:
        return {"ok": False, "reason": str(exc), "optional": True}

    common_causes = [c for c in confounders if c in df.columns and c not in {treatment, outcome}]
    data = df[[treatment, outcome] + common_causes].dropna().copy()
    if len(data) < 50:
        return {"ok": False, "reason": "n insuficiente tras dropna"}

    # tipar treatment numérico si es binario-like
    if not pd.api.types.is_numeric_dtype(data[treatment]):
        levels = sorted(data[treatment].unique(), key=lambda x: str(x))
        if len(levels) != 2:
            return {"ok": False, "reason": "DoWhy bridge espera treatment binario o numérico"}
        data[treatment] = (data[treatment] == levels[-1]).astype(int)
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()

    graph = f"""digraph {{
        {treatment} -> {outcome};
        {' '.join(f'{c} -> {treatment}; {c} -> {outcome};' for c in common_causes)}
    }}"""
    try:
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name=method,
        )
        value = float(estimate.value)
        return {
            "ok": True,
            "method": method,
            "ate_estimate": value,
            "n": int(len(data)),
            "common_causes": common_causes,
            "disclaimer": (
                "Estimación DoWhy bajo el DAG declarado. No demuestra causalidad; "
                "depende de supuestos de identificación."
            ),
            "limitations": [
                "Requiere que el DAG sea correcto.",
                "Método por defecto es regresión backdoor lineal.",
            ],
        }
    except Exception as exc:
        return {"ok": False, "reason": f"DoWhy falló: {exc}", "optional": True}
