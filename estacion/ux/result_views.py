"""Helpers para mostrar resultados sin JSON crudo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from estacion.ux.presentation import format_number, format_p_value


def dict_to_table(d: dict[str, Any], exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    rows = []
    for k, v in d.items():
        if k in exclude or isinstance(v, (dict, list, pd.DataFrame)):
            continue
        rows.append({"campo": k, "valor": v})
    return pd.DataFrame(rows)


def show_result_block(
    conclusion: str,
    *,
    table: pd.DataFrame | None = None,
    raw: Any = None,
    ok: bool = True,
) -> None:
    if ok:
        st.success(conclusion) if "no " not in conclusion.lower()[:20] else st.info(conclusion)
    else:
        st.warning(conclusion)
    if table is not None and not table.empty:
        st.dataframe(table, width="stretch", hide_index=True)
    if raw is not None:
        with st.expander("Ver datos crudos"):
            if isinstance(raw, dict):
                st.json(raw)
            else:
                st.write(raw)


def humanize_missingness(miss: dict[str, Any]) -> None:
    summary = miss.get("summary")
    if isinstance(summary, pd.DataFrame):
        st.dataframe(summary, width="stretch")
    elif isinstance(summary, list):
        st.dataframe(pd.DataFrame(summary), width="stretch")

    little = miss.get("little_mcar") or {}
    if little.get("reject_mcar") or (little.get("ok") and little.get("p_value") is not None and little.get("p_value") < 0.05):
        st.warning(
            little.get("interpretation")
            or "Hay indicios de que los faltantes no son completamente al azar (contra MCAR)."
        )
    elif little:
        st.info(
            little.get("interpretation")
            or "No hay evidencia fuerte contra MCAR en esta muestra."
        )
        with st.expander("Detalle Little MCAR"):
            st.dataframe(dict_to_table(little), width="stretch", hide_index=True)

    mar = miss.get("mar_screen") or {}
    tests = mar.get("tests") or []
    n_sig = mar.get("n_significant") or sum(1 for t in tests if t.get("significant_fdr"))
    if tests:
        st.write(
            f"**Pantalla MAR:** {n_sig} asociaciones faltante↔observado "
            "tras corregir por muchas pruebas."
        )
        st.dataframe(pd.DataFrame(tests).head(30), width="stretch")


def humanize_causal_adjusted(adj: dict[str, Any]) -> None:
    if not adj.get("ok"):
        reason = adj.get("reason") or "No se pudo estimar la asociación ajustada."
        if "categórico" in str(reason).lower() or "categorico" in str(reason).lower():
            st.error(
                "Para este análisis el resultado tiene que ser un número o sí/no. "
                "Elegí otra columna (por ejemplo revenue)."
            )
        else:
            st.warning(reason)
        return
    coef = adj.get("coef") or adj.get("ate") or adj.get("estimate")
    p = adj.get("p_value")
    st.success(
        f"Asociación ajustada ≈ {format_number(coef)} "
        f"(p = {format_p_value(p)}). No demuestra causalidad."
    )
    st.dataframe(
        dict_to_table(adj, exclude={"summary_text", "model", "residuals"}),
        width="stretch",
        hide_index=True,
    )


def describe_segments(
    df: pd.DataFrame,
    labels,
    index,
    numeric_cols: list[str],
    max_features: int = 3,
) -> list[str]:
    """Frases tipo 'Grupo 0: alto revenue, bajo discount'."""
    if labels is None or index is None:
        return []
    sub = df.loc[index, numeric_cols].copy()
    sub["_seg"] = list(labels)
    phrases = []
    overall = sub[numeric_cols].mean()
    for seg, g in sub.groupby("_seg"):
        if str(seg) == "-1":
            phrases.append(f"Ruido/outliers: {len(g)} filas sin grupo claro.")
            continue
        deltas = ((g[numeric_cols].mean() - overall) / (overall.abs() + 1e-9)).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        bits = []
        for col in deltas.index[:max_features]:
            direction = "alto" if deltas[col] > 0.05 else ("bajo" if deltas[col] < -0.05 else "típico")
            if direction != "típico":
                bits.append(f"{direction} {col}")
        desc = ", ".join(bits) if bits else "perfil similar al promedio"
        phrases.append(f"Grupo {seg} ({len(g)} filas): {desc}.")
    return phrases
