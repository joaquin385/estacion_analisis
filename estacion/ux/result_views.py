"""Helpers para mostrar resultados sin JSON crudo."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from estacion.ux.presentation import (
    format_number,
    format_int_es,
    format_p_value,
    format_p_clause,
    format_pct,
    display_value,
    dataframe_for_display,
)


FIELD_LABELS_ES = {
    "crude_coef": "Diferencia sin ajustar",
    "interpretation_guardrail": "Aviso de interpretación",
    "coef": "Coeficiente / diferencia ajustada",
    "ate": "Efecto estimado",
    "estimate": "Estimación",
    "pvalue": "p-valor",
    "p_value": "p-valor",
    "p_fdr": "p corregido",
    "p_raw": "p sin corregir",
    "ci": "Intervalo de confianza",
    "r_squared": "R²",
    "n": "N",
    "treatment": "Tratamiento (codificado)",
    "outcome": "Resultado",
    "confounders": "Confusores",
    "model": "Modelo",
    "odds_ratio": "Odds ratio",
    "ok": "Ok",
    "n_missing": "Nº vacíos",
    "missing_pct": "% vacíos",
    "missing_pct_a": "% vacíos A",
    "missing_pct_b": "% vacíos B",
    "missing_delta": "Δ % vacíos",
    "missing": "Vacíos",
    "significant_fdr": "Significativo (corregido)",
    "method": "Método",
    "statistic": "Estadístico",
    "psi": "PSI (cambio)",
    "mean_a": "Media A",
    "mean_b": "Media B",
    "ks_2samp": "Comparación de distribuciones",
    "alternative_explanations": "Otras explicaciones",
    "adjusted": "Detalle del ajuste",
    "column": "Columna",
    "type": "Tipo",
    "tvd": "Distancia TV",
    "cohens_d": "d de Cohen",
    "practically_important": "Relevante en la práctica",
    "mean": "Media",
    "std": "Desvío",
    "se": "Error estándar",
    "ci_low": "IC inferior",
    "ci_high": "IC superior",
    "median": "Mediana",
    "skewness": "Asimetría",
    "kurtosis": "Curtosis",
    "alpha": "Alfa",
    "variable": "Variable",
    "rule": "Regla",
    "severity": "Severidad",
    "detail": "Detalle",
    "value": "Valor",
    "threshold": "Umbral",
    "n_observed": "N observado",
    "n_observed_vars": "Variables observadas",
    "missing_col": "Columna con vacíos",
    "predictor": "Predictor",
    "effect": "Efecto (diff. de medias)",
    "missing_rate_target": "% vacíos (objetivo)",
    "reason": "Motivo",
    "campo": "Campo",
    "valor": "Valor",
    "Campo": "Campo",
    "Valor": "Valor",
    "df": "Grados de libertad",
    "reject_mcar_alpha05": "Se rechaza al azar (α=0,05)",
    "n_patterns_used": "Patrones usados",
    "pattern": "Patrón",
}


METHOD_LABELS_ES = {
    "ks_2samp": "Comparación de distribuciones",
    "psi": "Índice de estabilidad poblacional (PSI)",
    "chi2": "Chi-cuadrado",
    "ols_adjusted": "Regresión ajustada",
    "logit_adjusted": "Regresión logística ajustada",
    "t-interval": "Intervalo t",
    "wilson": "Wilson",
    "welch_t": "Prueba t de Welch",
    "mannwhitney_u": "Mann–Whitney",
    "mannwhitney_missing_vs_observed": "Mann–Whitney (vacío vs observado)",
    "chi2_missing_indicator": "Chi-cuadrado (indicador de vacío)",
    "kruskal_wallis": "Kruskal–Wallis",
    "anova_oneway": "ANOVA",
    "little_mcar": "Prueba de Little (faltantes al azar)",
}

RULE_LABELS_ES = {
    "max_duplicate_pct": "Máx. % de filas repetidas",
    "max_missing_pct": "Máx. % de vacíos",
    "max_iqr_outlier_pct": "Máx. % de valores extremos (IQR)",
    "max_dominant_category_pct": "Máx. % de categoría dominante",
    "non_negative_columns": "Columnas no negativas",
    "id_columns_must_be_unique": "IDs únicos",
    "forbidden_sentinel_values": "Valores trampa",
}

SEVERITY_VALUE_ES = {
    "info": "Info",
    "interesting": "Interesante",
    "notable": "Notable",
    "critical": "Crítico",
}

TYPE_LABELS_ES = {
    "numeric": "Numérica",
    "categorical": "Categórica",
    "datetime": "Fecha/hora",
    "other": "Otra",
}

DTYPE_LABELS_ES = {
    "int64": "entero",
    "int32": "entero",
    "int16": "entero",
    "int8": "entero",
    "Int64": "entero",
    "Int32": "entero",
    "uint64": "entero",
    "uint32": "entero",
    "float64": "decimal",
    "float32": "decimal",
    "Float64": "decimal",
    "Float32": "decimal",
    "object": "texto",
    "string": "texto",
    "str": "texto",
    "bool": "booleano",
    "boolean": "booleano",
    "bool64": "booleano",
    "category": "categoría",
    "datetime64[ns]": "fecha/hora",
    "datetime64": "fecha/hora",
    "timedelta64[ns]": "duración",
}


def dtype_label_es(dtype: Any) -> str:
    s = str(dtype)
    if s in DTYPE_LABELS_ES:
        return DTYPE_LABELS_ES[s]
    sl = s.lower()
    if "datetime" in sl:
        return "fecha/hora"
    if "timedelta" in sl:
        return "duración"
    if "float" in sl:
        return "decimal"
    if "int" in sl or "uint" in sl:
        return "entero"
    if "bool" in sl:
        return "booleano"
    if "object" in sl or "string" in sl or sl == "str":
        return "texto"
    if "category" in sl:
        return "categoría"
    return s


def reason_plain_es(text: Any) -> str:
    s = str(text) if text is not None else ""
    if not s:
        return "—"
    replacements = (
        ("missing parcial", "faltantes en parte de las filas"),
        ("missingness", "patrón de vacíos"),
        ("Se necesitan ≥2 columnas numéricas con missing parcial",
         "Se necesitan ≥2 columnas numéricas con faltantes en parte de las filas"),
        ("Un solo patrón de missingness", "Un solo patrón de vacíos"),
    )
    out = s
    for a, b in replacements:
        out = out.replace(a, b)
    return out

ROLE_LABELS_ES = {
    "identifier": "Identificador",
    "datetime": "Fecha/hora",
    "categorical": "Categórica",
    "numeric": "Numérica",
    "binary": "Binaria",
    "text": "Texto",
    "constant": "Constante",
    "unknown": "Desconocido",
    "target_candidate": "Posible objetivo",
}


def _label_field(k: str) -> str:
    return FIELD_LABELS_ES.get(k, FIELD_LABELS_ES.get(k.lower(), k))


def _fmt_val(v: Any, *, as_p: bool = False, as_pct: bool = False) -> Any:
    if as_p and isinstance(v, (int, float)):
        return format_p_value(float(v))
    if as_pct and isinstance(v, (int, float)):
        return format_pct(float(v), digits=1)
    return display_value(v)


def dict_to_table(d: dict[str, Any], exclude: set[str] | None = None) -> pd.DataFrame:
    exclude = exclude or set()
    rows = []
    for k, v in d.items():
        if k in exclude or isinstance(v, (dict, list, pd.DataFrame)):
            continue
        if k == "ok":
            continue  # no mostrar "Ok True"
        label = _label_field(str(k))
        if k in {"pvalue", "p_value", "p_fdr", "p_raw"}:
            rows.append({"Campo": label, "Valor": format_p_value(v if isinstance(v, (int, float)) else None)})
        elif k in {"method"}:
            rows.append({"Campo": label, "Valor": METHOD_LABELS_ES.get(str(v), v)})
        elif k in {"rule"}:
            rows.append({"Campo": label, "Valor": RULE_LABELS_ES.get(str(v), v)})
        elif k in {"severity"}:
            rows.append({"Campo": label, "Valor": SEVERITY_VALUE_ES.get(str(v), v)})
        elif k in {"reason"}:
            rows.append({"Campo": "Motivo", "Valor": reason_plain_es(v)})
        else:
            rows.append({"Campo": label, "Valor": _fmt_val(v)})
    return pd.DataFrame(rows)


def rename_df_columns_es(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    mapping = {c: _label_field(str(c)) for c in df.columns}
    out = df.rename(columns=mapping)
    for col in list(out.columns):
        cl = str(col).lower()
        original = next((k for k, v in mapping.items() if v == col), col)
        ol = str(original).lower()
        if col in {"Método", "method", "Method"} or cl == "método":
            out[col] = out[col].map(lambda x: METHOD_LABELS_ES.get(str(x), x))
        if col in {"Tipo", "type", "Type"} or cl == "tipo" or ol == "type":
            out[col] = out[col].map(lambda x: TYPE_LABELS_ES.get(str(x)) or dtype_label_es(x))
        if col in {"Regla", "rule"} or ol == "rule":
            out[col] = out[col].map(lambda x: RULE_LABELS_ES.get(str(x), x))
        if col in {"Severidad", "severity"} or ol == "severity":
            out[col] = out[col].map(lambda x: SEVERITY_VALUE_ES.get(str(x), x))
        if "p-valor" in cl or cl in {"p-valor", "p corregido", "p sin corregir"}:
            out[col] = out[col].map(
                lambda x: format_p_value(float(x)) if isinstance(x, (int, float)) else display_value(x)
            )
        if (
            "% vacíos" in str(col)
            or "missing_pct" in ol
            or ol.endswith("pct")
            or "missing_rate" in ol
            or ol == "missing_delta"
            or "Δ % vacíos" in str(col)
        ):
            out[col] = out[col].map(
                lambda x: format_pct(float(x), digits=1) if isinstance(x, (int, float)) else display_value(x)
            )
        # Umbrales/valores de reglas de calidad (keys originales), no la columna genérica "Valor"
        if ol in {"value", "threshold"}:
            out[col] = out[col].map(
                lambda x: (
                    format_pct(float(x), digits=1)
                    if isinstance(x, (int, float)) and abs(float(x)) <= 1
                    else display_value(x)
                )
            )
        # Efecto MAR: diferencia de medias (unidades de la columna predictora), no %
        if ol == "effect" or col.startswith("Efecto"):
            out[col] = out[col].map(
                lambda x: format_number(float(x), digits=2) if isinstance(x, (int, float)) else display_value(x)
            )
        if ol in {"n", "n_missing", "n_observed", "statistic", "df", "n_unique"} or col in {
            "N",
            "Nº vacíos",
            "N observado",
            "Estadístico",
            "Únicos",
        }:
            out[col] = out[col].map(
                lambda x: (
                    format_int_es(x)
                    if isinstance(x, (int, float)) and not isinstance(x, bool) and float(x).is_integer()
                    else display_value(x)
                )
            )
    return dataframe_for_display(out, format_numbers=True)


def descriptive_table_es(report: dict[str, Any]) -> pd.DataFrame:
    """Descriptivos con encabezados en español."""
    if not report:
        return pd.DataFrame()
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "variable"})
    # orden preferido
    order = [
        "variable",
        "n",
        "mean",
        "std",
        "ci_low",
        "ci_high",
        "median",
        "skewness",
        "kurtosis",
        "missing",
        "se",
        "alpha",
        "method",
    ]
    cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
    df = df[cols]
    return rename_df_columns_es(df)


def humanize_granularity_notes(notes: list[str] | None) -> list[str]:
    out = []
    for n in notes or []:
        s = str(n).replace("0 days ", "").replace("0 day ", "")
        s = s.replace("ratio único=", "proporción de valores únicos ≈ ")
        out.append(s)
    return out


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
        st.dataframe(rename_df_columns_es(table), width="stretch", hide_index=True)
    if raw is not None:
        with st.expander("Ver datos crudos"):
            if isinstance(raw, dict):
                st.json(raw)
            else:
                st.write(raw)


def humanize_missingness(miss: dict[str, Any]) -> None:
    summary = miss.get("summary")
    if isinstance(summary, pd.DataFrame):
        st.dataframe(rename_df_columns_es(summary), width="stretch")
    elif isinstance(summary, list):
        st.dataframe(rename_df_columns_es(pd.DataFrame(summary)), width="stretch")

    little = miss.get("little_mcar") or {}
    if little.get("reject_mcar") or (
        little.get("ok") and little.get("p_value") is not None and little.get("p_value") < 0.05
    ):
        st.warning(
            little.get("interpretation")
            or "Hay indicios de que los faltantes no son completamente al azar."
        )
    elif little:
        st.info(
            little.get("interpretation")
            or "No hay evidencia fuerte de que los faltantes dejen de ser al azar en esta muestra."
        )
        with st.expander("Detalle de la prueba de faltantes al azar"):
            st.dataframe(dict_to_table(little), width="stretch", hide_index=True)

    mar = miss.get("mar_screen") or {}
    tests = mar.get("tests") or []
    n_sig = mar.get("n_significant") or sum(1 for t in tests if t.get("significant_fdr"))
    if tests:
        st.write(
            f"**¿Los vacíos dependen de otras columnas?** {n_sig} asociaciones "
            "tras corregir por muchas pruebas."
        )
        st.dataframe(rename_df_columns_es(pd.DataFrame(tests).head(30)), width="stretch")


def _extract_adjusted_coef(adj: dict[str, Any]) -> tuple[Any, Any, str | None]:
    """Devuelve (coef, p, motivo_si_falta)."""
    if adj.get("coef") is not None:
        return adj.get("coef"), adj.get("p_value") or adj.get("pvalue"), None
    nested = adj.get("adjusted")
    if isinstance(nested, dict):
        coef = nested.get("coef")
        p = nested.get("pvalue") or nested.get("p_value")
        if coef is not None:
            return coef, p, None
        return None, p, "el modelo no devolvió un coeficiente para el tratamiento codificado"
    if adj.get("ate") is not None:
        return adj.get("ate"), adj.get("p_value"), None
    if adj.get("estimate") is not None:
        return adj.get("estimate"), adj.get("p_value"), None
    return None, None, "no hubo coeficiente en el resultado"


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

    coef, p, miss_reason = _extract_adjusted_coef(adj)
    treatment = adj.get("treatment")
    if coef is None:
        st.warning(
            "No se pudo calcular la asociación ajustada"
            + (f" ({miss_reason})." if miss_reason else ".")
            + " Si el tratamiento es categórico con varias categorías, el motor "
            "codifica una de ellas; revisá el detalle técnico."
        )
    else:
        st.success(
            f"Asociación ajustada ≈ {format_number(coef)} "
            f"({format_p_clause(p)}). No demuestra causalidad."
        )
        if treatment and "_" in str(treatment):
            st.caption(
                f"Tratamiento codificado como `{treatment}` "
                "(una categoría frente al resto). No implica causalidad."
            )

    flat: dict[str, Any] = {}
    for k, v in adj.items():
        if k in {"summary_text", "model", "residuals", "alternative_explanations", "ok"}:
            continue
        if k == "adjusted" and isinstance(v, dict):
            for nk, nv in v.items():
                if nk in {"model"}:
                    flat["model"] = METHOD_LABELS_ES.get(str(nv), nv)
                elif nk in {"pvalue", "p_value"}:
                    flat["p_value"] = nv
                elif not isinstance(nv, (dict, list)):
                    flat[nk] = nv
            continue
        if isinstance(v, (dict, list, pd.DataFrame)):
            continue
        flat[k] = v
    table = dict_to_table(flat)
    if not table.empty:
        st.dataframe(table, width="stretch", hide_index=True)
    guard = adj.get("interpretation_guardrail")
    if guard:
        st.caption(str(guard))


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
