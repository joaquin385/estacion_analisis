"""Segmentación: K-means, jerárquico, DBSCAN y Gower-like para mixtos."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA


def _prepare_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    max_rows: int = 8_000,
) -> tuple[np.ndarray, pd.Index, dict]:
    categorical_cols = categorical_cols or []
    cols = numeric_cols + categorical_cols
    data = df[cols].dropna()
    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
        sampled = True
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No hay columnas para segmentar")
    ct = ColumnTransformer(transformers)
    X = ct.fit_transform(data)
    meta = {"sampled": sampled, "n": len(data), "n_features": X.shape[1]}
    return X, data.index, meta


def _eval_labels(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels >= 0  # DBSCAN noise = -1
    if mask.sum() < 10 or len(set(labels[mask])) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "n_clusters": 0}
    try:
        sil = float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        sil = np.nan
    try:
        ch = float(calinski_harabasz_score(X[mask], labels[mask]))
    except Exception:
        ch = np.nan
    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "n_clusters": int(len(set(labels[mask]))),
        "noise_pct": float((labels < 0).mean()),
    }


def run_segmentation(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str] | None = None,
    k_range: range | list[int] | None = None,
) -> dict[str, Any]:
    """
    Ejecuta varios métodos y evalúa calidad.
    No asume que existen grupos naturales: reporta si silhouette es débil.
    """
    categorical_cols = categorical_cols or []
    # Limitar categóricas de alta cardinalidad
    categorical_cols = [
        c for c in categorical_cols if 1 < df[c].nunique(dropna=True) <= 15
    ]
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 2]
    if len(numeric_cols) + len(categorical_cols) < 2:
        return {"ok": False, "reason": "Variables insuficientes para segmentar"}

    try:
        X, idx, meta = _prepare_matrix(df, numeric_cols, categorical_cols)
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}

    if k_range is None:
        k_range = range(2, min(8, max(3, meta["n"] // 50) + 1))

    kmeans_results = []
    best_kmeans = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        metrics = _eval_labels(X, labels)
        row = {"k": k, **metrics, "inertia": float(km.inertia_)}
        kmeans_results.append(row)
        if best_kmeans is None or (
            pd.notna(metrics["silhouette"])
            and metrics["silhouette"] > (best_kmeans.get("silhouette") or -1)
        ):
            best_kmeans = {**row, "labels": labels}

    # Jerárquico con mejor k de kmeans o k=3
    k_h = int(best_kmeans["k"]) if best_kmeans else 3
    agg = AgglomerativeClustering(n_clusters=k_h)
    labels_h = agg.fit_predict(X)
    hier = {"k": k_h, **_eval_labels(X, labels_h)}

    # DBSCAN — sensibilidad a eps: probar grid pequeño
    dbscan_grid = []
    best_db = None
    # escala aproximada tras StandardScaler
    for eps in (0.5, 0.8, 1.2, 1.8):
        for ms in (5, 10):
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            metrics = _eval_labels(X, labels)
            row = {"eps": eps, "min_samples": ms, **metrics}
            dbscan_grid.append(row)
            if metrics["n_clusters"] >= 2 and (
                best_db is None
                or (metrics.get("silhouette") or -1) > (best_db.get("silhouette") or -1)
            ):
                best_db = {**row, "labels": labels}

    # Estabilidad kmeans: comparar dos semillas
    stability = None
    if best_kmeans is not None:
        k = best_kmeans["k"]
        l1 = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(X)
        l2 = KMeans(n_clusters=k, n_init=10, random_state=2).fit_predict(X)
        # Adjusted agreement via contingency / max matching approx: ARI
        from sklearn.metrics import adjusted_rand_score

        stability = {
            "ari_two_seeds": float(adjusted_rand_score(l1, l2)),
            "interpretation": "ARI~1 estable; ARI bajo → sensible a inicialización",
        }

    sil = best_kmeans.get("silhouette") if best_kmeans else np.nan
    natural_groups_likely = bool(pd.notna(sil) and sil >= 0.25)

    # PCA 2D para viz
    coords = None
    if X.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)

    # Mixtos: Gower + k-prototypes
    mixed = None
    if categorical_cols and numeric_cols:
        from estacion.clustering.gower import cluster_gower, kprototypes

        k_m = int(best_kmeans["k"]) if best_kmeans else 3
        gow = cluster_gower(df, numeric_cols, categorical_cols, n_clusters=k_m)
        kp = kprototypes(df, numeric_cols, categorical_cols, n_clusters=k_m)
        mixed = {
            "gower": {k: v for k, v in gow.items() if k not in {"labels", "index"}},
            "kprototypes": {k: v for k, v in kp.items() if k not in {"labels", "index"}},
            "gower_labels": gow.get("labels"),
            "kprototypes_labels": kp.get("labels"),
        }
        if gow.get("ok") and pd.notna(gow.get("silhouette")) and gow["silhouette"] >= 0.25:
            natural_groups_likely = True

    return {
        "ok": True,
        "meta": meta,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "kmeans": {
            "grid": [{k: v for k, v in r.items() if k != "labels"} for r in kmeans_results],
            "best": {k: v for k, v in (best_kmeans or {}).items() if k != "labels"},
            "labels": (best_kmeans or {}).get("labels"),
            "index": idx,
        },
        "hierarchical": hier,
        "dbscan": {
            "grid": dbscan_grid,
            "best": {k: v for k, v in (best_db or {}).items() if k != "labels"},
            "labels": (best_db or {}).get("labels"),
        },
        "mixed": mixed,
        "stability": stability,
        "natural_groups_likely": natural_groups_likely,
        "caveat": (
            "No encontramos grupos naturales claros (Silhouette < 0.25). "
            "Los colores del gráfico son una partición forzada; no los uses como segmentos de negocio."
            if not natural_groups_likely
            else "Hay indicios de estructura de grupos (Silhouette ≥ 0.25); validá interpretabilidad de negocio."
        ),
        "pca_coords": coords,
        "limitations": [
            "K-means asume clusters esféricos de tamaño similar.",
            "DBSCAN es sensible a eps/min_samples.",
            "Gower es O(n²) y se subsamplea; k-prototypes usa gamma heurístico.",
            "Muestreo aplicado si n es grande.",
        ],
    }
