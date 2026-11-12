"""
Distancia de Gower y k-prototypes para variables mixtas.

Implementación propia (sin dependencia extra) para uso local.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def gower_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> np.ndarray:
    """Matriz de distancias de Gower en [0, 1]."""
    data = df[numeric_cols + categorical_cols].copy()
    n = len(data)
    if n == 0:
        return np.zeros((0, 0))

    num = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    ranges = {}
    for c in numeric_cols:
        r = float(num[c].max() - num[c].min())
        ranges[c] = r if r > 0 else 1.0

    cats = data[categorical_cols].astype(str)
    D = np.zeros((n, n), dtype=float)
    # Simetría: solo triangular superior
    num_vals = num.to_numpy(dtype=float)
    cat_vals = cats.to_numpy()
    n_num = len(numeric_cols)
    n_cat = len(categorical_cols)
    ranges_arr = np.array([ranges[c] for c in numeric_cols], dtype=float) if n_num else np.array([])

    for i in range(n):
        for j in range(i + 1, n):
            d_num = 0.0
            w_num = 0
            for k in range(n_num):
                ai, bi = num_vals[i, k], num_vals[j, k]
                if np.isfinite(ai) and np.isfinite(bi):
                    d_num += abs(ai - bi) / ranges_arr[k]
                    w_num += 1
            d_cat = 0.0
            w_cat = 0
            for k in range(n_cat):
                if cat_vals[i, k] != "nan" and cat_vals[j, k] != "nan":
                    d_cat += 0.0 if cat_vals[i, k] == cat_vals[j, k] else 1.0
                    w_cat += 1
            w = w_num + w_cat
            dist = (d_num + d_cat) / w if w else 0.0
            D[i, j] = D[j, i] = dist
    return D


def cluster_gower(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_clusters: int = 3,
    max_rows: int = 1500,
) -> dict[str, Any]:
    data = df[numeric_cols + categorical_cols].dropna()
    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)
        sampled = True
    if len(data) < max(20, n_clusters * 5):
        return {"ok": False, "reason": "n insuficiente para Gower clustering"}

    D = gower_matrix(data, numeric_cols, categorical_cols)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(D)
    try:
        sil = float(silhouette_score(D, labels, metric="precomputed"))
    except Exception:
        sil = np.nan
    return {
        "ok": True,
        "method": "gower_agglomerative",
        "n_clusters": n_clusters,
        "silhouette": sil,
        "sampled": sampled,
        "n": len(data),
        "labels": labels,
        "index": data.index,
        "limitations": [
            "Gower en O(n²); se subsamplea si n es grande.",
            "Linkage average sobre distancias precomputadas.",
        ],
    }


def kprototypes(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_clusters: int = 3,
    gamma: float | None = None,
    max_iter: int = 50,
    random_state: int = 42,
    max_rows: int = 5000,
) -> dict[str, Any]:
    """
    K-prototypes (Huang): mezcla k-means (num) + k-modes (cat).
    """
    data = df[numeric_cols + categorical_cols].dropna().copy()
    sampled = False
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=random_state)
        sampled = True
    n = len(data)
    if n < max(20, n_clusters * 5):
        return {"ok": False, "reason": "n insuficiente"}

    Xn = data[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # estandarizar numéricas
    mu = Xn.mean(axis=0)
    sd = Xn.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (Xn - mu) / sd
    Xc = data[categorical_cols].astype(str).to_numpy()

    if gamma is None:
        gamma = 0.5 * Xn.std()  # heurística Huang

    rng = np.random.default_rng(random_state)
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    cen_n = Xn[init_idx].copy()
    cen_c = Xc[init_idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # assignment
        new_labels = np.empty(n, dtype=int)
        for i in range(n):
            best, best_d = 0, np.inf
            for k in range(n_clusters):
                d_num = np.sum((Xn[i] - cen_n[k]) ** 2)
                d_cat = np.sum(Xc[i] != cen_c[k])
                d = d_num + gamma * d_cat
                if d < best_d:
                    best_d, best = d, k
            new_labels[i] = best
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # update
        for k in range(n_clusters):
            members = labels == k
            if not members.any():
                idx = rng.integers(0, n)
                cen_n[k] = Xn[idx]
                cen_c[k] = Xc[idx]
                continue
            cen_n[k] = Xn[members].mean(axis=0)
            # moda por columna
            for j in range(Xc.shape[1]):
                vals, counts = np.unique(Xc[members, j], return_counts=True)
                cen_c[k, j] = vals[np.argmax(counts)]

    # silhouette en espacio numérico estandarizado (approx)
    try:
        sil = float(silhouette_score(Xn, labels)) if len(set(labels)) > 1 else np.nan
    except Exception:
        sil = np.nan

    sizes = {int(k): int((labels == k).sum()) for k in range(n_clusters)}
    return {
        "ok": True,
        "method": "kprototypes",
        "n_clusters": n_clusters,
        "gamma": float(gamma),
        "silhouette_numeric_approx": sil,
        "cluster_sizes": sizes,
        "sampled": sampled,
        "n": n,
        "labels": labels,
        "index": data.index,
        "centroids_numeric": cen_n.tolist(),
        "centroids_categorical": cen_c.tolist(),
        "limitations": [
            "Silhouette reportado solo sobre el bloque numérico estandarizado.",
            "gamma por defecto es heurístico.",
            "Inicialización aleatoria; repetir semillas para estabilidad.",
        ],
    }
