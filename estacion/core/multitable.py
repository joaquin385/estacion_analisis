"""Grafo de relaciones entre tablas (joins candidatos)."""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

from estacion.core.loader import suggest_join_keys


def build_relationship_graph(
    tables: dict[str, pd.DataFrame],
    min_overlap: float = 0.3,
) -> dict[str, Any]:
    suggestions = suggest_join_keys(tables, min_overlap=min_overlap)
    g = nx.Graph()
    for name, df in tables.items():
        g.add_node(
            name,
            n_rows=len(df),
            n_cols=df.shape[1],
            columns=list(map(str, df.columns))[:40],
        )
    for s in suggestions:
        g.add_edge(
            s["left"],
            s["right"],
            key=s["key"],
            overlap=s["overlap"],
            weight=s["overlap"],
        )

    components = [list(c) for c in nx.connected_components(g)]
    # ranking de hubs
    degree = dict(g.degree())
    return {
        "nodes": [
            {"id": n, **g.nodes[n]} for n in g.nodes
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "key": d.get("key"),
                "overlap": d.get("overlap"),
            }
            for u, v, d in g.edges(data=True)
        ],
        "n_components": len(components),
        "components": components,
        "hubs": sorted(degree.items(), key=lambda x: -x[1]),
        "suggestions": suggestions,
        "note": (
            "Grafo heurístico por nombre de columna + solapamiento de valores. "
            "No implica FK reales de un esquema."
        ),
    }


def suggest_analysis_paths(graph: dict[str, Any]) -> list[str]:
    paths = []
    for e in graph.get("edges", []):
        paths.append(
            f"Unir `{e['source']}` ⋈ `{e['target']}` por `{e['key']}` "
            f"(overlap={e['overlap']:.0%}) y perfilar la tabla resultante."
        )
    if graph.get("n_components", 1) > 1:
        paths.append(
            "Hay componentes desconectados: revisar claves con nombres distintos "
            "(p.ej. customer_id vs id_cliente)."
        )
    if not paths:
        paths.append("No se detectaron joins claros; comparar esquemas manualmente.")
    return paths
