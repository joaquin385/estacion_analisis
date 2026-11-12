"""Edición y validación de DAGs causales hipotéticos."""

from __future__ import annotations

from typing import Any

import networkx as nx


ALLOWED_ROLES = ("treatment", "outcome", "confounder", "mediator", "collider", "other")


def empty_dag() -> dict[str, Any]:
    return {
        "nodes": [],
        "edges": [],
        "roles": {},
        "note": "DAG hipotético editable; no se infiere de los datos.",
    }


def build_dag(
    nodes: list[str],
    edges: list[tuple[str, str] | list[str]],
    roles: dict[str, str] | None = None,
) -> dict[str, Any]:
    roles = roles or {}
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    clean_edges = []
    for e in edges:
        if len(e) != 2:
            continue
        a, b = str(e[0]), str(e[1])
        if a in nodes and b in nodes and a != b:
            g.add_edge(a, b)
            clean_edges.append([a, b])
    return {
        "nodes": list(g.nodes),
        "edges": clean_edges,
        "roles": {n: roles.get(n, "other") for n in g.nodes},
        "note": "DAG hipotético editable; no se infiere de los datos.",
    }


def validate_dag(dag: dict[str, Any]) -> dict[str, Any]:
    g = nx.DiGraph()
    g.add_nodes_from(dag.get("nodes") or [])
    g.add_edges_from([tuple(e) for e in dag.get("edges") or []])
    roles = dag.get("roles") or {}

    issues = []
    warnings = []
    if not nx.is_directed_acyclic_graph(g):
        issues.append("El grafo tiene ciclos; un DAG causal debe ser acíclico.")
        try:
            cycle = nx.find_cycle(g, orientation="original")
            issues.append(f"Ciclo ejemplo: {cycle}")
        except Exception:
            pass

    treatments = [n for n, r in roles.items() if r == "treatment"]
    outcomes = [n for n, r in roles.items() if r == "outcome"]
    if len(treatments) != 1:
        warnings.append("Se recomienda exactamente 1 nodo con rol 'treatment'.")
    if len(outcomes) != 1:
        warnings.append("Se recomienda exactamente 1 nodo con rol 'outcome'.")

    # colliders: dos padres al mismo nodo
    colliders = [n for n in g.nodes if g.in_degree(n) >= 2]
    for n in colliders:
        if roles.get(n) not in {"collider", "outcome", "mediator"}:
            warnings.append(
                f"'{n}' tiene ≥2 padres (posible collider). Cuidado al condicionar sobre él."
            )

    # backdoor crude: confounders should point to both T and Y
    adjustment_suggestion = None
    if len(treatments) == 1 and len(outcomes) == 1:
        t, y = treatments[0], outcomes[0]
        confounders = [n for n, r in roles.items() if r == "confounder"]
        mediators = [n for n, r in roles.items() if r == "mediator"]
        # check edges exist heuristically
        for c in confounders:
            if not (g.has_edge(c, t) and g.has_edge(c, y)):
                warnings.append(
                    f"Confusor '{c}' debería apuntar a treatment y outcome en el DAG."
                )
        for m in mediators:
            if not (g.has_edge(t, m) and g.has_edge(m, y)):
                warnings.append(
                    f"Mediador '{m}' debería estar en el camino treatment→mediator→outcome."
                )
        adjustment_suggestion = {
            "adjust_for": confounders,
            "do_not_adjust_for_mediators_if_total_effect": mediators,
            "do_not_adjust_for_colliders": [n for n, r in roles.items() if r == "collider"],
            "note": (
                "Sugerencia heurística por roles; no es identificación automática "
                "(p.ej. criterio backdoor formal completo)."
            ),
        }

    return {
        "ok": len(issues) == 0,
        "is_dag": nx.is_directed_acyclic_graph(g),
        "issues": issues,
        "warnings": warnings,
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "adjustment_suggestion": adjustment_suggestion,
    }


def dag_to_plotly(dag: dict[str, Any]):
    """Figura plotly simple (layout spring)."""
    import plotly.graph_objects as go

    g = nx.DiGraph()
    g.add_nodes_from(dag.get("nodes") or [])
    g.add_edges_from([tuple(e) for e in dag.get("edges") or []])
    if g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="DAG vacío")
        return fig

    pos = nx.spring_layout(g, seed=42)
    roles = dag.get("roles") or {}
    color_map = {
        "treatment": "#E36414",
        "outcome": "#0F4C5C",
        "confounder": "#5C4D7A",
        "mediator": "#2A9D8F",
        "collider": "#BC4749",
        "other": "#6D597A",
    }

    edge_x, edge_y = [], []
    for a, b in g.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", line=dict(width=1.5, color="#999"), hoverinfo="none"
    )

    node_x = [pos[n][0] for n in g.nodes]
    node_y = [pos[n][1] for n in g.nodes]
    node_color = [color_map.get(roles.get(n, "other"), "#6D597A") for n in g.nodes]
    node_text = [f"{n}<br>rol={roles.get(n, 'other')}" for n in g.nodes]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in g.nodes],
        textposition="top center",
        marker=dict(size=22, color=node_color, line=dict(width=1, color="#333")),
        hovertext=node_text,
        hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="DAG causal hipotético",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#F7F4EF",
        height=420,
    )
    return fig


def suggest_edges_from_roles(roles: dict[str, str]) -> list[list[str]]:
    """Aristas típicas a partir de roles (punto de partida editable)."""
    t = [n for n, r in roles.items() if r == "treatment"]
    y = [n for n, r in roles.items() if r == "outcome"]
    conf = [n for n, r in roles.items() if r == "confounder"]
    med = [n for n, r in roles.items() if r == "mediator"]
    edges = []
    if t and y:
        edges.append([t[0], y[0]])
        for c in conf:
            edges.append([c, t[0]])
            edges.append([c, y[0]])
        for m in med:
            edges.append([t[0], m])
            edges.append([m, y[0]])
    return edges
