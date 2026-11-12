"""Editor DAG interactivo vía vis-network (CDN) embebido en Streamlit."""

from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components

from estacion.causal.dag_editor import build_dag, validate_dag


def render_interactive_dag(
    nodes: list[str],
    edges: list[list[str]] | None = None,
    roles: dict[str, str] | None = None,
    height: int = 520,
) -> dict[str, Any] | None:
    """
    Muestra un grafo editable (arrastrar nodos; doble-click arista para crear desde selección).

    Limitación Streamlit: la edición de aristas se hace con botones HTML + textArea
    que el usuario confirma; no hay round-trip automático sin componente custom publicado.
    Devolvemos el HTML y un protocolo: el usuario copia el JSON exportado.
    """
    edges = edges or []
    roles = roles or {n: "other" for n in nodes}
    color_map = {
        "treatment": "#E36414",
        "outcome": "#0F4C5C",
        "confounder": "#5C4D7A",
        "mediator": "#2A9D8F",
        "collider": "#BC4749",
        "other": "#6D597A",
    }
    vis_nodes = [
        {
            "id": n,
            "label": n,
            "color": color_map.get(roles.get(n, "other"), "#6D597A"),
            "title": f"{n} ({roles.get(n, 'other')})",
        }
        for n in nodes
    ]
    vis_edges = [{"from": a, "to": b, "arrows": "to"} for a, b in edges]

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin:0; font-family: sans-serif; background:#F7F4EF; }}
    #mynetwork {{ width:100%; height:{height-120}px; border:1px solid #ccc; background:white; }}
    .bar {{ padding:8px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
    button {{ padding:6px 10px; cursor:pointer; }}
    textarea {{ width:100%; height:70px; font-family:monospace; font-size:12px; }}
    .hint {{ color:#555; font-size:12px; padding:0 8px; }}
  </style>
</head>
<body>
  <div class="bar">
    <label>Desde <select id="src"></select></label>
    <label>Hacia <select id="dst"></select></label>
    <button onclick="addEdge()">Agregar arista</button>
    <button onclick="delEdge()">Quitar arista</button>
    <button onclick="exportDag()">Exportar JSON</button>
  </div>
  <div class="hint">Arrastrá nodos para reorganizar. Usá los selects para editar aristas y luego Exportar.</div>
  <div id="mynetwork"></div>
  <textarea id="out" readonly placeholder="JSON del DAG exportado"></textarea>
  <script>
    const nodes = new vis.DataSet({json.dumps(vis_nodes)});
    const edges = new vis.DataSet({json.dumps(vis_edges)});
    const nodeIds = {json.dumps(nodes)};
    const roles = {json.dumps(roles)};
    const src = document.getElementById('src');
    const dst = document.getElementById('dst');
    nodeIds.forEach(n => {{
      src.innerHTML += `<option value="${{n}}">${{n}}</option>`;
      dst.innerHTML += `<option value="${{n}}">${{n}}</option>`;
    }});
    const container = document.getElementById('mynetwork');
    const network = new vis.Network(container, {{nodes, edges}}, {{
      physics: {{enabled: true, stabilization: true}},
      interaction: {{hover: true, dragNodes: true}},
      edges: {{smooth: true}},
    }});
    function addEdge() {{
      const a = src.value, b = dst.value;
      if (a === b) return;
      const exists = edges.get().some(e => e.from===a && e.to===b);
      if (!exists) edges.add({{from:a, to:b, arrows:'to'}});
    }}
    function delEdge() {{
      const a = src.value, b = dst.value;
      edges.get().forEach(e => {{ if (e.from===a && e.to===b) edges.remove(e.id); }});
    }}
    function exportDag() {{
      const payload = {{
        nodes: nodeIds,
        edges: edges.get().map(e => [e.from, e.to]),
        roles: roles,
      }};
      document.getElementById('out').value = JSON.stringify(payload);
    }}
    exportDag();
  </script>
</body>
</html>
"""
    components.html(html, height=height, scrolling=True)
    return None


def parse_exported_dag_json(text: str) -> dict[str, Any]:
    data = json.loads(text)
    dag = build_dag(data.get("nodes") or [], data.get("edges") or [], data.get("roles") or {})
    return {"dag": dag, "validation": validate_dag(dag)}
