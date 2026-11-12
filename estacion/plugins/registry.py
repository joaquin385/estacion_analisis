from __future__ import annotations

from typing import Any

import pandas as pd

from estacion.plugins import healthcare, finance, retail, manufacturing


_PLUGINS: dict[str, dict[str, Any]] = {
    "healthcare": {
        "name": "healthcare",
        "title": "Salud / clínicas",
        "description": "Rangos demográficos, signos vitales, IDs de encuentro, flags de mortalidad.",
        "quality_domain": "healthcare",
        "check": healthcare.run_checks,
    },
    "finance": {
        "name": "finance",
        "title": "Finanzas / mercados",
        "description": "Retornos, precios, volúmenes, fechas monótonas, outliers extremos.",
        "quality_domain": "financial",
        "check": finance.run_checks,
    },
    "retail": {
        "name": "retail",
        "title": "Retail / e-commerce",
        "description": "Precios, cantidades, descuentos, consistencia revenue≈price×qty.",
        "quality_domain": "commercial",
        "check": retail.run_checks,
    },
    "manufacturing": {
        "name": "manufacturing",
        "title": "Manufactura / operaciones",
        "description": "Temperatura, cycle time, sentinelas de sensores, scrap/defectos.",
        "quality_domain": "operational",
        "check": manufacturing.run_checks,
    },
}


def list_plugins() -> list[dict[str, Any]]:
    return [
        {k: v[k] for k in ("name", "title", "description", "quality_domain")}
        for v in _PLUGINS.values()
    ]


def get_plugin(name: str) -> dict[str, Any] | None:
    return _PLUGINS.get(name)


def run_plugin_checks(name: str, df: pd.DataFrame) -> dict[str, Any]:
    plug = _PLUGINS.get(name)
    if not plug:
        return {"ok": False, "reason": f"Plugin desconocido: {name}"}
    result = plug["check"](df)
    result["plugin"] = name
    result["title"] = plug["title"]
    return result
