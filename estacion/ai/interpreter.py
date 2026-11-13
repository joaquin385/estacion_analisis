"""
Capa de IA opcional y desacoplada.

NO calcula estadísticas. Solo puede reformular hallazgos ya computados.
Por defecto funciona sin API: genera narrativas deterministas a partir de Finding.
"""

from __future__ import annotations

from typing import Any

from estacion.core.types import Finding


def interpret_findings(
    findings: list[Finding],
    style: str = "analyst",
    llm_client: Any | None = None,
    try_ollama: bool = False,
    ollama_model: str = "llama3.2",
) -> list[dict[str, str]]:
    """
    Parameters
    ----------
    llm_client : optional
        Objeto con método `.complete(prompt: str) -> str`. Si es None, usa plantillas locales.
    try_ollama : bool
        Si True y no hay llm_client, intenta Ollama local (sin API paga).
    """
    client = llm_client
    if client is None and try_ollama:
        try:
            from estacion.ai.ollama_client import get_optional_llm

            client = get_optional_llm(model=ollama_model)
        except Exception:
            client = None

    narratives = []
    for f in findings:
        if client is not None:
            prompt = _build_prompt(f)
            try:
                text = client.complete(prompt)
            except Exception as exc:
                text = _template_narrative(f) + f"\n\n_(LLM no disponible: {exc})_"
        else:
            text = _template_narrative(f)
        narratives.append({"id": f.id, "title": f.title, "narrative": text})
    return narratives


def _build_prompt(f: Finding) -> str:
    return (
        "Sos un analista de datos senior. A partir del hallazgo siguiente, "
        "escribí una interpretación breve en español. "
        "NO inventes cifras que no estén en la evidencia. "
        "Recordá que correlación no implica causalidad. "
        "Mencioná limitaciones y explicaciones alternativas.\n\n"
        f"Título: {f.title}\n"
        f"Observación: {f.observation}\n"
        f"Evidencia: {f.evidence}\n"
        f"Método: {f.method}\n"
        f"Alternativas: {f.alternative_explanations}\n"
        f"Limitaciones: {f.limitations}\n"
    )


def _template_narrative(f: Finding) -> str:
    from estacion.ux.presentation import format_magnitude, humanize_finding

    sig = ""
    if f.statistically_significant is True:
        sig = "El resultado es estadísticamente significativo bajo el criterio usado. "
    elif f.statistically_significant is False:
        sig = "No alcanza significancia estadística bajo el criterio usado. "
    prac = ""
    if f.practically_important is True:
        prac = "La magnitud sugiere relevancia práctica potencial. "
    elif f.practically_important is False:
        prac = "La magnitud puede ser limitada en términos prácticos. "

    h = humanize_finding(f)
    mag = format_magnitude(f.magnitude, finding=f)
    alts = "; ".join(f.alternative_explanations[:3])
    follow = "; ".join(f.follow_up[:2])
    return (
        f"{h['what']} {sig}{prac}"
        f"Método: {f.method}. Magnitud: {mag}. "
        f"Explicaciones alternativas: {alts}. "
        f"Para confirmar o refutar: {follow}."
    )
