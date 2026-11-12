"""
Cliente LLM local vía Ollama (sin API paga).

Requiere Ollama corriendo en localhost. Si no está, el intérprete degrada a plantillas.
"""

from __future__ import annotations

from typing import Any
import json
import urllib.request
import urllib.error


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data.get("response", "")).strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama no disponible en {self.host}: {exc}") from exc


def get_optional_llm(prefer: str = "ollama", **kwargs: Any):
    """Factory: intenta Ollama; si falla retorna None."""
    if prefer == "ollama":
        client = OllamaClient(**kwargs)
        return client if client.available() else None
    return None
