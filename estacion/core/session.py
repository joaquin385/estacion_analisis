"""Persistencia local de sesiones de análisis."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import uuid

import pandas as pd

from estacion.core.types import DatasetProfile, Finding


DEFAULT_SESSION_DIR = Path("outputs") / "sessions"


@dataclass
class SessionMeta:
    id: str
    name: str
    created_at: str
    updated_at: str
    dataset_name: str
    n_rows: int
    n_cols: int
    n_findings: int
    notes: str = ""


class SessionStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_SESSION_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, session_id: str) -> Path:
        return self.root / session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.root.glob("*/meta.json"), reverse=True):
            try:
                items.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        items.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return items

    def save(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        findings: list[Finding],
        artifacts: dict[str, Any] | None = None,
        name: str | None = None,
        session_id: str | None = None,
        notes: str = "",
        save_parquet: bool = True,
    ) -> SessionMeta:
        sid = session_id or uuid.uuid4().hex[:12]
        now = datetime.now().isoformat(timespec="seconds")
        d = self._dir(sid)
        d.mkdir(parents=True, exist_ok=True)

        meta_path = d / "meta.json"
        created = now
        if meta_path.exists():
            try:
                created = json.loads(meta_path.read_text(encoding="utf-8")).get("created_at", now)
            except Exception:
                pass

        meta = SessionMeta(
            id=sid,
            name=name or profile.name,
            created_at=created,
            updated_at=now,
            dataset_name=profile.name,
            n_rows=profile.n_rows,
            n_cols=profile.n_cols,
            n_findings=len(findings),
            notes=notes,
        )
        meta_path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "profile.json").write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (d / "findings.json").write_text(
            json.dumps([f.to_dict() for f in findings], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        # artifacts: quitar matrices grandes / numpy
        safe_art = _jsonable(artifacts or {})
        (d / "artifacts.json").write_text(
            json.dumps(safe_art, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if save_parquet:
            try:
                df.to_parquet(d / "data.parquet", index=False)
            except Exception:
                df.to_csv(d / "data.csv", index=False)
        else:
            # Siempre persistir datos en CSV para poder rehidratar la sesión
            df.to_csv(d / "data.csv", index=False)
        return meta

    def load(self, session_id: str) -> dict[str, Any]:
        d = self._dir(session_id)
        if not d.exists():
            raise FileNotFoundError(f"Sesión no encontrada: {session_id}")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        profile = json.loads((d / "profile.json").read_text(encoding="utf-8"))
        findings = json.loads((d / "findings.json").read_text(encoding="utf-8"))
        artifacts = {}
        if (d / "artifacts.json").exists():
            artifacts = json.loads((d / "artifacts.json").read_text(encoding="utf-8"))
        df = None
        if (d / "data.parquet").exists():
            try:
                df = pd.read_parquet(d / "data.parquet")
            except Exception:
                df = None
        if df is None and (d / "data.csv").exists():
            df = pd.read_csv(d / "data.csv")
        return {
            "meta": meta,
            "profile": profile,
            "findings": findings,
            "artifacts": artifacts,
            "df": df,
            "path": str(d),
        }

    def delete(self, session_id: str) -> None:
        import shutil

        d = self._dir(session_id)
        if d.exists():
            shutil.rmtree(d)


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k not in {"labels", "pca_coords", "index"}}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in list(obj)[:500]]
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "columns": list(obj.columns), "shape": list(obj.shape)}
    if isinstance(obj, pd.Series):
        return obj.astype(object).where(obj.notna(), None).head(100).tolist()
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "shape": list(obj.shape)}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except Exception:
        pass
    return str(obj)
