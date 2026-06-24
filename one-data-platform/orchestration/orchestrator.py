from __future__ import annotations

"""The orchestration layer - govern scheduled pipelines without rebuilding a scheduler.

ADR-0001 decision applied: we do NOT write our own cron/DAG engine. We plug in
**Apache Airflow** (open source, battle-tested) and the platform *governs and surfaces*
it. This module is a thin abstraction with two backends:

  - `AirflowOrchestrator`  - talks to a real Airflow via its REST API when AIRFLOW_URL
                             is set (lazy httpx import).
  - `LocalOrchestrator`    - a deterministic simulation from `pipelines.yaml` so the
                             platform runs anywhere with no Airflow installed.

`get_orchestrator()` returns the real one if configured, else the local sim - exactly
like the connector layer falls back gracefully. Apps and the gateway only use the
interface (`list_dags`, `get_dag`, `trigger`), so the backend can change underneath.
"""

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PIPELINES_FILE = Path(__file__).resolve().parent / "pipelines.yaml"


def load_pipelines(path: Optional[str] = None) -> List[Dict[str, Any]]:
    file = Path(path) if path else PIPELINES_FILE
    if not file.exists():
        return []
    data = yaml.safe_load(file.read_text()) or {}
    return data.get("pipelines", [])


@dataclass
class Run:
    dag_id: str
    status: str
    started: str
    duration_s: float


class LocalOrchestrator:
    """Offline simulation - deterministic last-run status per DAG per day."""

    backend = "local-sim"

    def __init__(self, path: Optional[str] = None) -> None:
        self._pipelines = load_pipelines(path)

    def _sim_status(self, dag_id: str) -> str:
        # Stable per dag+date: ~1 in 5 "failed" so the UI shows both states.
        key = f"{dag_id}:{datetime.now(timezone.utc).date()}"
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        return "failed" if h % 5 == 0 else "success"

    def list_dags(self) -> List[Dict[str, Any]]:
        out = []
        for p in self._pipelines:
            out.append({
                "id": p["id"],
                "schedule": p.get("schedule", ""),
                "owner": p.get("owner", ""),
                "description": p.get("description", ""),
                "tasks": p.get("tasks", []),
                "last_status": self._sim_status(p["id"]),
            })
        return out

    def get_dag(self, dag_id: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.list_dags() if d["id"] == dag_id), None)

    def trigger(self, dag_id: str) -> Run:
        if self.get_dag(dag_id) is None:
            raise KeyError(f"no pipeline {dag_id!r}")
        return Run(dag_id=dag_id, status="queued",
                   started=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   duration_s=0.0)


class AirflowOrchestrator:
    """Talks to a real Airflow via its stable REST API. Used when AIRFLOW_URL is set."""

    backend = "airflow"

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _client(self):
        import httpx  # lazy - only needed when a real Airflow is wired in
        auth = (os.environ.get("AIRFLOW_USER", "admin"), os.environ.get("AIRFLOW_PASSWORD", ""))
        return httpx.Client(base_url=self.base_url, auth=auth, timeout=10)

    def list_dags(self) -> List[Dict[str, Any]]:
        with self._client() as c:
            dags = c.get("/api/v1/dags").json().get("dags", [])
        out = []
        for d in dags:
            out.append({"id": d["dag_id"], "schedule": d.get("schedule_interval", ""),
                        "owner": ",".join(d.get("owners", [])), "description": d.get("description", ""),
                        "tasks": [], "last_status": "unknown"})
        return out

    def get_dag(self, dag_id: str) -> Optional[Dict[str, Any]]:
        return next((d for d in self.list_dags() if d["id"] == dag_id), None)

    def trigger(self, dag_id: str) -> Run:
        with self._client() as c:
            c.post(f"/api/v1/dags/{dag_id}/dagRuns", json={})
        return Run(dag_id=dag_id, status="queued",
                   started=datetime.now(timezone.utc).isoformat(timespec="seconds"), duration_s=0.0)


def get_orchestrator():
    """Real Airflow if AIRFLOW_URL is set, else the local simulation."""
    url = os.environ.get("AIRFLOW_URL")
    return AirflowOrchestrator(url) if url else LocalOrchestrator()
