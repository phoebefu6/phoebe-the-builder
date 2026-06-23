from __future__ import annotations

"""The connector layer - one safe place to reach every data source.

The problem it solves: without this, every app keeps its own copy of database
passwords and API keys, scattered across 60 codebases and `.env` files. Here,
the *wiring* lives in one registry (`connections.yaml`) and the *secrets* live in
environment variables. An app asks for a source by name and gets a ready
connection - it never sees or stores the credential.

Three rules this module enforces:
  1. Secrets come from env vars, never from the YAML or the code.
  2. Secrets are never returned in status/listing output or logs (redacted).
  3. Asking for a source by name is the only way in - `get_connection("orders_db")`.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REGISTRY_FILE = Path(__file__).resolve().parent / "connections.yaml"

# Which config field names hold the *name of an env var* that stores a secret.
SECRET_ENV_FIELDS = ("password_env", "access_key_env", "secret_key_env", "token_env")


@dataclass
class ConnectionSpec:
    name: str
    type: str
    config: Dict[str, Any]

    def secret_env_vars(self) -> List[str]:
        """Env var names this connection needs (may be empty)."""
        return [self.config[f] for f in SECRET_ENV_FIELDS if f in self.config]

    def needs_secret(self) -> bool:
        return bool(self.secret_env_vars())


def load_connections(path: Optional[str] = None) -> Dict[str, ConnectionSpec]:
    """Load the registry into name -> ConnectionSpec."""
    file = Path(path) if path else REGISTRY_FILE
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text()) or {}
    specs: Dict[str, ConnectionSpec] = {}
    for entry in data.get("connections", []):
        name = entry.get("name")
        if not name:
            continue
        specs[name] = ConnectionSpec(name=name, type=entry.get("type", "unknown"),
                                     config={k: v for k, v in entry.items() if k != "name"})
    return specs


def secret_status(spec: ConnectionSpec) -> str:
    """'n/a' (no secret needed), 'configured' (all env vars set), or 'missing'."""
    envs = spec.secret_env_vars()
    if not envs:
        return "n/a"
    return "configured" if all(os.environ.get(e) for e in envs) else "missing"


def redact(spec: ConnectionSpec) -> Dict[str, Any]:
    """A safe-to-log view: wiring + secret *status*, never secret *values*."""
    safe = {k: v for k, v in spec.config.items() if k not in SECRET_ENV_FIELDS}
    return {
        "name": spec.name,
        "type": spec.type,
        "secret": secret_status(spec),
        "needs_env": spec.secret_env_vars(),  # names only, never values
        **{k: v for k, v in safe.items() if k != "description"},
        "description": spec.config.get("description", ""),
    }


def list_status(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Status of every connection - safe to show in a dashboard (no secrets)."""
    return [redact(s) for s in load_connections(path).values()]


def get_connection(name: str, path: Optional[str] = None) -> Any:
    """Return a usable connection for `name`.

    - sqlite -> a real `sqlite3.Connection` (the demo path; needs no secret).
    - other types -> a *resolved config dict* with the secret pulled from env,
      ready to hand to the appropriate driver. Raises if a required secret is
      missing, so an app fails fast and loudly rather than connecting half-configured.
    """
    specs = load_connections(path)
    spec = specs.get(name)
    if spec is None:
        raise KeyError(f"no connection named {name!r}")

    status = secret_status(spec)
    if status == "missing":
        missing = [e for e in spec.secret_env_vars() if not os.environ.get(e)]
        raise RuntimeError(
            f"connection {name!r} is missing secret(s): set env var(s) {missing}"
        )

    if spec.type == "sqlite":
        return sqlite3.connect(spec.config.get("database", ":memory:"))

    # For remote types we resolve (not print) the secret and return a config the
    # caller's driver would use. The secret value stays inside this dict, never logged.
    resolved = {k: v for k, v in spec.config.items() if k not in SECRET_ENV_FIELDS}
    resolved["type"] = spec.type
    for field in SECRET_ENV_FIELDS:
        if field in spec.config:
            key = field.replace("_env", "")  # password_env -> password
            resolved[key] = os.environ.get(spec.config[field])
    return resolved
