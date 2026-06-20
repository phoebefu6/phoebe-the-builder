from __future__ import annotations

"""Load and query the app registry (`registry/apps.yaml`).

The gateway uses this to answer two questions:
  - "What apps exist?"            -> load_apps()
  - "Which can THIS role open?"   -> visible_apps(role)

Keeping registry access behind these functions means the source could later be a
database instead of a YAML file without changing the gateway.
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from rbac import can_access

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "registry" / "apps.yaml"


def load_apps() -> List[Dict[str, str]]:
    """Return every app entry from the registry file."""
    if not REGISTRY_FILE.exists():
        return []
    data = yaml.safe_load(REGISTRY_FILE.read_text()) or {}
    return data.get("apps", [])


def get_app(slug: str) -> Optional[Dict[str, str]]:
    for app in load_apps():
        if app.get("slug") == slug:
            return app
    return None


def visible_apps(role: str) -> List[Dict[str, str]]:
    """Apps this role is allowed to open, with an 'allowed' flag already resolved.

    We return ALL apps but tag each with whether the role can open it - the
    workspace can then show allowed apps and (optionally) hint at locked ones.
    """
    out: List[Dict[str, str]] = []
    for app in load_apps():
        allowed = can_access(role, str(app.get("required_role", "admin")))
        out.append({**app, "allowed": allowed})
    return out
