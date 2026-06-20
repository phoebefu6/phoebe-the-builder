from __future__ import annotations

"""The user store - who can log in, and what role they have.

For the MVP this is a JSON file (`users.json`) next to this code. It's readable,
debuggable, and good enough to learn with. Later we swap it for a real database
(SQLite/Postgres) without changing the gateway - that's the point of keeping this
behind a few small functions.

Roles are the seed of RBAC (Step 2). We store them now so login already knows who
is an analyst vs a data scientist vs an admin.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from auth import hash_password, verify_password

USERS_FILE = Path(__file__).parent / "users.json"

# Demo accounts seeded on first run. Passwords are DEMO ONLY and intentionally
# simple so you can log in and explore. Real users would be created by an admin.
# The file stores only salted hashes - never these plain passwords.
_SEED = [
    {"email": "phoebe@team.io", "password": "admin123", "role": "admin"},
    {"email": "ana@team.io", "password": "analyst123", "role": "analyst"},
    {"email": "sam@team.io", "password": "scientist123", "role": "data_scientist"},
    {"email": "ria@team.io", "password": "aieng123", "role": "ai_engineer"},
]


def _load() -> Dict[str, Dict[str, str]]:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save(users: Dict[str, Dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))


def seed_if_empty() -> None:
    """Create the demo users on first run. Stores hashed passwords only."""
    users = _load()
    if users:
        return
    users = {
        u["email"]: {"password_hash": hash_password(u["password"]), "role": u["role"]}
        for u in _SEED
    }
    _save(users)


def authenticate(email: str, password: str) -> Optional[str]:
    """Return the user's role if email+password are correct, else None."""
    seed_if_empty()  # self-healing: make sure demo users exist before any login
    users = _load()
    record = users.get(email.strip().lower())
    if record is None:
        return None
    if not verify_password(password, record["password_hash"]):
        return None
    return record["role"]


def get_role(email: str) -> Optional[str]:
    record = _load().get(email.strip().lower())
    return record["role"] if record else None
