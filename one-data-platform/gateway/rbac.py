from __future__ import annotations

"""RBAC - Role-Based Access Control. The "what are you allowed to do?" check.

We use the simplest model that matches how the team actually nests: roles are
*levels*, and each higher role includes everything below it.

    analyst (10) < data_scientist (20) < ai_engineer (30) < admin (99)

So a data_scientist can open analyst apps (a superset), and admin can open
everything (admin governs all). An app declares the *minimum* role it needs; a
user passes if their level is at least that high.

This is a teaching simplification. Real systems often use explicit permission
*sets* (e.g. "can_train_models", "can_deploy_llm") instead of a single ladder -
we can graduate to that later without changing the gateway, because every check
goes through `can_access()`.
"""

from typing import Dict

ROLE_LEVELS: Dict[str, int] = {
    "analyst": 10,
    "data_scientist": 20,
    "ai_engineer": 30,
    "admin": 99,
}


def level(role: str) -> int:
    """Numeric rank of a role. Unknown roles get 0 (can open nothing)."""
    return ROLE_LEVELS.get(role, 0)


def can_access(user_role: str, required_role: str) -> bool:
    """True if the user's role is at least the app's required role."""
    return level(user_role) >= level(required_role)
