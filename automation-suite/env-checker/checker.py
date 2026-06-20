from __future__ import annotations

"""Core logic: validate environment variables against a schema *before* deploy.

Deployments fail at 2am because `DATABASE_URL` was never set, or `PORT` is the
string "eight thousand". This module reads a small schema, checks the actual
environment, and returns a structured report. The CLI turns a non-empty
'missing'/'invalid' report into a non-zero exit so CI blocks the deploy.

Schema format (one var per line):

    NAME=required                 # a required var, any value
    NAME=required:^postgres://    # required, value must match the regex
    NAME=optional:^\\d+$           # optional, but if set must match
    # lines starting with # are comments
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VarSpec:
    name: str
    required: bool
    pattern: Optional[str] = None
    comment: str = ""


def parse_schema(text: str) -> List[VarSpec]:
    """Parse schema text into VarSpecs. Malformed lines are skipped, not fatal."""
    specs: List[VarSpec] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Split off an inline comment.
        comment = ""
        if "#" in line:
            line, comment = line.split("#", 1)
            line, comment = line.strip(), comment.strip()
        if "=" not in line:
            continue
        name, rule = line.split("=", 1)
        name, rule = name.strip(), rule.strip()
        if not name:
            continue
        pattern: Optional[str] = None
        if ":" in rule:
            kind, pattern = rule.split(":", 1)
            kind, pattern = kind.strip(), pattern.strip()
        else:
            kind = rule
        specs.append(VarSpec(name=name, required=(kind.lower() == "required"), pattern=pattern or None, comment=comment))
    return specs


def parse_dotenv(text: str) -> Dict[str, str]:
    """Parse a .env file into a dict (KEY=value lines)."""
    env: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        env[key.strip()] = value
    return env


def check_env(specs: List[VarSpec], env: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """Check `env` against `specs`. Returns categorized results."""
    report: Dict[str, List[Dict[str, str]]] = {
        "missing": [],          # required, absent or empty
        "invalid": [],          # present but fails the regex
        "ok": [],               # present and valid
        "optional_missing": [], # optional, absent (informational)
    }
    for spec in specs:
        value = env.get(spec.name)
        present = value is not None and value != ""

        if not present:
            if spec.required:
                report["missing"].append({"name": spec.name, "reason": "required but not set", "hint": spec.comment})
            else:
                report["optional_missing"].append({"name": spec.name, "hint": spec.comment})
            continue

        if spec.pattern is not None and not re.search(spec.pattern, value):
            report["invalid"].append(
                {"name": spec.name, "reason": f"value does not match /{spec.pattern}/", "hint": spec.comment}
            )
            continue

        report["ok"].append({"name": spec.name})
    return report


def is_passing(report: Dict[str, List[Dict[str, str]]]) -> bool:
    """Deploy-gate: pass only when nothing is missing or invalid."""
    return not report["missing"] and not report["invalid"]
