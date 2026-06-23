from __future__ import annotations

"""The audit log - an append-only record of "who did what, when".

This is the single most valuable thing to an enterprise buyer: provable evidence
of every login and every access to a dashboard, dataset, model, or LLM. The key
property is **append-only / immutable** - we only ever add lines, never edit or
delete them. That's what makes it trustworthy to an auditor.

Implementation: one JSON object per line (JSONL) in `audit_log.jsonl`. Simple,
greppable, and easy to ship to a real log store (S3, Datadog, a SIEM) later
without changing the callers - they only use `log_event` and `read_events`.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

AUDIT_FILE = Path(__file__).parent / "audit_log.jsonl"


def log_event(actor: str, action: str, *, target: str = "", status: str = "ok",
              role: str = "") -> Dict[str, str]:
    """Append one event. Returns the record written.

    actor  - who (email)         action - what (login/open_app/logout/...)
    target - on what (app slug)  status - outcome (success/denied/granted/...)
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "actor": actor,
        "role": role,
        "action": action,
        "target": target,
        "status": status,
    }
    # Append-only: open in 'a' mode, never truncate, never rewrite.
    with AUDIT_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def read_events(limit: int = 100, actor: Optional[str] = None,
                action: Optional[str] = None) -> List[Dict[str, str]]:
    """Return the most recent events (newest first), optionally filtered."""
    if not AUDIT_FILE.exists():
        return []
    events: List[Dict[str, str]] = []
    for line in AUDIT_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if actor and ev.get("actor") != actor:
            continue
        if action and ev.get("action") != action:
            continue
        events.append(ev)
    events.reverse()  # newest first
    return events[:limit]


def summary() -> Dict[str, int]:
    """Quick counts for a dashboard - total events, denials, distinct actors."""
    events = read_events(limit=10_000)
    return {
        "total": len(events),
        "denied": sum(1 for e in events if e.get("status") == "denied"),
        "actors": len({e.get("actor") for e in events}),
    }
