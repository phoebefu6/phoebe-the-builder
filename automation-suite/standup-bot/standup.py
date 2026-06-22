from __future__ import annotations

"""Core logic: collect daily standup updates and roll them into one digest.

Standups scatter across Slack threads, DMs, and memory. Here each person submits
three fields - yesterday / today / blockers - and the bot aggregates the day into
a single clean digest, surfacing blockers up top so nothing gets lost.

Two digest paths:
  - `build_digest`         deterministic Markdown, always works, no API.
  - `summarize_with_claude` optional narrative summary when ANTHROPIC_API_KEY is set
                            (SDK imported lazily; falls back to the template).

Pure logic + a tiny in-memory store, so the notebook and the FastAPI service share it.
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

CLAUDE_MODEL = "claude-opus-4-8"


@dataclass
class Update:
    name: str
    yesterday: str = ""
    today: str = ""
    blockers: str = ""

    @property
    def has_blocker(self) -> bool:
        b = self.blockers.strip().lower()
        return bool(b) and b not in {"none", "no", "n/a", "-", "nothing"}


@dataclass
class StandupStore:
    """In-memory store keyed by ISO date. Swap for a DB later behind these methods."""
    by_date: Dict[str, List[Update]] = field(default_factory=lambda: defaultdict(list))

    def submit(self, update: Update, day: Optional[str] = None) -> None:
        day = day or date.today().isoformat()
        # One update per person per day - re-submitting replaces.
        updates = self.by_date[day]
        for i, u in enumerate(updates):
            if u.name.strip().lower() == update.name.strip().lower():
                updates[i] = update
                return
        updates.append(update)

    def get(self, day: Optional[str] = None) -> List[Update]:
        return list(self.by_date.get(day or date.today().isoformat(), []))


def build_digest(updates: List[Update], day: Optional[str] = None) -> str:
    """Deterministic Markdown digest. Blockers surfaced at the top."""
    day = day or date.today().isoformat()
    if not updates:
        return f"# Daily Standup - {day}\n\n_No updates submitted yet._\n"

    blocked = [u for u in updates if u.has_blocker]
    lines: List[str] = [f"# Daily Standup - {day}", "",
                        f"**{len(updates)} update(s)** · **{len(blocked)} blocker(s)**", ""]

    if blocked:
        lines.append("## 🚧 Blockers (action needed)")
        for u in blocked:
            lines.append(f"- **{u.name}**: {u.blockers.strip()}")
        lines.append("")

    lines.append("## Updates")
    for u in updates:
        lines.append(f"### {u.name}")
        if u.yesterday.strip():
            lines.append(f"- *Yesterday:* {u.yesterday.strip()}")
        if u.today.strip():
            lines.append(f"- *Today:* {u.today.strip()}")
        lines.append(f"- *Blockers:* {u.blockers.strip() if u.has_blocker else 'none'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_prompt(updates: List[Update], day: str) -> str:
    body = "\n".join(
        f"- {u.name}: yesterday={u.yesterday!r}; today={u.today!r}; blockers={u.blockers!r}"
        for u in updates
    )
    return (
        f"Summarize this engineering standup for {day} in concise Markdown. "
        "Lead with a one-line team summary, then a Blockers section (who is blocked on "
        "what, most urgent first), then Themes (what the team is collectively focused on). "
        "Be brief and factual.\n\n"
        f"Updates:\n{body}"
    )


def summarize_with_claude(updates: List[Update], day: Optional[str] = None,
                          api_key: Optional[str] = None) -> str:
    """Narrative digest via Claude. Falls back to the template if no key/SDK."""
    day = day or date.today().isoformat()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key or not updates:
        return build_digest(updates, day)
    try:
        from anthropic import Anthropic
    except ImportError:
        return build_digest(updates, day)

    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=800,
        messages=[{"role": "user", "content": _build_prompt(updates, day)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
