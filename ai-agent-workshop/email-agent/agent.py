from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Critique:
    passed: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class DraftResult:
    final_draft: str
    iterations: int
    history: List[str] = field(default_factory=list)
    critiques: List[Critique] = field(default_factory=list)


TONE_OPENERS = {
    "friendly": "Hi {name},\n\nThanks for reaching out!",
    "formal": "Dear {name},\n\nThank you for your message.",
    "brief": "Hi {name},",
}

TONE_CLOSERS = {
    "friendly": "Let me know if anything else comes up.\n\nBest,\n{sender}",
    "formal": "Please let me know if you require further information.\n\nKind regards,\n{sender}",
    "brief": "Thanks,\n{sender}",
}


def _extract_name(email_from: str) -> str:
    match = re.match(r"^([A-Za-z]+)", email_from.strip())
    return match.group(1) if match else "there"


def _extract_action_items(body: str) -> List[str]:
    """Pull out sentences that look like questions or requests."""
    sentences = re.split(r"(?<=[.?!])\s+", body.strip())
    items = [s.strip() for s in sentences if "?" in s or re.search(r"\b(please|need|can you|could you)\b", s, re.I)]
    return items or sentences[:1]


def _mock_llm_draft(subject: str, sender_name: str, body: str, tone: str, my_name: str) -> str:
    """Template-based draft generator, used when no ANTHROPIC_API_KEY is set."""
    action_items = _extract_action_items(body)
    opener = TONE_OPENERS.get(tone, TONE_OPENERS["friendly"]).format(name=sender_name)
    closer = TONE_CLOSERS.get(tone, TONE_CLOSERS["friendly"]).format(sender=my_name)

    body_lines = []
    for item in action_items:
        body_lines.append(f"Re: \"{item.strip()}\" — yes, I can confirm this and will follow up with details shortly.")

    return f"{opener}\n\n" + "\n".join(body_lines) + f"\n\n{closer}"


def _mock_llm_critique(draft: str, subject: str, body: str) -> Critique:
    """Rule-based self-critique standing in for an LLM judge call."""
    issues = []
    if len(draft.split()) < 15:
        issues.append("draft too short — doesn't fully address the email")
    if subject.lower() not in draft.lower() and len(draft.split()) < 20:
        issues.append("draft may not reference the topic clearly")
    if "?" in body and "confirm" not in draft.lower() and "yes" not in draft.lower() and "no" not in draft.lower():
        issues.append("original email asked a question that isn't clearly answered")
    return Critique(passed=len(issues) == 0, issues=issues)


def _call_claude(prompt: str) -> Optional[str]:
    """Real LLM call via Anthropic SDK, only used when ANTHROPIC_API_KEY is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        return None


def draft_email_reply(
    subject: str,
    email_from: str,
    body: str,
    tone: str = "friendly",
    my_name: str = "Phoebe",
    max_iterations: int = 3,
) -> DraftResult:
    """Agent loop: draft -> self-critique -> revise, until critique passes or max_iterations hit."""
    sender_name = _extract_name(email_from)
    history: List[str] = []
    critiques: List[Critique] = []

    draft = _call_claude(
        f"Draft a {tone} reply from {my_name} to {sender_name} about subject '{subject}'. Original email:\n{body}"
    ) or _mock_llm_draft(subject, sender_name, body, tone, my_name)
    history.append(draft)

    for i in range(max_iterations):
        critique = _mock_llm_critique(draft, subject, body)
        critiques.append(critique)
        if critique.passed:
            break
        # revise: append a clarifying line addressing the flagged issue
        fix_line = "\n\n(P.S. To directly answer your question — yes, that works on my end.)"
        draft = draft + fix_line
        history.append(draft)

    return DraftResult(final_draft=draft, iterations=len(history), history=history, critiques=critiques)
