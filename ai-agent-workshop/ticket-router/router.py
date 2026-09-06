from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Ticket:
    ticket_id: str
    subject: str
    body: str


@dataclass
class RouteResult:
    ticket_id: str
    team: str
    confidence: float
    reasoning: str
    matched_keywords: List[str] = field(default_factory=list)


TEAM_KEYWORDS: Dict[str, List[str]] = {
    "billing": ["invoice", "charge", "refund", "payment", "subscription", "billing", "credit card", "receipt", "overcharged"],
    "technical": ["error", "bug", "crash", "not working", "broken", "timeout", "login failed", "500", "exception", "outage"],
    "account": ["password", "reset", "locked out", "email change", "delete account", "username", "two-factor", "profile"],
    "sales": ["pricing", "upgrade", "demo", "quote", "enterprise plan", "trial", "discount", "renew contract"],
}

DEFAULT_TEAM = "general"


def _keyword_score(text: str, keywords: List[str]) -> List[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw in text_lower]


def _rule_based_route(subject: str, body: str) -> tuple[str, float, List[str]]:
    """Weight subject matches 2x body matches, since subject is a stronger signal."""
    best_team = DEFAULT_TEAM
    best_score = 0.0
    best_matches: List[str] = []

    for team, keywords in TEAM_KEYWORDS.items():
        subject_matches = _keyword_score(subject, keywords)
        body_matches = _keyword_score(body, keywords)
        all_matches = list(dict.fromkeys(subject_matches + body_matches))
        score = len(subject_matches) * 2 + len(body_matches)
        if score > best_score:
            best_score = score
            best_team = team
            best_matches = all_matches

    total_possible = max(len(TEAM_KEYWORDS.get(best_team, [])), 1)
    confidence = min(best_score / (total_possible * 1.5), 1.0) if best_score > 0 else 0.2
    return best_team, round(confidence, 2), best_matches


def _call_claude_route(subject: str, body: str) -> Optional[tuple[str, float, str]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        teams = list(TEAM_KEYWORDS.keys()) + [DEFAULT_TEAM]
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Classify this support ticket into exactly one team: {', '.join(teams)}.\n"
                    f"Subject: {subject}\nBody: {body}\n"
                    "Reply in the format: team|confidence(0-1)|one-sentence reasoning"
                ),
            }],
        )
        text = response.content[0].text.strip()
        parts = text.split("|")
        if len(parts) == 3:
            team = parts[0].strip().lower()
            confidence = float(re.sub(r"[^0-9.]", "", parts[1]) or 0.5)
            reasoning = parts[2].strip()
            return team, min(confidence, 1.0), reasoning
    except Exception:
        return None
    return None


def route_ticket(ticket: Ticket, low_confidence_threshold: float = 0.35) -> RouteResult:
    claude_result = _call_claude_route(ticket.subject, ticket.body)
    if claude_result:
        team, confidence, reasoning = claude_result
        matched = []
    else:
        team, confidence, matched = _rule_based_route(ticket.subject, ticket.body)
        if matched:
            reasoning = f"matched keywords: {', '.join(matched)}"
        else:
            reasoning = "no strong keyword match, defaulting to general queue"

    if confidence < low_confidence_threshold:
        return RouteResult(ticket.ticket_id, "human-triage", confidence, f"low confidence ({confidence}), needs human review — {reasoning}", matched)

    return RouteResult(ticket.ticket_id, team, confidence, reasoning, matched)


def route_batch(tickets: List[Ticket], low_confidence_threshold: float = 0.35) -> List[RouteResult]:
    return [route_ticket(t, low_confidence_threshold) for t in tickets]
