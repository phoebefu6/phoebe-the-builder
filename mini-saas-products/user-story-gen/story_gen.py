from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Feature type detection -------------------------------------------------

FEATURE_KEYWORDS: Dict[str, List[str]] = {
    "auth": ["log in", "login", "sign up", "signup", "password", "sso", "register", "2fa", "authenticate"],
    "search": ["search", "filter", "find", "look up", "sort", "browse"],
    "notification": ["notify", "notification", "alert", "email me", "remind", "push", "digest"],
    "report": ["report", "dashboard", "export", "chart", "analytics", "download", "summary"],
    "upload": ["upload", "import", "attach", "drag and drop", "bulk load"],
    "integration": ["integrate", "sync", "webhook", "connect to", "api access", "slack", "salesforce"],
    "crud": ["create", "add", "edit", "update", "delete", "manage", "save", "archive"],
}

DEFAULT_PERSONA: Dict[str, str] = {
    "auth": "returning user",
    "search": "power user",
    "notification": "busy team member",
    "report": "team lead",
    "upload": "data owner",
    "integration": "ops engineer",
    "crud": "everyday user",
    "general": "user",
}

# --- Acceptance criteria templates (Given / When / Then) ---------------------

AC_TEMPLATES: Dict[str, List[str]] = {
    "auth": [
        "Given valid credentials, when the user submits the form, then they land on their home screen within 2 seconds",
        "Given invalid credentials, when the user submits the form, then a clear error shows without revealing which field was wrong",
        "Given 5 failed attempts, when the user tries again, then the account is temporarily locked and a reset path is offered",
    ],
    "search": [
        "Given a matching term, when the user searches, then results appear ranked by relevance within 1 second",
        "Given no matches, when the user searches, then an empty state suggests alternative terms",
        "Given active filters, when the user refines the query, then filters persist and the result count updates",
    ],
    "notification": [
        "Given the trigger event occurs, when conditions are met, then the notification is delivered within 1 minute",
        "Given notifications are muted, when the trigger fires, then nothing is sent and the event is still logged",
        "Given a notification is received, when the user clicks it, then they deep-link to the relevant item",
    ],
    "report": [
        "Given data exists for the selected range, when the user opens the report, then totals match the source system",
        "Given the report is on screen, when the user exports it, then the file downloads in the chosen format with all visible rows",
        "Given no data for the range, when the report loads, then an empty state explains why and suggests a wider range",
    ],
    "upload": [
        "Given a valid file, when the user uploads it, then a success message confirms rows/items processed",
        "Given an invalid or oversized file, when the user uploads it, then a specific validation error names the problem",
        "Given a slow connection, when the upload is in progress, then a progress indicator shows and cancel is available",
    ],
    "integration": [
        "Given valid credentials for the external system, when the user connects, then a test call confirms the link",
        "Given the connection drops, when a sync runs, then the failure is retried and surfaced in a status log",
        "Given a successful sync, when the user checks either system, then records match on both sides",
    ],
    "crud": [
        "Given valid input, when the user saves, then the item appears in the list immediately",
        "Given required fields are missing, when the user saves, then inline errors mark each missing field",
        "Given a delete action, when the user confirms, then the item is removed and an undo option is offered briefly",
    ],
    "general": [
        "Given the feature is available, when the user performs the main action, then the expected outcome is visible",
        "Given invalid input or state, when the user tries the action, then a clear error explains what to fix",
        "Given the action succeeds, when the user returns later, then the result has persisted",
    ],
}

DEFAULT_BENEFIT: Dict[str, str] = {
    "auth": "I can access my account securely without friction",
    "search": "I can find what I need without scanning everything manually",
    "notification": "I never miss something that needs my attention",
    "report": "I can make decisions from numbers instead of gut feel",
    "upload": "I can get my data in without manual re-entry",
    "integration": "my tools stay in sync without copy-paste",
    "crud": "I can keep my information accurate and up to date",
    "general": "I can get my job done faster",
}

VAGUE_WORDS = ["fast", "easy", "better", "user-friendly", "intuitive", "nice", "modern", "seamless", "simple"]
IMPLEMENTATION_WORDS = ["button", "dropdown", "database", "endpoint", "modal", "postgres", "react", "table schema"]
SCOPE_WORDS = ["all ", "everything", "entire", "every ", "any kind"]


@dataclass
class UserStory:
    raw: str
    persona: str
    capability: str
    benefit: str
    feature_type: str
    acceptance_criteria: List[str] = field(default_factory=list)
    invest_score: int = 0
    invest_flags: List[str] = field(default_factory=list)

    @property
    def story(self) -> str:
        return f"As a {self.persona}, I want to {self.capability}, so that {self.benefit}."


def detect_feature_type(text: str) -> str:
    lowered = text.lower()
    for ftype, keywords in FEATURE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return ftype
    return "general"


def parse_idea(text: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Split a raw idea into (persona, capability, benefit). Persona/benefit may be absent."""
    match = re.match(
        r"as an? (?P<persona>[^,]+),?\s*i want (?:to )?(?P<cap>.+?)(?:,?\s*so that (?P<benefit>.+))?$",
        text.strip().rstrip("."),
        re.IGNORECASE,
    )
    if match:
        return match.group("persona").strip(), match.group("cap").strip(), (
            match.group("benefit").strip() if match.group("benefit") else None
        )
    capability = re.sub(r"^(i want to|we need to|add|build|implement|users? should be able to)\s+", "",
                        text.strip().rstrip("."), flags=re.IGNORECASE)
    return None, capability.strip(), None


def score_invest(story: UserStory) -> Tuple[int, List[str]]:
    """Score a story against INVEST. Each of 6 checks worth up to ~17 points."""
    flags: List[str] = []
    lowered = story.capability.lower()
    score = 0

    if " and " not in lowered and ";" not in lowered:
        score += 17
    else:
        flags.append("Independent: capability bundles multiple things — split at the 'and'")

    if not any(w in lowered for w in IMPLEMENTATION_WORDS):
        score += 17
    else:
        flags.append("Negotiable: describes the implementation, not the need — state the outcome instead")

    if story.benefit and story.benefit not in DEFAULT_BENEFIT.values():
        score += 17
    else:
        flags.append("Valuable: no explicit benefit given — add a real 'so that' (a default was used)")

    if story.feature_type != "general" and len(lowered.split()) >= 3:
        score += 17
    else:
        flags.append("Estimable: too vague to size — name the concrete capability")

    if len(lowered.split()) <= 14 and not any(w in lowered for w in SCOPE_WORDS):
        score += 16
    else:
        flags.append("Small: scope sounds big — carve out the thinnest useful slice")

    if not any(w in lowered for w in VAGUE_WORDS):
        score += 16
    else:
        flags.append("Testable: vague quality words ('easy', 'fast') — replace with a measurable check")

    return score, flags


def _normalize_capability(capability: str) -> str:
    cap = capability[0].lower() + capability[1:] if capability else capability
    cap = re.sub(r"\btheir\b", "my", cap)
    cap = re.sub(r"\bthem\b", "me", cap)
    return re.sub(r"^to\s+", "", cap)


def generate_story(idea: str, persona: str = "") -> UserStory:
    parsed_persona, capability, parsed_benefit = parse_idea(idea)
    capability = _normalize_capability(capability)
    ftype = detect_feature_type(idea)
    final_persona = persona.strip() or parsed_persona or DEFAULT_PERSONA[ftype]
    benefit = parsed_benefit or DEFAULT_BENEFIT[ftype]
    story = UserStory(
        raw=idea.strip(),
        persona=final_persona,
        capability=capability,
        benefit=benefit,
        feature_type=ftype,
        acceptance_criteria=list(AC_TEMPLATES[ftype]),
    )
    story.invest_score, story.invest_flags = score_invest(story)
    return story


def generate_backlog(ideas_text: str, persona: str = "") -> List[UserStory]:
    lines = [ln.strip("-• \t") for ln in ideas_text.strip().splitlines() if ln.strip()]
    return [generate_story(line, persona) for line in lines]


def backlog_to_markdown(stories: List[UserStory]) -> str:
    parts: List[str] = ["# Generated User Stories\n"]
    for i, s in enumerate(stories, 1):
        parts.append(f"## Story {i}: {s.capability.capitalize()}")
        parts.append(f"**{s.story}**  \n*Type: {s.feature_type} · INVEST score: {s.invest_score}/100*\n")
        parts.append("**Acceptance criteria:**")
        parts.extend(f"- {ac}" for ac in s.acceptance_criteria)
        if s.invest_flags:
            parts.append("\n**Improve this story:**")
            parts.extend(f"- {f}" for f in s.invest_flags)
        parts.append("")
    return "\n".join(parts)
