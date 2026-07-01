from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompetitorProfile:
    """Structured intel extracted from one competitor's raw text (site copy, blurb, notes)."""

    name: str
    pricing: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    target_market: str = ""
    positioning: str = ""


@dataclass
class IntelReport:
    competitors: list[CompetitorProfile] = field(default_factory=list)
    feature_matrix: dict = field(default_factory=dict)  # feature -> {competitor: bool}
    takeaways: list[str] = field(default_factory=list)


# ------------------------- heuristic extraction -------------------------

_PRICE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?/\s?(?:mo|month|user|yr|year|seat))?", re.I)
_PLAN = re.compile(r"\b(free|freemium|starter|basic|pro|team|business|enterprise|premium)\b", re.I)

# Feature vocabulary we scan every competitor for, so the matrix columns line up.
FEATURE_LEXICON = {
    "API access": re.compile(r"\bAPI\b", re.I),
    "SSO / SAML": re.compile(r"\b(sso|saml|single sign)\b", re.I),
    "Integrations": re.compile(r"\bintegrat", re.I),
    "Analytics / reporting": re.compile(r"\b(analytics|dashboard|reporting|insights)\b", re.I),
    "Mobile app": re.compile(r"\b(mobile app|ios|android)\b", re.I),
    "Automation / workflows": re.compile(r"\b(automat|workflow)\b", re.I),
    "Collaboration": re.compile(r"\b(collaborat|team|shared|comment)\b", re.I),
    "AI features": re.compile(r"\b(ai|machine learning|ml|gpt|llm|copilot)\b", re.I),
    "Security / compliance": re.compile(r"\b(soc 2|gdpr|hipaa|encrypt|compliance)\b", re.I),
    "Free tier": re.compile(r"\b(free tier|free plan|free forever|freemium)\b", re.I),
}

_MARKET = re.compile(
    r"\b(for\s+(?:small|enterprise|mid-market|startups?|smbs?|teams?|developers?|marketers?|"
    r"agencies|freelancers)[^.]{0,40})",
    re.I,
)


def extract_profile(name: str, text: str) -> CompetitorProfile:
    """Parse one competitor's raw text into a structured profile. Deterministic, no LLM."""
    prices = list(dict.fromkeys(_PRICE.findall(text)))
    plans = list(dict.fromkeys(m.group(0).title() for m in _PLAN.finditer(text)))
    pricing = prices + [p for p in plans if p.lower() not in " ".join(prices).lower()]

    features = [feat for feat, pat in FEATURE_LEXICON.items() if pat.search(text)]

    m = _MARKET.search(text)
    target = m.group(1).strip() if m else ""

    # positioning = first sentence, trimmed
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0] if text.strip() else ""
    positioning = first[:160]

    return CompetitorProfile(
        name=name,
        pricing=pricing[:6],
        features=features,
        target_market=target,
        positioning=positioning,
    )


def build_feature_matrix(profiles: list[CompetitorProfile]) -> dict:
    """Feature -> {competitor -> has_it}. Only rows where at least one competitor has it."""
    matrix: dict = {}
    for feat in FEATURE_LEXICON:
        row = {p.name: (feat in p.features) for p in profiles}
        if any(row.values()):
            matrix[feat] = row
    return matrix


def _heuristic_takeaways(profiles: list[CompetitorProfile], matrix: dict) -> list[str]:
    takeaways = []
    # feature gaps: features nobody has
    all_feats = set(FEATURE_LEXICON)
    covered = {f for f, row in matrix.items() if any(row.values())}
    gaps = sorted(all_feats - covered)
    if gaps:
        takeaways.append(f"White space - no tracked competitor offers: {', '.join(gaps)}.")
    # feature leader
    leader = max(profiles, key=lambda p: len(p.features), default=None)
    if leader:
        takeaways.append(f"{leader.name} has the broadest feature set ({len(leader.features)} tracked features).")
    # pricing spread
    priced = [p for p in profiles if p.pricing]
    if priced:
        takeaways.append(f"{len(priced)} of {len(profiles)} competitors publish pricing openly.")
    return takeaways


def heuristic_report(competitors: dict) -> IntelReport:
    """Build a full comparison report from {name: raw_text} with no LLM."""
    profiles = [extract_profile(name, text) for name, text in competitors.items()]
    matrix = build_feature_matrix(profiles)
    takeaways = _heuristic_takeaways(profiles, matrix)
    return IntelReport(competitors=profiles, feature_matrix=matrix, takeaways=takeaways)


# ----------------------------- LLM synthesis -----------------------------

def llm_report(competitors: dict, api_key: str) -> IntelReport:
    """Use Claude to extract profiles and synthesize sharper strategic takeaways."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    blob = "\n\n".join(f"### {name}\n{text}" for name, text in competitors.items())
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a competitive analyst. From the competitor descriptions below, "
                    "extract a structured profile for each and 3-5 strategic takeaways (gaps, "
                    "positioning, pricing patterns). Use ONLY facts stated in the text. "
                    "Respond with ONLY valid JSON, no markdown fences:\n"
                    '{"competitors": [{"name": "...", "pricing": ["..."], "features": ["..."], '
                    '"target_market": "...", "positioning": "..."}], "takeaways": ["..."]}\n\n'
                    f"{blob}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    profiles = [
        CompetitorProfile(
            name=c.get("name", ""),
            pricing=c.get("pricing", []),
            features=c.get("features", []),
            target_market=c.get("target_market", ""),
            positioning=c.get("positioning", ""),
        )
        for c in data.get("competitors", [])
    ]
    return IntelReport(
        competitors=profiles,
        feature_matrix=build_feature_matrix(profiles),
        takeaways=data.get("takeaways", []),
    )


def summarize_competitors(competitors: dict, api_key: Optional[str] = None) -> IntelReport:
    """Summarize a set of competitors into profiles + feature matrix + takeaways.

    Uses Claude if an API key is available, else deterministic extraction."""
    competitors = {k: v for k, v in competitors.items() if v.strip()}
    if not competitors:
        return IntelReport()
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_report(competitors)
    return llm_report(competitors, api_key)


SAMPLE_COMPETITORS = {
    "FlowCRM": (
        "FlowCRM is a CRM for small teams and startups. Plans: Free, Pro at $29/user/mo, "
        "and Business at $59/user/mo. Includes a REST API, Zapier integrations, mobile app "
        "for iOS and Android, and pipeline analytics. SOC 2 compliant."
    ),
    "Salesboard": (
        "Salesboard is an enterprise sales platform for mid-market and enterprise teams. "
        "Pricing is custom / enterprise only. Offers SSO and SAML, advanced reporting "
        "dashboards, workflow automation, and AI-powered lead scoring. GDPR and HIPAA ready."
    ),
    "LeadJar": (
        "LeadJar is a lightweight lead tracker for freelancers and agencies. Free forever "
        "plan plus a Premium tier at $15/mo. Simple contact management with email "
        "integrations. No mobile app yet."
    ),
}
