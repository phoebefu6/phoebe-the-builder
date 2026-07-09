from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

# Standard RICE impact scale (Intercom convention).
IMPACT_SCALE = {"massive": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5, "minimal": 0.25}


@dataclass
class Feature:
    name: str
    reach: float          # # users/events affected per time period
    impact: float         # one of IMPACT_SCALE values (3 / 2 / 1 / 0.5 / 0.25)
    confidence: float     # percent, 0-100
    effort: float         # person-months (>0)


@dataclass
class Scored:
    name: str
    rice: float
    reach: float
    impact: float
    confidence: float
    effort: float
    rank: int = 0
    quadrant: str = ""


def rice_score(f: Feature) -> float:
    """RICE = (Reach x Impact x Confidence%) / Effort. Higher = do sooner."""
    if f.effort <= 0:
        return 0.0
    return (f.reach * f.impact * (f.confidence / 100.0)) / f.effort


def _quadrant(value: float, effort: float, val_med: float, eff_med: float) -> str:
    """Value/effort quadrant. Value proxy = Reach x Impact x Confidence% (RICE numerator)."""
    high_val, low_eff = value >= val_med, effort < eff_med
    if high_val and low_eff:
        return "Quick win"
    if high_val and not low_eff:
        return "Big bet"
    if not high_val and low_eff:
        return "Fill-in"
    return "Time sink"


def prioritize(features: List[Feature]) -> List[Scored]:
    if not features:
        return []
    values = [f.reach * f.impact * (f.confidence / 100.0) for f in features]
    efforts = [f.effort for f in features]
    val_med = statistics.median(values)
    eff_med = statistics.median(efforts)

    scored = [
        Scored(
            name=f.name, rice=round(rice_score(f), 2), reach=f.reach, impact=f.impact,
            confidence=f.confidence, effort=f.effort,
            quadrant=_quadrant(v, f.effort, val_med, eff_med),
        )
        for f, v in zip(features, values)
    ]
    scored.sort(key=lambda s: s.rice, reverse=True)
    for i, s in enumerate(scored, 1):
        s.rank = i
    return scored


def to_frame(scored: List[Scored]) -> pd.DataFrame:
    return pd.DataFrame([
        {"rank": s.rank, "feature": s.name, "RICE": s.rice, "quadrant": s.quadrant,
         "reach": s.reach, "impact": s.impact, "confidence": s.confidence, "effort": s.effort}
        for s in scored
    ])


def estimate_rice(description: str) -> Optional[Dict[str, float]]:
    """Optional: ask Claude to estimate RICE inputs from a feature description.
    Returns {reach, impact, confidence, effort} or None if no API key / failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "Estimate RICE inputs for this feature. Return STRICT JSON with keys "
                    "reach (users/quarter, integer), impact (one of 3,2,1,0.5,0.25), "
                    "confidence (percent 0-100), effort (person-months, number). "
                    "Be conservative on confidence.\n\nFeature: " + description
                ),
            }],
        )
        text = resp.content[0].text
        return json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return None


SAMPLE_FEATURES: List[Feature] = [
    Feature("Dark mode", reach=8000, impact=1.0, confidence=90, effort=2),
    Feature("SSO / SAML login", reach=1500, impact=2.0, confidence=80, effort=3),
    Feature("Mobile app", reach=12000, impact=3.0, confidence=60, effort=12),
    Feature("CSV export", reach=3000, impact=1.0, confidence=100, effort=1),
    Feature("AI recommendations", reach=10000, impact=2.0, confidence=50, effort=8),
    Feature("Two-factor auth", reach=5000, impact=2.0, confidence=90, effort=2),
    Feature("Custom themes", reach=2000, impact=0.5, confidence=80, effort=4),
    Feature("Slack integration", reach=4000, impact=1.0, confidence=85, effort=2),
]
