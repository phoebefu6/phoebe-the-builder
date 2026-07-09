from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The five validation dimensions, each scored 1-5.
DIMENSIONS = [
    "Problem Severity",
    "Market Size",
    "Differentiation",
    "Feasibility",
    "Willingness to Pay",
]

# Cheapest experiment to de-risk each dimension (riskiest-assumption-first).
EXPERIMENTS: Dict[str, str] = {
    "Problem Severity": "Run 10 problem interviews. Ask about the last time they hit this pain and what they did — do NOT pitch. Looking for emotion + existing workarounds.",
    "Market Size": "Bottom-up TAM: (# reachable customers) x (realistic annual price). Cross-check with search-volume and competitor customer counts.",
    "Differentiation": "Build a 1-page comparison vs. the top 3 alternatives (including 'do nothing' and spreadsheets). If you can't name a 10x axis, that's the risk.",
    "Feasibility": "Timebox a 1-day spike on the single hardest technical assumption. Ship a fake/manual 'Wizard of Oz' version before automating.",
    "Willingness to Pay": "Put up a fake-door / pre-sale landing page with a real price and a 'Buy' button. Measure clicks-to-checkout. Ask 5 people for a pre-order or LOI.",
}

# Lean Canvas blocks with a guiding question for the offline scaffold.
LEAN_CANVAS_BLOCKS: List[tuple] = [
    ("Problem", "What are the top 1-3 problems? What do customers do today (existing alternatives)?"),
    ("Customer Segments", "Who exactly has this problem? Who are your early adopters?"),
    ("Unique Value Proposition", "Single, clear, compelling message: why you're different and worth attention."),
    ("Solution", "The smallest set of features that addresses the top problems."),
    ("Channels", "How do you reach customers (free + paid paths to them)?"),
    ("Revenue Streams", "How do you make money? What's the pricing model?"),
    ("Cost Structure", "Fixed and variable costs to run this."),
    ("Key Metrics", "The few numbers that tell you it's working (activation, retention, revenue)."),
    ("Unfair Advantage", "What can't be easily copied or bought?"),
]


@dataclass
class ValidationResult:
    idea: str
    scores: Dict[str, int]
    lean_canvas: Dict[str, str]
    riskiest_assumptions: List[str]
    experiments: List[Dict[str, str]]
    verdict: str
    rationale: str = ""

    @property
    def overall(self) -> float:
        return round(sum(self.scores.values()) / len(self.scores), 2) if self.scores else 0.0


def _verdict(overall: float) -> str:
    if overall >= 4.0:
        return "Strong signal — validate the top risk, then move fast."
    if overall >= 3.0:
        return "Promising but unproven — de-risk before building."
    if overall >= 2.0:
        return "Weak — reframe the problem or segment before spending a day of dev."
    return "Not ready — the riskiest assumptions are unaddressed. Do NOT build yet."


def _riskiest(scores: Dict[str, int], k: int = 2) -> List[str]:
    """Riskiest assumptions = the lowest-scoring dimensions. Validate these FIRST —
    the whole point of Lean Startup is to test what's most likely to kill you."""
    return [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1])[:k]]


def _scaffold_canvas(idea: str) -> Dict[str, str]:
    return {name: f"[{q}]" for name, q in LEAN_CANVAS_BLOCKS}


def _claude_enrich(idea: str, scores: Dict[str, int]) -> Optional[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        blocks = ", ".join(name for name, _ in LEAN_CANVAS_BLOCKS)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=900,
            messages=[{
                "role": "user",
                "content": (
                    "You are a skeptical pre-seed investor. For this startup idea, return STRICT JSON "
                    f'with keys: "lean_canvas" (object with these blocks: {blocks}), '
                    '"scores" (object mapping each of ' + ", ".join(DIMENSIONS) + " to an integer 1-5), "
                    'and "rationale" (2 sentences, blunt). Score honestly; most ideas are 2-3.\n\n'
                    f"Idea: {idea}"
                ),
            }],
        )
        text = resp.content[0].text.strip()
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1]) if start >= 0 else None
    except Exception:
        return None


def validate_idea(
    idea: str,
    scores: Optional[Dict[str, int]] = None,
    use_claude: bool = True,
) -> ValidationResult:
    """Apply the Lean Canvas + riskiest-assumption method to an idea.

    Offline: uses the caller's self-scored dimensions (or a neutral 3) and a canvas scaffold.
    With ANTHROPIC_API_KEY: Claude fills the canvas and scores the idea as a skeptical investor."""
    canvas = _scaffold_canvas(idea)
    rationale = ""
    resolved = {d: 3 for d in DIMENSIONS}
    if scores:
        resolved.update({d: int(scores[d]) for d in DIMENSIONS if d in scores})

    if use_claude:
        enriched = _claude_enrich(idea, resolved)
        if enriched:
            canvas = {name: enriched.get("lean_canvas", {}).get(name, canvas[name]) for name, _ in LEAN_CANVAS_BLOCKS}
            for d in DIMENSIONS:
                v = enriched.get("scores", {}).get(d)
                if isinstance(v, (int, float)):
                    resolved[d] = max(1, min(5, int(v)))
            rationale = enriched.get("rationale", "")

    resolved = {d: max(1, min(5, resolved[d])) for d in DIMENSIONS}
    risky = _riskiest(resolved)
    experiments = [{"assumption": d, "experiment": EXPERIMENTS[d]} for d in risky]
    overall = sum(resolved.values()) / len(resolved)

    return ValidationResult(
        idea=idea,
        scores=resolved,
        lean_canvas=canvas,
        riskiest_assumptions=risky,
        experiments=experiments,
        verdict=_verdict(overall),
        rationale=rationale,
    )


SAMPLE_IDEA = (
    "A subscription app that uses AI to generate personalized weekly meal plans for busy "
    "parents based on their kids' allergies, then auto-orders the groceries."
)

SAMPLE_SCORES = {
    "Problem Severity": 4,
    "Market Size": 4,
    "Differentiation": 2,
    "Feasibility": 3,
    "Willingness to Pay": 2,
}
