from __future__ import annotations

# LLM Model Router - core logic.
#
# Sending every request to your biggest model is simple and expensive. Most
# traffic is easy - short classifications, extractions, formatting - and a
# small, cheap model handles it fine. Only a minority genuinely needs the
# frontier model. This router scores each request's complexity from cheap
# signals (length, reasoning/code cues, output-structure needs) and sends it
# to the smallest tier that clears the bar, then reports the cost saved versus
# always-large routing.
#
# The scoring is transparent and rule-based so you can see *why* a request was
# routed where it was - and tune the thresholds to your own quality bar. Prices
# are illustrative per-1M-token rates; edit TIERS for your providers.
#
# Fully offline - no API keys.
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Model tiers (illustrative USD per 1M tokens, blended in/out)
# --------------------------------------------------------------------------- #

@dataclass
class Tier:
    name: str
    price_per_mtok: float   # USD per 1M tokens (blended)
    max_score: float        # route here if complexity <= this


TIERS: List[Tier] = [
    Tier("small", 0.30, 0.34),    # e.g. Haiku-class
    Tier("medium", 3.00, 0.67),   # e.g. Sonnet-class
    Tier("large", 15.00, 1.01),   # e.g. Opus-class
]

# Stems (no trailing \b) so inflections match: explain/explaining,
# reason/reasoning, architect/architectures, trade-off/trade-offs.
_REASON_CUES = re.compile(
    r"\b(?:why|explain|analy[sz]|reason|prove|deriv|strateg|trade[- ]?off|"
    r"compar|evaluat|debug|design|architect|recommend|refactor|optimi[sz]|"
    r"step[- ]by[- ]step|root cause)\w*", re.I)
_CODE_CUES = re.compile(
    r"\b(?:code|function|regex|sql|python|javascript|algorithm|stack trace|"
    r"unit test|typescript|api)\b|```", re.I)
_STRUCT_CUES = re.compile(r"\b(?:json|schema|table|csv|markdown|yaml|xml)\b", re.I)
_SIMPLE_CUES = re.compile(
    r"\b(?:classif|categori[sz]|label|extract|sentiment|yes/no|translat|"
    r"tag|detect|spellcheck)\w*", re.I)


@dataclass
class Routed:
    request: str
    score: float
    tier: str
    reasons: List[str]
    est_tokens: int


def complexity_score(text: str) -> Tuple[float, List[str]]:
    """0..1 complexity from transparent, cheap signals. Returns (score, why)."""
    reasons: List[str] = []
    score = 0.0

    words = len(text.split())
    if words > 120:
        score += 0.35
        reasons.append(f"long input ({words} words)")
    elif words > 40:
        score += 0.18
        reasons.append(f"medium input ({words} words)")

    n_reason = len(_REASON_CUES.findall(text))
    if n_reason:
        score += min(0.40, 0.20 * n_reason)
        reasons.append(f"reasoning cues x{n_reason}")

    if _CODE_CUES.search(text):
        score += 0.30
        reasons.append("code/technical content")

    if _STRUCT_CUES.search(text):
        score += 0.10
        reasons.append("structured output")

    # Multi-question / multi-step requests are harder.
    n_q = text.count("?")
    if n_q >= 2:
        score += 0.12
        reasons.append(f"multiple questions ({n_q})")

    if _SIMPLE_CUES.search(text) and words <= 40 and not _CODE_CUES.search(text):
        score -= 0.20
        reasons.append("simple task cue")

    score = round(max(0.0, min(1.0, score)), 3)
    if not reasons:
        reasons.append("short, generic request")
    return score, reasons


def route(text: str, est_tokens: int = 800) -> Routed:
    """Route a request to the smallest tier whose bar it clears."""
    score, reasons = complexity_score(text)
    chosen = TIERS[-1]
    for t in TIERS:
        if score <= t.max_score:
            chosen = t
            break
    return Routed(text, score, chosen.name, reasons, est_tokens)


def _price(tier_name: str) -> float:
    return next(t.price_per_mtok for t in TIERS if t.name == tier_name)


def cost_of(routed: Routed, tier_name: str) -> float:
    """USD to serve this request on a given tier."""
    return routed.est_tokens / 1_000_000 * _price(tier_name)


# --------------------------------------------------------------------------- #
# Sample traffic - a realistic mix skewed toward easy requests
# --------------------------------------------------------------------------- #

SAMPLE_TRAFFIC: List[Tuple[str, int]] = [
    ("Classify this review sentiment as positive, negative, or neutral: 'Love it!'", 300),
    ("Extract the invoice number and total from this line.", 250),
    ("Translate 'good morning' to Spanish.", 120),
    ("Tag this ticket with one of: billing, login, shipping.", 200),
    ("Summarize this paragraph in one sentence.", 400),
    ("What is the capital of France?", 80),
    ("Return the order status as JSON with fields id and state.", 260),
    ("Explain step-by-step why this SQL query returns duplicate rows and how to fix it: "
     "SELECT * FROM orders o JOIN items i ON o.id = i.order_id.", 1400),
    ("Design a retry-with-backoff strategy for our ingestion pipeline and explain the "
     "trade-offs versus a dead-letter queue.", 1600),
    ("Debug this Python stack trace and propose a fix with a unit test.", 1500),
    ("Compare three architectures for a real-time feature store and recommend one, "
     "explaining the reasoning and cost trade-offs for our scale.", 1800),
    ("Refactor this function for readability and analyze the time complexity.", 1200),
    ("Is this email spam? Answer yes or no.", 150),
    ("Categorize this support message into a product area.", 220),
    ("Write and explain a regex that validates E.164 phone numbers, with edge cases.", 1300),
]


def run_traffic() -> Dict[str, object]:
    """Route all sample requests; compare cost vs always-large."""
    rows = []
    routed_cost = 0.0
    always_large_cost = 0.0
    tier_counts = {t.name: 0 for t in TIERS}

    for text, toks in SAMPLE_TRAFFIC:
        r = route(text, toks)
        rc = cost_of(r, r.tier)
        lc = cost_of(r, "large")
        routed_cost += rc
        always_large_cost += lc
        tier_counts[r.tier] += 1
        rows.append({
            "request": text[:60] + ("..." if len(text) > 60 else ""),
            "score": r.score,
            "tier": r.tier,
            "why": "; ".join(r.reasons),
            "cost": round(rc, 6),
        })

    saved = always_large_cost - routed_cost
    pct = round(100 * saved / always_large_cost, 1) if always_large_cost else 0.0
    return {
        "rows": rows,
        "tier_counts": tier_counts,
        "routed_cost": round(routed_cost, 6),
        "always_large_cost": round(always_large_cost, 6),
        "saved": round(saved, 6),
        "saved_pct": pct,
    }


if __name__ == "__main__":
    res = run_traffic()
    print(f"Routed {len(SAMPLE_TRAFFIC)} requests\n")
    for row in res["rows"]:
        print(f"[{row['tier']:<6}] score={row['score']:<5} {row['request']}")
    print()
    print("Tier distribution:", res["tier_counts"])
    print(f"Always-large cost: ${res['always_large_cost']:.4f}")
    print(f"Routed cost:       ${res['routed_cost']:.4f}")
    print(f"Saved:             ${res['saved']:.4f} ({res['saved_pct']}%)")
