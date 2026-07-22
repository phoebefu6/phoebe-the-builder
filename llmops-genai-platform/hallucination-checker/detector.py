from __future__ import annotations

# Hallucination Detector - core logic.
#
# A RAG system retrieves context, then the LLM writes an answer. The failure
# mode everyone fears: the answer states something the retrieved context does
# NOT support - a confident hallucination. This module scores *groundedness*:
# it splits the answer into claims, and for each claim measures how well it is
# supported by the source context. Unsupported claims get flagged so you can
# block, warn, or force a regeneration before the answer reaches a user.
#
# The support signal here is a dependency-free lexical one (token overlap +
# cosine over content words, with a numeric-consistency check). It is NOT a
# real NLI model - swap in an entailment model or an LLM judge for production -
# but it reliably catches the common case: a sentence whose key terms and
# numbers never appear in the context.
#
# Fully offline - no API keys.

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "for", "and", "or", "in", "on", "at", "by", "with", "as", "it", "its",
    "this", "that", "these", "those", "from", "has", "have", "had", "will",
    "can", "may", "also", "which", "their", "they", "you", "your",
}


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def split_claims(answer: str) -> List[str]:
    """Split an answer into sentence-level claims."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]


def _cosine(a: Dict[str, int], b: Dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _bag(tokens: List[str]) -> Dict[str, int]:
    bag: Dict[str, int] = {}
    for t in tokens:
        bag[t] = bag.get(t, 0) + 1
    return bag


@dataclass
class ClaimResult:
    claim: str
    support: float          # 0..1 lexical support from context
    grounded: bool
    numbers_ok: bool        # every number in the claim appears in context
    reason: str


@dataclass
class AnswerResult:
    groundedness: float          # mean support across claims, 0..1
    grounded: bool               # all claims meet threshold
    claims: List[ClaimResult] = field(default_factory=list)

    @property
    def n_flagged(self) -> int:
        return sum(1 for c in self.claims if not c.grounded)


def score_claim(claim: str, context: str, threshold: float) -> ClaimResult:
    """Support = max cosine of the claim against any context sentence,
    blended with content-word coverage. Numbers must also be present."""
    ctx_sentences = split_claims(context) or [context]
    claim_tokens = _tokens(claim)
    claim_bag = _bag(claim_tokens)

    best = 0.0
    for cs in ctx_sentences:
        best = max(best, _cosine(claim_bag, _bag(_tokens(cs))))

    # Content-word coverage against the whole context (catches spread-out support).
    ctx_vocab = set(_tokens(context))
    coverage = (sum(1 for t in set(claim_tokens) if t in ctx_vocab) /
                len(set(claim_tokens))) if claim_tokens else 1.0
    support = round(max(best, 0.6 * coverage + 0.4 * best), 3)

    # Numeric consistency: any number in the claim must appear in the context.
    claim_nums = set(_NUM_RE.findall(claim))
    ctx_nums = set(_NUM_RE.findall(context))
    numbers_ok = claim_nums.issubset(ctx_nums)

    grounded = support >= threshold and numbers_ok
    if not numbers_ok:
        reason = "number(s) not found in context: " + ", ".join(
            sorted(claim_nums - ctx_nums))
    elif support >= threshold:
        reason = "supported by context"
    else:
        reason = f"weak support ({support:.2f} < {threshold:.2f})"
    return ClaimResult(claim, support, grounded, numbers_ok, reason)


def check_answer(answer: str, context: str, threshold: float = 0.35) -> AnswerResult:
    """Score every claim in the answer against the context."""
    claims = [score_claim(c, context, threshold) for c in split_claims(answer)]
    mean = round(sum(c.support for c in claims) / len(claims), 3) if claims else 0.0
    return AnswerResult(groundedness=mean,
                        grounded=all(c.grounded for c in claims),
                        claims=claims)


# --------------------------------------------------------------------------- #
# Sample data - RAG (context, answer) pairs, some with a planted hallucination
# --------------------------------------------------------------------------- #

SAMPLES: List[Dict[str, str]] = [
    {
        "label": "grounded",
        "question": "What is the refund window and who approves it?",
        "context": ("Customers may request a refund within 30 days of purchase. "
                    "Refunds are approved by the billing team. Approved refunds "
                    "are returned to the original payment method within 5 business days."),
        "answer": ("Customers can request a refund within 30 days. The billing "
                   "team approves refunds, and the money returns to the original "
                   "payment method within 5 business days."),
    },
    {
        "label": "wrong number",
        "question": "What is the refund window?",
        "context": ("Customers may request a refund within 30 days of purchase. "
                    "Refunds are approved by the billing team."),
        "answer": ("Customers can request a refund within 60 days of purchase, "
                   "approved by the billing team."),
    },
    {
        "label": "unsupported claim",
        "question": "Summarize the security posture.",
        "context": ("All data is encrypted at rest using AES-256. Access requires "
                    "multi-factor authentication. Audit logs are retained for one year."),
        "answer": ("Data is encrypted at rest with AES-256 and access needs MFA. "
                   "The company is also SOC 2 Type II certified and stores data only "
                   "in EU data centers."),
    },
    {
        "label": "fabricated entity",
        "question": "Who is the CEO?",
        "context": ("Acme Corp was founded in 2011 and makes warehouse robots. "
                    "It employs 400 people across three sites."),
        "answer": ("Acme Corp, founded in 2011, makes warehouse robots and is led "
                   "by CEO Jane Fairbanks."),
    },
]


def run_samples(threshold: float = 0.35) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for s in SAMPLES:
        res = check_answer(s["answer"], s["context"], threshold)
        rows.append({
            "label": s["label"],
            "groundedness": res.groundedness,
            "grounded": res.grounded,
            "flagged": res.n_flagged,
            "flagged_claims": [c.claim for c in res.claims if not c.grounded],
        })
    return rows


if __name__ == "__main__":
    th = 0.35
    print(f"Hallucination check (threshold={th})\n")
    for s in SAMPLES:
        res = check_answer(s["answer"], s["context"], th)
        tag = "GROUNDED" if res.grounded else "FLAGGED "
        print(f"[{tag}] {s['label']:<18} groundedness={res.groundedness:.2f}")
        for c in res.claims:
            if not c.grounded:
                print(f"          ! {c.claim}")
                print(f"            -> {c.reason}")
    print()
