from __future__ import annotations

# Few-Shot Example Selector - core logic.
#
# Most prompts ship a *static* block of few-shot examples: the same 3 examples
# for every input. That underperforms, because the best examples for "where is
# my order" are different from the best examples for "reset my password".
# Dynamic selection fixes this: embed the incoming query, retrieve the k
# nearest labeled examples from a pool, and put *those* in the prompt.
#
# This module implements the selector with a dependency-free lexical embedder
# (bag-of-words cosine) so it runs offline, and quantifies the win: compared to
# random example selection, nearest-example selection puts more same-intent
# examples in front of the model, which raises a k-NN label-match proxy for
# downstream accuracy. Swap in real embeddings for production.
#
# Fully offline - no API keys.
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "do", "does", "i", "how", "can", "to",
    "of", "for", "my", "me", "you", "your", "what", "with", "on", "in", "it",
    "please", "need", "want", "get", "have", "am",
}


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    for t in _TOKEN_RE.findall(text.lower()):
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


def embed(text: str) -> Dict[str, float]:
    """Offline 'embedding': L2-normalized bag-of-words term frequencies."""
    counts = Counter(_tokens(text))
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {t: v / norm for t, v in counts.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    return sum(a[t] * b[t] for t in common)


@dataclass
class Example:
    text: str
    label: str


def select(query: str, pool: List[Example], k: int = 3) -> List[Tuple[Example, float]]:
    """Return the k nearest examples to `query` with their similarity."""
    qv = embed(query)
    scored = [(ex, round(cosine(qv, embed(ex.text)), 3)) for ex in pool]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def predict(query: str, pool: List[Example], k: int = 3) -> str:
    """Majority label among the k nearest examples (a k-NN proxy for how the
    model would behave given those examples in its prompt)."""
    picks = select(query, pool, k)
    votes = Counter(ex.label for ex, _ in picks)
    return votes.most_common(1)[0][0]


def relevant_at_k(query_label: str, picks: List[Tuple[Example, float]]) -> float:
    """Fraction of selected examples that share the query's true label."""
    if not picks:
        return 0.0
    return sum(1 for ex, _ in picks if ex.label == query_label) / len(picks)


# --------------------------------------------------------------------------- #
# Sample data - a small labeled example pool + held-out test queries
# --------------------------------------------------------------------------- #

POOL: List[Example] = [
    Example("How do I get a refund for my order?", "refund"),
    Example("I want my money back on this purchase", "refund"),
    Example("Can I return this item and be reimbursed?", "refund"),
    Example("Where is my package right now?", "track_order"),
    Example("Track my shipment status", "track_order"),
    Example("My order hasn't arrived yet, where is it?", "track_order"),
    Example("I forgot my password and can't log in", "password_reset"),
    Example("How do I reset my account password?", "password_reset"),
    Example("Help me recover access to my account", "password_reset"),
    Example("I want to cancel my subscription", "cancel"),
    Example("How do I stop my monthly plan?", "cancel"),
    Example("Please end my membership", "cancel"),
    Example("What are your customer support hours?", "hours"),
    Example("When are you open?", "hours"),
    Example("What time does support close today?", "hours"),
]

TEST_QUERIES: List[Example] = [
    Example("How can I request a refund on my last order?", "refund"),
    Example("Can you tell me where my delivery is?", "track_order"),
    Example("I can't sign in, I lost my password", "password_reset"),
    Example("I'd like to cancel my plan", "cancel"),
    Example("What hours is your support team available?", "hours"),
    Example("My parcel is late, how do I check it?", "track_order"),
]


def _rng(seed: int):
    """Tiny deterministic LCG so the random baseline is reproducible offline."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def evaluate(k: int = 3, seed: int = 7) -> Dict[str, object]:
    """Compare nearest-example selection vs random selection on the test set."""
    rand = _rng(seed)
    rows = []
    near_rel, rand_rel, near_correct, rand_correct = [], [], 0, 0

    for q in TEST_QUERIES:
        near = select(q.text, POOL, k)
        near_r = relevant_at_k(q.label, near)
        near_rel.append(near_r)
        near_pred = Counter(ex.label for ex, _ in near).most_common(1)[0][0]
        near_correct += int(near_pred == q.label)

        # Random baseline: shuffle indices deterministically, take first k.
        idx = list(range(len(POOL)))
        idx.sort(key=lambda _: next(rand))
        rpick = [(POOL[i], 0.0) for i in idx[:k]]
        rand_r = relevant_at_k(q.label, rpick)
        rand_rel.append(rand_r)
        rand_pred = Counter(ex.label for ex, _ in rpick).most_common(1)[0][0]
        rand_correct += int(rand_pred == q.label)

        rows.append({
            "query": q.text,
            "true_label": q.label,
            "nearest_pred": near_pred,
            "near_relevant@k": round(near_r, 2),
            "rand_relevant@k": round(rand_r, 2),
            "top_example": near[0][0].text if near else "",
        })

    n = len(TEST_QUERIES)
    return {
        "k": k,
        "rows": rows,
        "near_relevant_mean": round(sum(near_rel) / n, 3),
        "rand_relevant_mean": round(sum(rand_rel) / n, 3),
        "near_accuracy": round(near_correct / n, 3),
        "rand_accuracy": round(rand_correct / n, 3),
    }


if __name__ == "__main__":
    r = evaluate(k=3)
    print(f"Few-shot example selection (k={r['k']}, pool={len(POOL)})\n")
    print(f"{'query':<45} true        nearest_pred  rel@k")
    for row in r["rows"]:
        print(f"{row['query']:<45} {row['true_label']:<11} "
              f"{row['nearest_pred']:<13} {row['near_relevant@k']}")
    print()
    print(f"Relevant@k  - nearest: {r['near_relevant_mean']:.2f}  |  "
          f"random: {r['rand_relevant_mean']:.2f}")
    print(f"Accuracy    - nearest: {r['near_accuracy']:.2f}  |  "
          f"random: {r['rand_accuracy']:.2f}")
