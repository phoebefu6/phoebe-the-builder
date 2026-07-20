from __future__ import annotations

# Semantic Response Cache - core logic.
#
# Users ask the same thing in different words. An exact-match cache misses
# "how do I get a refund?" vs "how can I request a refund?" - so you pay the
# model again for an answer you already have. A *semantic* cache compares the
# meaning of the new query against past queries and reuses the stored response
# when they are close enough. This module ships a dependency-free lexical
# embedder (bag-of-words cosine with light normalization) so it runs fully
# offline - swap in real embeddings for production.

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "the", "is", "are", "do", "does", "i", "how", "can", "to",
    "of", "for", "my", "me", "you", "your", "what", "with", "on", "in",
}


def _normalize(tokens: List[str]) -> List[str]:
    """Lowercase already done by tokenizer; drop stopwords + naive plural -s."""
    out: List[str] = []
    for t in tokens:
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        out.append(t)
    return out


def embed(text: str) -> Dict[str, float]:
    """Tiny offline 'embedding': normalized bag-of-words term-frequency vector.
    Not a real semantic model, but enough to catch paraphrase overlap."""
    toks = _normalize(_TOKEN_RE.findall(text.lower()))
    vec: Dict[str, float] = {}
    for t in toks:
        vec[t] = vec.get(t, 0.0) + 1.0
    # L2 normalize so cosine is a plain dot product.
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm:
        for k in vec:
            vec[k] /= norm
    return vec


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    return sum(a[t] * b[t] for t in common)


@dataclass
class CacheEntry:
    query: str
    vec: Dict[str, float]
    response: str
    hits: int = 0


@dataclass
class LookupResult:
    hit: bool
    response: str
    similarity: float
    matched_query: Optional[str] = None
    cost_saved: float = 0.0


class SemanticCache:
    """Cache keyed by query meaning. On lookup, return the stored response if
    the nearest past query is at least `threshold` similar."""

    def __init__(
        self,
        threshold: float = 0.8,
        cost_per_call: float = 0.01,
        model_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.threshold = threshold
        self.cost_per_call = cost_per_call
        self.model_fn = model_fn or (lambda q: f"[generated answer for: {q}]")
        self.entries: List[CacheEntry] = []
        self.stats = {"lookups": 0, "hits": 0, "misses": 0, "cost_saved": 0.0}

    def _nearest(self, vec: Dict[str, float]) -> Tuple[Optional[CacheEntry], float]:
        best, best_sim = None, 0.0
        for e in self.entries:
            s = cosine(vec, e.vec)
            if s > best_sim:
                best, best_sim = e, s
        return best, best_sim

    def ask(self, query: str) -> LookupResult:
        """Main entry: return a cached response on a semantic hit, otherwise
        call the model, store the answer, and return it as a miss."""
        self.stats["lookups"] += 1
        vec = embed(query)
        entry, sim = self._nearest(vec)
        if entry is not None and sim >= self.threshold:
            entry.hits += 1
            self.stats["hits"] += 1
            self.stats["cost_saved"] += self.cost_per_call
            return LookupResult(
                hit=True,
                response=entry.response,
                similarity=round(sim, 3),
                matched_query=entry.query,
                cost_saved=self.cost_per_call,
            )
        # Miss - generate, store, return.
        self.stats["misses"] += 1
        response = self.model_fn(query)
        self.entries.append(CacheEntry(query, vec, response))
        return LookupResult(hit=False, response=response, similarity=round(sim, 3))

    @property
    def hit_rate(self) -> float:
        n = self.stats["lookups"]
        return round(self.stats["hits"] / n, 3) if n else 0.0

    def summary(self) -> dict:
        return {
            **self.stats,
            "cost_saved": round(self.stats["cost_saved"], 4),
            "hit_rate": self.hit_rate,
            "unique_cached": len(self.entries),
        }


# --------------------------------------------------------------------------
# Sample query stream - paraphrases of a few real intents
# --------------------------------------------------------------------------

SAMPLE_QUERIES: List[str] = [
    "How do I get a refund?",
    "How can I request a refund?",          # paraphrase of #1
    "What is your refund policy?",          # related but distinct wording
    "How do I reset my password?",
    "I forgot my password, how do I reset it?",  # paraphrase of #4
    "How do I get a refund for my order?",  # paraphrase of #1
    "Where can I download my invoice?",
    "How do I download an invoice?",        # paraphrase of #7
    "How do I reset my password?",          # exact repeat of #4
]


def run_sample(threshold: float = 0.7) -> Tuple[SemanticCache, List[LookupResult]]:
    cache = SemanticCache(threshold=threshold, cost_per_call=0.01)
    results = [cache.ask(q) for q in SAMPLE_QUERIES]
    return cache, results


if __name__ == "__main__":
    cache, results = run_sample(threshold=0.7)
    for q, r in zip(SAMPLE_QUERIES, results):
        tag = f"HIT  (sim={r.similarity} ~ '{r.matched_query}')" if r.hit else f"MISS (best sim={r.similarity})"
        print(f"[{'HIT ' if r.hit else 'MISS'}] {q:45} {tag if r.hit else ''}")
    s = cache.summary()
    print(f"\n{s['hits']}/{s['lookups']} hits · hit_rate={s['hit_rate']} · "
          f"${s['cost_saved']} saved · {s['unique_cached']} unique answers cached")
