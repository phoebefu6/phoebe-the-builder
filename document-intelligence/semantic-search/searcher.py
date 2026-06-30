from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchHit:
    """A single retrieved document with its similarity score."""

    text: str
    score: float
    index: int


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# ----------------------- local TF-IDF "embedding" -----------------------
# A deterministic, dependency-free vectorizer. Stands in for a real embedding
# model: it captures term importance (IDF) so "how do I get my money back"
# still ranks the refund passage above a literal keyword match. Upgrade path
# to dense Voyage/OpenAI embeddings is one swap away (see embed_remote).


class TfidfVectorizer:
    """Minimal TF-IDF vectorizer: fit a vocab + IDF on a corpus, transform text to a vector."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.vocab: set[str] = set()

    def fit(self, corpus: list[str]) -> "TfidfVectorizer":
        n = len(corpus)
        df: Counter = Counter()
        for doc in corpus:
            for term in set(_tokens(doc)):
                df[term] += 1
        # smoothed idf so terms in every doc still carry a little weight
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.vocab = set(self.idf)
        return self

    def transform(self, text: str) -> dict[str, float]:
        counts = Counter(t for t in _tokens(text) if t in self.vocab)
        if not counts:
            return {}
        total = sum(counts.values())
        return {t: (c / total) * self.idf[t] for t, c in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------- vector store ---------------------------


@dataclass
class SemanticIndex:
    """A tiny in-memory vector store: embed a corpus once, then rank by cosine similarity.

    Backend defaults to local TF-IDF (no API key). Pass embed_fn to swap in dense
    embeddings (Voyage AI, OpenAI, etc.) - the search logic is identical."""

    docs: list[str] = field(default_factory=list)
    vectors: list = field(default_factory=list)
    _vectorizer: Optional[TfidfVectorizer] = None
    embed_fn: Optional[object] = None  # callable(text) -> list[float], for dense embeddings

    def build(self, docs: list[str]) -> "SemanticIndex":
        self.docs = [d.strip() for d in docs if d.strip()]
        if self.embed_fn is not None:
            self.vectors = [self.embed_fn(d) for d in self.docs]
        else:
            self._vectorizer = TfidfVectorizer().fit(self.docs)
            self.vectors = [self._vectorizer.transform(d) for d in self.docs]
        return self

    def _embed_query(self, query: str):
        if self.embed_fn is not None:
            return self.embed_fn(query)
        return self._vectorizer.transform(query)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.docs:
            return []
        qv = self._embed_query(query)
        sim = _dense_cosine if self.embed_fn is not None else _cosine
        scored = [
            SearchHit(text=self.docs[i], score=sim(qv, self.vectors[i]), index=i)
            for i in range(len(self.docs))
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def _dense_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ------------------- optional dense embeddings (Voyage) -------------------


def make_voyage_embedder(api_key: Optional[str] = None):
    """Return an embed_fn backed by Voyage AI (Anthropic's recommended embeddings).

    Lets SemanticIndex use true dense vectors instead of TF-IDF. Requires the `voyageai`
    package and a VOYAGE_API_KEY. Falls back to None (local TF-IDF) if unavailable."""
    api_key = api_key or os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    try:
        import voyageai
    except ImportError:
        return None

    client = voyageai.Client(api_key=api_key)

    def embed(text: str) -> list[float]:
        return client.embed([text], model="voyage-3", input_type="document").embeddings[0]

    return embed


def keyword_baseline(query: str, docs: list[str], top_k: int = 5) -> list[SearchHit]:
    """Naive substring/keyword match - the 'Ctrl+F' baseline to contrast against semantic search."""
    q_terms = set(_tokens(query))
    hits = []
    for i, d in enumerate(docs):
        d_terms = _tokens(d)
        overlap = sum(1 for t in d_terms if t in q_terms)
        hits.append(SearchHit(text=d, score=float(overlap), index=i))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


SAMPLE_DOCS = [
    "Reset your password from the login screen by clicking 'Forgot password'.",
    "We offer full refunds within 30 days of purchase, no questions asked.",
    "Our mobile app is available on both iOS and Android.",
    "To cancel your subscription, go to Settings and choose 'Cancel plan'.",
    "Enterprise customers get a dedicated account manager and priority support.",
    "Data is encrypted at rest with AES-256 and in transit with TLS 1.3.",
    "You can export all your data to CSV from the account dashboard.",
    "Two-factor authentication adds a one-time code at sign-in for extra security.",
    "Billing happens monthly on the anniversary of your signup date.",
    "Get money back if you are not satisfied - just contact support within a month.",
]
