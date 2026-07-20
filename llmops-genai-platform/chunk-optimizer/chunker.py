from __future__ import annotations

# Chunking Strategy Tester - core logic.
#
# In RAG, the retriever can only surface a chunk that actually *contains* the
# answer. Chunk too big and you dilute the signal (and waste context tokens);
# chunk too small and you split the answer across chunks so no single one is
# retrievable. This module splits a corpus with several strategies, retrieves
# over the chunks with a dependency-free lexical scorer, and scores each
# strategy on a gold eval set - so you can pick a chunking config on evidence
# instead of vibes. Fully offline: no API keys, no embeddings service.

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# --------------------------------------------------------------------------
# Sample corpus + gold eval set (stand-ins for your real knowledge base)
# --------------------------------------------------------------------------

SAMPLE_DOCS: Dict[str, str] = {
    "refund_policy": (
        "Customers may request a refund within 30 days of purchase. "
        "Refunds are issued to the original payment method within 5 to 7 "
        "business days. Digital goods are non-refundable once downloaded. "
        "To start a refund, email support with your order number. "
        "Shipping fees are not refundable except when the item arrived damaged."
    ),
    "security": (
        "All customer data is encrypted at rest using AES-256. "
        "Data in transit is protected with TLS 1.3. "
        "We run a quarterly third-party penetration test. "
        "Access to production databases requires multi-factor authentication. "
        "Secrets are stored in a managed vault and rotated every 90 days."
    ),
    "onboarding": (
        "New hires complete IT setup on day one and receive a laptop. "
        "The buddy program pairs each new hire with a mentor for 30 days. "
        "Benefits enrollment must be completed within the first two weeks. "
        "The first performance check-in happens at the 90-day mark. "
        "All employees finish security awareness training before week three."
    ),
    "sla": (
        "Priority 1 incidents receive a response within 15 minutes. "
        "Priority 2 incidents receive a response within 2 hours. "
        "Our uptime commitment is 99.9 percent measured monthly. "
        "Scheduled maintenance windows are announced 72 hours in advance. "
        "Service credits apply when monthly uptime falls below the commitment."
    ),
}


@dataclass
class EvalCase:
    qid: str
    question: str
    doc_id: str  # document the answer lives in
    answer_span: str  # exact phrase a good chunk must contain


SAMPLE_CASES: List[EvalCase] = [
    EvalCase("q1", "How long do I have to ask for a refund?", "refund_policy",
             "within 30 days"),
    EvalCase("q2", "How are refunds paid back?", "refund_policy",
             "original payment method within 5 to 7"),
    EvalCase("q3", "How is data encrypted when stored?", "security",
             "AES-256"),
    EvalCase("q4", "How often are secrets rotated?", "security",
             "rotated every 90 days"),
    EvalCase("q5", "When does a new hire get a mentor?", "onboarding",
             "buddy program pairs each new hire with a mentor for 30 days"),
    EvalCase("q6", "When is the first performance review?", "onboarding",
             "90-day mark"),
    EvalCase("q7", "What is the response time for a P1 incident?", "sla",
             "within 15 minutes"),
    EvalCase("q8", "What is the monthly uptime commitment?", "sla",
             "99.9 percent"),
]


# --------------------------------------------------------------------------
# Chunk representation
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


# --------------------------------------------------------------------------
# Chunking strategies
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def chunk_fixed_words(
    doc_id: str, text: str, size: int, overlap: int = 0
) -> List[Chunk]:
    """Fixed window of `size` words with `overlap` words carried forward."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        # overlap >= size would never advance -> infinite loop. Clamp it.
        overlap = max(0, min(overlap, size - 1))
    words = _words(text)
    chunks: List[Chunk] = []
    step = size - overlap
    i = 0
    idx = 0
    while i < len(words):
        piece = " ".join(words[i : i + size])
        chunks.append(Chunk(f"{doc_id}#w{idx}", doc_id, piece))
        idx += 1
        i += step
    return chunks


def chunk_by_sentence(
    doc_id: str, text: str, per_chunk: int = 1
) -> List[Chunk]:
    """Group `per_chunk` sentences per chunk - respects natural boundaries."""
    if per_chunk <= 0:
        raise ValueError("per_chunk must be positive")
    sents = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    chunks: List[Chunk] = []
    for idx, i in enumerate(range(0, len(sents), per_chunk)):
        piece = " ".join(sents[i : i + per_chunk])
        chunks.append(Chunk(f"{doc_id}#s{idx}", doc_id, piece))
    return chunks


def chunk_whole_doc(doc_id: str, text: str) -> List[Chunk]:
    """Baseline: the entire document is one chunk (no splitting)."""
    return [Chunk(f"{doc_id}#full", doc_id, text)]


# A strategy is a name + a function that turns the whole corpus into chunks.
Strategy = Callable[[Dict[str, str]], List[Chunk]]


def build_strategies() -> Dict[str, Strategy]:
    """Named chunking configs to compare head-to-head."""

    def _apply(fn: Callable[[str, str], List[Chunk]]) -> Strategy:
        def run(docs: Dict[str, str]) -> List[Chunk]:
            out: List[Chunk] = []
            for doc_id, text in docs.items():
                out.extend(fn(doc_id, text))
            return out

        return run

    return {
        "whole_doc": _apply(chunk_whole_doc),
        "words_8": _apply(lambda d, t: chunk_fixed_words(d, t, 8, 0)),
        "words_15": _apply(lambda d, t: chunk_fixed_words(d, t, 15, 0)),
        "words_20_overlap_5": _apply(
            lambda d, t: chunk_fixed_words(d, t, 20, 5)
        ),
        "sentence_1": _apply(lambda d, t: chunk_by_sentence(d, t, 1)),
        "sentence_2": _apply(lambda d, t: chunk_by_sentence(d, t, 2)),
    }


# --------------------------------------------------------------------------
# Dependency-free lexical retriever (TF-IDF cosine)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalRetriever:
    """Tiny TF-IDF cosine retriever over a chunk list - no external deps."""

    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.df: Dict[str, int] = {}
        self._tf: List[Dict[str, int]] = []
        for ch in chunks:
            counts: Dict[str, int] = {}
            for tok in _tokenize(ch.text):
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                self.df[tok] = self.df.get(tok, 0) + 1
        self.n = max(1, len(chunks))
        self._vecs = [self._to_vec(tf) for tf in self._tf]

    def _idf(self, tok: str) -> float:
        return math.log((1 + self.n) / (1 + self.df.get(tok, 0))) + 1.0

    def _to_vec(self, tf: Dict[str, int]) -> Dict[str, float]:
        return {tok: c * self._idf(tok) for tok, c in tf.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def query(self, text: str, k: int = 3) -> List[Tuple[Chunk, float]]:
        q_counts: Dict[str, int] = {}
        for tok in _tokenize(text):
            q_counts[tok] = q_counts.get(tok, 0) + 1
        q_vec = self._to_vec(q_counts)
        scored = [
            (self.chunks[i], self._cosine(q_vec, self._vecs[i]))
            for i in range(len(self.chunks))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------
# Scoring a strategy
# --------------------------------------------------------------------------


@dataclass
class StrategyResult:
    name: str
    n_chunks: int
    avg_chunk_words: float
    hit_rate_at_k: float  # fraction of queries whose answer chunk is in top-k
    mrr: float  # mean reciprocal rank of the first answer-bearing chunk
    per_query: List[dict] = field(default_factory=list)


def _chunk_has_answer(chunk: Chunk, case: EvalCase) -> bool:
    """A chunk 'wins' if it is from the right doc AND holds the answer span."""
    if chunk.doc_id != case.doc_id:
        return False
    return case.answer_span.lower() in chunk.text.lower()


def evaluate_strategy(
    name: str,
    strategy: Strategy,
    docs: Dict[str, str],
    cases: List[EvalCase],
    k: int = 3,
) -> StrategyResult:
    chunks = strategy(docs)
    retriever = LexicalRetriever(chunks)
    avg_words = (
        sum(len(_words(c.text)) for c in chunks) / len(chunks) if chunks else 0.0
    )

    hits = 0
    rr_sum = 0.0
    per_query: List[dict] = []
    for case in cases:
        ranked = retriever.query(case.question, k=k)
        rank = None
        for pos, (chunk, _score) in enumerate(ranked, start=1):
            if _chunk_has_answer(chunk, case):
                rank = pos
                break
        hit = rank is not None
        hits += int(hit)
        rr_sum += (1.0 / rank) if rank else 0.0
        top = ranked[0][0] if ranked else None
        per_query.append(
            {
                "qid": case.qid,
                "question": case.question,
                "hit": hit,
                "rank": rank,
                "top_chunk_id": top.chunk_id if top else None,
                "top_chunk": (top.text[:80] + "...") if top else None,
            }
        )

    n = max(1, len(cases))
    return StrategyResult(
        name=name,
        n_chunks=len(chunks),
        avg_chunk_words=round(avg_words, 1),
        hit_rate_at_k=round(hits / n, 3),
        mrr=round(rr_sum / n, 3),
        per_query=per_query,
    )


def compare_strategies(
    docs: Dict[str, str] | None = None,
    cases: List[EvalCase] | None = None,
    k: int = 3,
) -> List[StrategyResult]:
    """Run every strategy and return results sorted best-first."""
    docs = docs or dict(SAMPLE_DOCS)
    cases = cases or list(SAMPLE_CASES)
    results = [
        evaluate_strategy(name, strat, docs, cases, k=k)
        for name, strat in build_strategies().items()
    ]
    results.sort(key=lambda r: (r.hit_rate_at_k, r.mrr), reverse=True)
    return results


if __name__ == "__main__":
    print(f"{'strategy':22} {'chunks':>6} {'avg_w':>6} {'hit@k':>6} {'mrr':>6}")
    print("-" * 52)
    for r in compare_strategies(k=3):
        print(
            f"{r.name:22} {r.n_chunks:>6} {r.avg_chunk_words:>6} "
            f"{r.hit_rate_at_k:>6} {r.mrr:>6}"
        )
