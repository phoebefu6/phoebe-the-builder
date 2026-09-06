from __future__ import annotations

# RAG Evaluation Harness - measure whether a retriever change actually helped.
#
# Fully offline. A tiny lexical (TF-IDF cosine) retriever stands in for a real
# vector store so the harness runs with zero dependencies beyond stdlib. Swap
# `LexicalRetriever` for your own retriever that returns a ranked list of
# doc_ids and every metric below still applies.
#
# Retrieval metrics (ranking-aware, gold = set of relevant doc_ids):
#   hit@k       : did at least one gold doc appear in the top-k?
#   recall@k    : fraction of gold docs found in the top-k
#   precision@k : fraction of top-k that are gold
#   mrr@k       : 1 / rank of the first gold doc (0 if none in top-k)
#   ndcg@k      : normalized discounted cumulative gain (position-weighted)
# Answer-level metric:
#   answer_hit  : does the retrieved context contain the gold answer string?
#                 (a cheap, deterministic stand-in for LLM-graded faithfulness)
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------- #
# A stand-in retriever: TF-IDF cosine over a small corpus. Deterministic,      #
# dependency-free, good enough to demonstrate the harness end to end.          #
# --------------------------------------------------------------------------- #
@dataclass
class LexicalRetriever:
    """Minimal TF-IDF cosine retriever. Real systems swap this out."""

    docs: Dict[str, str]  # doc_id -> text
    _idf: Dict[str, float] = field(default_factory=dict)
    _vecs: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.docs)
        df: Counter = Counter()
        tokenized = {d: tokenize(t) for d, t in self.docs.items()}
        for toks in tokenized.values():
            for term in set(toks):
                df[term] += 1
        # smoothed idf so a term in every doc still carries a little weight
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        for doc_id, toks in tokenized.items():
            self._vecs[doc_id] = self._vectorize(toks)

    def _vectorize(self, toks: Sequence[str]) -> Dict[str, float]:
        tf = Counter(toks)
        vec = {t: (c / len(toks)) * self._idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        q = self._vectorize(tokenize(query))
        scored: List[Tuple[str, float]] = []
        for doc_id, vec in self._vecs.items():
            # cosine == dot product since both sides are L2-normalized
            shared = set(q) & set(vec)
            score = sum(q[t] * vec[t] for t in shared)
            scored.append((doc_id, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [d for d, s in scored[:k] if s > 0]


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    top = retrieved[:k]
    return 1.0 if set(top) & set(gold) else 0.0


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in set(gold))
    return hits / len(top)


def mrr_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    goldset = set(gold)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in goldset:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    goldset = set(gold)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in goldset:
            dcg += 1.0 / math.log2(i + 2)  # rel is binary (0/1)
    ideal_hits = min(len(goldset), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------------------- #
# Eval case + harness                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class EvalCase:
    qid: str
    question: str
    gold_docs: List[str]
    gold_answer: str = ""  # optional substring for the answer_hit check


def _answer_hit(retrieved: Sequence[str], docs: Dict[str, str], gold_answer: str) -> float:
    """Is the gold answer present in the retrieved context? Cheap faithfulness proxy."""
    if not gold_answer:
        return math.nan  # not scored for this case
    context = " ".join(docs.get(d, "") for d in retrieved).lower()
    return 1.0 if gold_answer.lower() in context else 0.0


def evaluate(
    cases: Sequence[EvalCase],
    retriever: Callable[[str, int], List[str]],
    docs: Dict[str, str],
    k: int = 5,
) -> Dict[str, object]:
    """Run every case through the retriever and aggregate metrics.

    Edge case handled: a query the retriever answers with an empty list (no
    lexical overlap) scores 0 across the board instead of raising - a silent
    retrieval miss is exactly the failure this harness must surface, not hide.
    """
    if not cases:
        raise ValueError("No eval cases provided - nothing to evaluate.")

    per_case: List[Dict[str, object]] = []
    answer_scores: List[float] = []
    for c in cases:
        retrieved = retriever(c.question, k)
        row = {
            "qid": c.qid,
            "question": c.question,
            f"hit@{k}": hit_at_k(retrieved, c.gold_docs, k),
            f"recall@{k}": recall_at_k(retrieved, c.gold_docs, k),
            f"precision@{k}": precision_at_k(retrieved, c.gold_docs, k),
            f"mrr@{k}": mrr_at_k(retrieved, c.gold_docs, k),
            f"ndcg@{k}": ndcg_at_k(retrieved, c.gold_docs, k),
            "retrieved": retrieved,
            "gold_docs": c.gold_docs,
        }
        ah = _answer_hit(retrieved, docs, c.gold_answer)
        if not math.isnan(ah):
            answer_scores.append(ah)
            row["answer_hit"] = ah
        per_case.append(row)

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in per_case) / len(per_case)

    aggregate = {
        f"hit@{k}": _mean(f"hit@{k}"),
        f"recall@{k}": _mean(f"recall@{k}"),
        f"precision@{k}": _mean(f"precision@{k}"),
        f"mrr@{k}": _mean(f"mrr@{k}"),
        f"ndcg@{k}": _mean(f"ndcg@{k}"),
    }
    if answer_scores:
        aggregate["answer_hit"] = sum(answer_scores) / len(answer_scores)

    return {"k": k, "aggregate": aggregate, "per_case": per_case}


def compare(
    baseline: Dict[str, object], candidate: Dict[str, object]
) -> Dict[str, Dict[str, float]]:
    """Diff two eval runs (e.g. old vs new chunking). Returns metric deltas."""
    base = baseline["aggregate"]  # type: ignore[index]
    cand = candidate["aggregate"]  # type: ignore[index]
    out: Dict[str, Dict[str, float]] = {}
    for metric in base:
        b = float(base[metric])
        c = float(cand.get(metric, 0.0))
        out[metric] = {"baseline": b, "candidate": c, "delta": c - b}
    return out


# --------------------------------------------------------------------------- #
# Sample corpus + eval set (so everything runs standalone)                     #
# --------------------------------------------------------------------------- #
SAMPLE_DOCS: Dict[str, str] = {
    "d1": "Our refund policy allows returns within 30 days of purchase for a full refund.",
    "d2": "To reset your password, click 'Forgot password' on the login screen and check your email.",
    "d3": "The Pro plan costs 49 dollars per month and includes unlimited seats and priority support.",
    "d4": "Data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
    "d5": "Our support team is available Monday to Friday, 9am to 6pm Singapore time.",
    "d6": "You can export your data as CSV or JSON from the Settings > Export page.",
    "d7": "Two-factor authentication can be enabled under Security settings using an authenticator app.",
    "d8": "The Free plan supports up to 3 projects and 1 gigabyte of storage.",
}

SAMPLE_CASES: List[EvalCase] = [
    EvalCase("q1", "How do I get a refund?", ["d1"], "30 days"),
    EvalCase("q2", "I forgot my password, how do I log in?", ["d2"], "forgot password"),
    EvalCase("q3", "What does the Pro plan cost?", ["d3"], "49 dollars"),
    EvalCase("q4", "Is my data encrypted?", ["d4"], "aes-256"),
    EvalCase("q5", "When can I contact support?", ["d5"], "monday to friday"),
    EvalCase("q6", "How do I download my data?", ["d6"], "csv"),
    EvalCase("q7", "How do I turn on 2FA?", ["d7"], "authenticator"),
    EvalCase("q8", "How many projects on the free tier?", ["d8"], "3 projects"),
]


if __name__ == "__main__":
    retr = LexicalRetriever(SAMPLE_DOCS)
    result = evaluate(SAMPLE_CASES, retr.retrieve, SAMPLE_DOCS, k=3)
    print(f"Aggregate metrics @k={result['k']}:")
    for metric, val in result["aggregate"].items():  # type: ignore[union-attr]
        print(f"  {metric:14s} {val:.3f}")
