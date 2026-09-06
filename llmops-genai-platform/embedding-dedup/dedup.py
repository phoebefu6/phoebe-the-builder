from __future__ import annotations

# Near-duplicate finder for a RAG corpus. Three-signal gate (cosine + containment +
# numeric fingerprint), rare-token blocking so it is not O(n^2) on real corpora,
# union-find clustering, and a keep/drop plan with the retrieval impact.
#
# Fully offline: TF-IDF vectors stand in for embeddings so the demo runs with no API
# key. Any real embedding matrix can be passed in via `vectors=` - the gate is the
# part that matters, not the vectorizer.
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")
NUMBER_RE = re.compile(r"\b\d[\d,]*\.?\d*%?\b")
WS_RE = re.compile(r"\s+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "must", "no", "not", "of", "on", "or", "our", "so", "than", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "we", "were", "what", "when",
    "which", "will", "with", "you", "your",
}

# ---------------------------------------------------------------- text normalisation


def normalize(text: str) -> str:
    """Casefold + collapse whitespace + drop markdown decoration.

    Two exports of the same page differ in exactly these ways, which is why an exact
    hash over raw text finds almost nothing in practice.
    """
    t = text.lower().replace("’", "'")
    t = re.sub(r"[#*_`>|]+", " ", t)
    t = re.sub(r"[^a-z0-9'\-\.%\s]", " ", t)
    return WS_RE.sub(" ", t).strip()


def exact_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()[:12]


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = TOKEN_RE.findall(normalize(text))
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    return toks


def numbers_in(text: str) -> Set[str]:
    """Numeric fingerprint: the figures a document asserts.

    Two filings off the same template share almost all their prose and disagree on
    exactly the part that matters. This is the signal that separates them.
    """
    return {n.replace(",", "").rstrip(".") for n in NUMBER_RE.findall(text)}


def est_tokens(text: str) -> int:
    """~4 chars/token. Good enough for a savings estimate, wrong for billing."""
    return max(1, round(len(text) / 4))


# ---------------------------------------------------------------- vectorisation


class TfidfIndex:
    """Minimal TF-IDF with L2-normalised rows, so cosine is a dot product.

    Deliberately dependency-light: swapping in real embeddings changes `vectors`
    and nothing else in this file.
    """

    def __init__(self, texts: Sequence[str], sublinear: bool = True) -> None:
        self.token_lists = [tokenize(t) for t in texts]
        vocab: Dict[str, int] = {}
        for toks in self.token_lists:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab
        n, v = len(texts), max(1, len(vocab))
        counts = np.zeros((n, v), dtype=np.float64)
        for i, toks in enumerate(self.token_lists):
            for t in toks:
                counts[i, vocab[t]] += 1.0
        self.df = (counts > 0).sum(axis=0)
        self.idf = np.log((1.0 + n) / (1.0 + self.df)) + 1.0
        tf = np.log1p(counts) if sublinear else counts
        m = tf * self.idf
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = m / norms

    def rare_tokens(self, i: int, max_df_ratio: float = 0.5) -> Set[str]:
        """Tokens in doc i that are not near-universal in the corpus."""
        n = self.vectors.shape[0]
        cap = max(1.0, max_df_ratio * n)
        return {t for t in set(self.token_lists[i]) if self.df[self.vocab[t]] <= cap}

    def query(self, text: str) -> np.ndarray:
        v = np.zeros(max(1, len(self.vocab)))
        for t in tokenize(text):
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        v = np.log1p(v) * self.idf
        nrm = np.linalg.norm(v)
        return v if nrm == 0 else v / nrm


# ---------------------------------------------------------------- candidate blocking


def candidate_pairs(index: TfidfIndex, max_df_ratio: float = 0.5) -> Set[Tuple[int, int]]:
    """Inverted index on discriminative tokens; only co-occurring docs get compared.

    All-pairs cosine is 500M comparisons at 30k docs. Two documents with no rare
    token in common cannot clear a 0.8 cosine bar, so they never need scoring.
    """
    n = len(index.token_lists)
    postings: Dict[str, List[int]] = {}
    cap = max(1.0, max_df_ratio * n)
    for i in range(n):
        for t in set(index.token_lists[i]):
            if index.df[index.vocab[t]] <= cap:
                postings.setdefault(t, []).append(i)
    pairs: Set[Tuple[int, int]] = set()
    for docs in postings.values():
        if len(docs) < 2:
            continue
        for a_idx, a in enumerate(docs):
            for b in docs[a_idx + 1:]:
                pairs.add((a, b) if a < b else (b, a))
    return pairs


# ---------------------------------------------------------------- the three signals


def containment(a: Set[str], b: Set[str]) -> float:
    """|A ∩ B| / min(|A|,|B|) over rare tokens - catches subset documents.

    Cosine punishes a length mismatch, so a paragraph copied verbatim into a longer
    page scores low. Containment does not care about the extra material.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def numeric_agreement(a: Set[str], b: Set[str], min_numbers: int = 2) -> Optional[float]:
    """Jaccard over numeric fingerprints. None = not enough numbers to judge."""
    if len(a) < min_numbers or len(b) < min_numbers:
        return None
    union = a | b
    return len(a & b) / len(union) if union else None


# ---------------------------------------------------------------- clustering


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


MIN_TOKENS_TO_JUDGE = 10


def find_duplicates(
    docs: Sequence[Dict],
    cos_threshold: float = 0.80,
    contain_threshold: float = 0.65,
    subset_threshold: float = 0.90,
    numeric_threshold: float = 0.50,
    max_df_ratio: float = 0.5,
    vectors: Optional[np.ndarray] = None,
    text_key: str = "text",
) -> Dict:
    """Cluster near-duplicates and report every pair the gate rejected, with the reason.

    Two ways to link, because cosine and containment answer different questions:
      * `near`   - cosine >= cos_threshold AND containment >= contain_threshold
                   (two versions of the same document, similar length)
      * `subset` - containment >= subset_threshold at any cosine
                   (a paragraph absorbed into a longer page: cosine punishes the
                   length mismatch, containment does not care about the extra text)
    Either way the numeric fingerprint can veto the link.

    Edge cases handled up front: a corpus of 0 or 1 documents has no pairs to score,
    and fragments below MIN_TOKENS_TO_JUDGE (nav bars, footers, "See also" stubs) are
    too short for any of the three signals to mean anything - they are held out and
    reported rather than dragged into someone else's duplicate group.
    """
    texts = [d[text_key] for d in docs]
    n = len(texts)
    if n < 2:
        return {
            "clusters": [], "pairs": [], "rejected": [], "fragments": list(range(n)) if n else [],
            "n_docs": n, "n_candidate_pairs": 0, "n_all_pairs": 0,
            "index": TfidfIndex(texts) if n else None,
        }

    index = TfidfIndex(texts)
    V = index.vectors if vectors is None else _l2(np.asarray(vectors, dtype=np.float64))

    rare = [index.rare_tokens(i, max_df_ratio) for i in range(n)]
    nums = [numbers_in(texts[i]) for i in range(n)]
    fragments = {i for i in range(n) if len(index.token_lists[i]) < MIN_TOKENS_TO_JUDGE}

    # Exact matches after normalisation are free - hash, do not score.
    by_hash: Dict[str, List[int]] = {}
    for i, t in enumerate(texts):
        by_hash.setdefault(exact_hash(t), []).append(i)

    uf = _UnionFind(n)
    pairs: List[Dict] = []
    rejected: List[Dict] = []
    for group in by_hash.values():
        for j in group[1:]:
            uf.union(group[0], j)
            pairs.append({
                "a": group[0], "b": j, "cosine": 1.0, "containment": 1.0,
                "numeric": 1.0, "kind": "exact",
            })

    linked = {(p["a"], p["b"]) for p in pairs}
    cands = candidate_pairs(index, max_df_ratio)
    for a, b in sorted(cands):
        if (a, b) in linked or a in fragments or b in fragments:
            continue
        cos = float(V[a] @ V[b])
        con = containment(rare[a], rare[b])
        # Not close on either axis - no story to tell, do not clutter the report.
        if cos < 0.35 and con < subset_threshold:
            continue
        num = numeric_agreement(nums[a], nums[b])
        row = {"a": a, "b": b, "cosine": round(cos, 3), "containment": round(con, 3),
               "numeric": None if num is None else round(num, 3)}

        is_near = cos >= cos_threshold and con >= contain_threshold
        is_subset = con >= subset_threshold
        vetoed = num is not None and num < numeric_threshold

        if (is_near or is_subset) and not vetoed:
            row["kind"] = "subset" if (is_subset and not is_near) else "near"
            pairs.append(row)
            uf.union(a, b)
            continue

        reasons = []
        if vetoed:
            reasons.append(f"numeric fingerprints disagree ({num:.2f} < {numeric_threshold:.2f})")
        if not (is_near or is_subset):
            if cos < cos_threshold:
                reasons.append(f"cosine {cos:.2f} < {cos_threshold:.2f}")
            if con < contain_threshold:
                reasons.append(f"containment {con:.2f} < {contain_threshold:.2f}")
            elif con < subset_threshold:
                reasons.append(f"containment {con:.2f} < subset bar {subset_threshold:.2f}")
        row["reasons"] = reasons
        rejected.append(row)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        # Canonical survivor = the longest member: a superset keeps the detail a
        # shorter copy dropped. Ties break on original order for reproducibility.
        keep = max(members, key=lambda i: (len(texts[i]), -i))
        clusters.append({
            "keep": keep,
            "drop": [i for i in members if i != keep],
            "members": sorted(members),
            "size": len(members),
        })
    clusters.sort(key=lambda c: (-c["size"], c["keep"]))

    return {
        "clusters": clusters,
        "pairs": pairs,
        "rejected": sorted(rejected, key=lambda r: -max(r["cosine"], r["containment"])),
        "fragments": sorted(fragments),
        "n_docs": n,
        "n_candidate_pairs": len(cands),
        "n_all_pairs": n * (n - 1) // 2,
        "index": index,
    }


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ---------------------------------------------------------------- the plan


def dedup_plan(
    docs: Sequence[Dict],
    result: Dict,
    price_per_mtok: float = 0.13,
    text_key: str = "text",
) -> Dict:
    """Keep/drop rows plus what dropping them buys. Savings are an estimate."""
    dropped: List[Dict] = []
    for c in result["clusters"]:
        for i in c["drop"]:
            dropped.append({
                "index": i,
                "id": docs[i].get("id", i),
                "duplicate_of": docs[c["keep"]].get("id", c["keep"]),
                "tokens": est_tokens(docs[i][text_key]),
            })
    tokens_saved = sum(d["tokens"] for d in dropped)
    total_tokens = sum(est_tokens(d[text_key]) for d in docs)
    return {
        "keep_count": len(docs) - len(dropped),
        "drop_count": len(dropped),
        "dropped": dropped,
        "tokens_saved": tokens_saved,
        "total_tokens": total_tokens,
        "pct_index_saved": round(100 * tokens_saved / total_tokens, 1) if total_tokens else 0.0,
        "embedding_cost_saved": round(tokens_saved / 1e6 * price_per_mtok, 6),
    }


def retrieve(
    index: TfidfIndex,
    query: str,
    k: int = 5,
    exclude: Optional[Iterable[int]] = None,
) -> List[Tuple[int, float]]:
    """Top-k by cosine. `exclude` simulates the post-dedup index."""
    drop = set(exclude or ())
    scores = index.vectors @ index.query(query)
    order = np.argsort(-scores)
    out = [(int(i), round(float(scores[i]), 3)) for i in order if i not in drop and scores[i] > 0]
    return out[:k]


def distinct_answers(result: Dict, hits: Sequence[Tuple[int, float]]) -> int:
    """How many *different* answers a result page actually carries.

    Five slots holding three copies of one page is a two-answer result page. This is
    the cost duplicates impose that no cost report shows.
    """
    label: Dict[int, str] = {}
    for c_idx, c in enumerate(result["clusters"]):
        for i in c["members"]:
            label[i] = f"c{c_idx}"
    return len({label.get(i, f"d{i}") for i, _ in hits})


def redundant_slots(result: Dict, hits: Sequence[Tuple[int, float]]) -> int:
    """Slots holding a copy of something already shown higher up the page."""
    return len(hits) - distinct_answers(result, hits) if hits else 0


def evaluate(predicted: Iterable[Tuple[int, int]], truth: Iterable[Tuple[int, int]]) -> Dict:
    """Pair-level precision/recall. A dedup false positive destroys information."""
    p, t = {tuple(sorted(x)) for x in predicted}, {tuple(sorted(x)) for x in truth}
    tp, fp, fn = len(p & t), len(p - t), len(t - p)
    prec = tp / (tp + fp) if p else 0.0
    rec = tp / (tp + fn) if t else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "false_merges": sorted(p - t), "missed": sorted(t - p)}


def truth_pairs(docs: Sequence[Dict], key: str = "dup_group") -> List[Tuple[int, int]]:
    groups: Dict[str, List[int]] = {}
    for i, d in enumerate(docs):
        g = d.get(key)
        if g:
            groups.setdefault(g, []).append(i)
    out = []
    for members in groups.values():
        for a_idx, a in enumerate(members):
            for b in members[a_idx + 1:]:
                out.append((a, b))
    return out


def cluster_pairs(result: Dict) -> List[Tuple[int, int]]:
    """Every pair implied by the clusters (transitive closure, not just linked pairs)."""
    out = []
    for c in result["clusters"]:
        m = c["members"]
        for a_idx, a in enumerate(m):
            for b in m[a_idx + 1:]:
                out.append((a, b))
    return out


# ---------------------------------------------------------------- sample corpus
# A support/finance knowledge base with the four things a real RAG index contains:
# re-exported copies, an edited fork, a paragraph absorbed into a longer page, and
# template siblings that are NOT duplicates.

_REFUND = (
    "Refund policy. Customers may request a full refund within 30 days of purchase "
    "provided the licence has been activated on fewer than three devices. Refunds are "
    "issued to the original payment method within 5 business days of approval. "
    "Annual plans cancelled after day 30 are prorated to the next billing anniversary. "
    "Refunds cannot be issued for usage-based overage already invoiced."
)

SAMPLE_DOCS: List[Dict] = [
    # --- Group A: same page exported three ways (exact after normalisation)
    {"id": "kb-refund-policy", "source": "helpdesk", "dup_group": "A", "text": _REFUND},
    {"id": "web-refunds-page", "source": "website", "dup_group": "A",
     "text": "**" + _REFUND.replace(". ", ".   ").upper() + "**\n"},
    {"id": "pdf-refunds-v3", "source": "pdf-export", "dup_group": "A",
     "text": _REFUND.replace("Refund policy.", "**Refund Policy**\n") + "\n\n_Page 4 of 12_"},

    # --- Group B: an edited fork - reworded, one clause added, one dropped
    {"id": "kb-onboarding-2024", "source": "helpdesk", "dup_group": "B",
     "text": "Customer onboarding checklist. Create the workspace and invite the billing "
             "owner first. Connect the primary data source before inviting analysts, "
             "because permissions inherit from the workspace at connection time. Run the "
             "sample import to confirm the schema mapping. Schedule the 30-minute "
             "walkthrough within the first week of activation."},
    {"id": "kb-onboarding-2025-draft", "source": "notion", "dup_group": "B",
     "text": "Customer onboarding checklist (2025 revision). Create the workspace and "
             "invite the billing owner first. Connect the primary data source before "
             "inviting analysts, because permissions inherit from the workspace at "
             "connection time. Run the sample import to confirm the schema mapping. "
             "Verify SSO before the walkthrough. Schedule the 30-minute walkthrough "
             "within the first week of activation."},

    # --- Group C: a paragraph swallowed by a longer page (cosine fails, containment saves it)
    {"id": "kb-sla-shipping", "source": "helpdesk", "dup_group": "C",
     "text": "Standard fulfilment is dispatched within two business days of payment "
             "clearing. Express fulfilment is dispatched same day when the order is "
             "placed before 1400 local time."},
    {"id": "web-shipping-faq", "source": "website", "dup_group": "C",
     "text": "Shipping FAQ. Where is my order? Tracking appears in your account within a "
             "few hours of dispatch. Standard fulfilment is dispatched within two business "
             "days of payment clearing. Express fulfilment is dispatched same day when the "
             "order is placed before 1400 local time. Do you ship to PO boxes? Not for "
             "express orders, because the courier requires a signature on delivery. What "
             "about customs? Duties for international destinations are collected by the "
             "carrier at the door and are not included in the quoted shipping price."},

    # --- Template siblings: near-identical prose, different figures. NOT duplicates.
    {"id": "fin-q1-summary", "source": "finance", "dup_group": None,
     "text": "Quarterly revenue summary. Net revenue for the quarter was 4820000, up 12 "
             "percent against the comparable prior-year quarter. Gross margin held at 71 "
             "percent. Net revenue retention finished at 108 percent. Sales and marketing "
             "expense was 1310000, or 27 percent of net revenue. Free cash flow was 402000."},
    {"id": "fin-q2-summary", "source": "finance", "dup_group": None,
     "text": "Quarterly revenue summary. Net revenue for the quarter was 5140000, up 9 "
             "percent against the comparable prior-year quarter. Gross margin held at 69 "
             "percent. Net revenue retention finished at 111 percent. Sales and marketing "
             "expense was 1455000, or 28 percent of net revenue. Free cash flow was 517000."},

    # --- Boilerplate fragments: similar to everything, meaningful to nothing
    {"id": "nav-footer-a", "source": "website", "dup_group": None,
     "text": "See also: pricing, security, status page."},
    {"id": "nav-footer-b", "source": "website", "dup_group": None,
     "text": "See also: pricing, careers, status page."},

    # --- Genuinely unique documents
    {"id": "kb-sso-setup", "source": "helpdesk", "dup_group": None,
     "text": "Configuring SAML single sign-on. Upload the identity provider metadata XML, "
             "then map the email assertion to the account identifier. Domain capture is "
             "required before enforcement, otherwise existing password users are locked "
             "out at the next session refresh."},
    {"id": "kb-rate-limits", "source": "helpdesk", "dup_group": None,
     "text": "API rate limits. The default ceiling is 600 requests per minute per "
             "workspace token, burstable to 900 for 10 seconds. Exceeding the ceiling "
             "returns HTTP 429 with a Retry-After header. Bulk export endpoints are "
             "metered separately at 20 requests per minute."},
    {"id": "eng-postmortem-oct", "source": "confluence", "dup_group": None,
     "text": "Incident postmortem. A schema migration dropped the index backing the "
             "session lookup, which raised p99 login latency from 180 milliseconds to 9 "
             "seconds for 41 minutes. Rollback was delayed because the migration had no "
             "down step. Action: block deploys of migrations without a tested rollback."},
]


def _cli() -> None:
    docs = SAMPLE_DOCS
    print(f"corpus: {len(docs)} documents\n")

    full = find_duplicates(docs)
    truth = truth_pairs(docs)

    print(f"blocking: scored {full['n_candidate_pairs']} candidate pairs "
          f"instead of {full['n_all_pairs']} all-pairs "
          f"({100 - 100 * full['n_candidate_pairs'] / max(1, full['n_all_pairs']):.0f}% skipped)\n")

    kinds = {(p["a"], p["b"]): p["kind"] for p in full["pairs"]}
    print("clusters found:")
    for c in full["clusters"]:
        keep = docs[c["keep"]]["id"]
        drop = ", ".join(docs[i]["id"] for i in c["drop"])
        via = sorted({k for (a, b), k in kinds.items() if a in c["members"] and b in c["members"]})
        print(f"  keep {keep:<28} drop: {drop:<42} via {'+'.join(via)}")

    print("\nrejected pairs (similar enough to consider, gate said no):")
    for r in full["rejected"][:6]:
        print(f"  {docs[r['a']]['id']:<20} ~ {docs[r['b']]['id']:<20} "
              f"cos={r['cosine']:.2f} con={r['containment']:.2f}  {'; '.join(r['reasons'])}")

    print(f"\nfragments held out (too short to judge): "
          f"{', '.join(docs[i]['id'] for i in full['fragments']) or 'none'}")

    cos_only = find_duplicates(docs, contain_threshold=0.0, subset_threshold=1.01,
                              numeric_threshold=0.0)
    no_veto = find_duplicates(docs, numeric_threshold=0.0)
    print("\nevaluation vs labelled duplicate groups")
    ablations = (
        ("cosine only", cos_only),
        ("+ containment", no_veto),
        ("+ numeric veto", full),
    )
    for name, res in ablations:
        m = evaluate(cluster_pairs(res), truth)
        print(f"  {name:<18} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
              f"(fp={m['fp']}, fn={m['fn']})")
        for a, b in m["false_merges"]:
            print(f"      false merge: {docs[a]['id']} + {docs[b]['id']}")

    plan = dedup_plan(docs, full)
    print(f"\nplan: keep {plan['keep_count']}, drop {plan['drop_count']} "
          f"({plan['pct_index_saved']}% of index tokens, "
          f"${plan['embedding_cost_saved']:.6f} to re-embed)")

    q = "how many business days for a refund to reach the original payment method"
    idx = full["index"]
    dropped = {d["index"] for d in plan["dropped"]}
    before = retrieve(idx, q, k=3)
    after = retrieve(idx, q, k=3, exclude=dropped)
    print(f"\nretrieval for {q!r}")
    print("  before:", ", ".join(f"{docs[i]['id']}({s})" for i, s in before))
    print("  after: ", ", ".join(f"{docs[i]['id']}({s})" for i, s in after))
    print(f"  distinct answers in top-3: {distinct_answers(full, before)}"
          f" -> {distinct_answers(full, after)}"
          f"  | redundant slots: {redundant_slots(full, before)} -> {redundant_slots(full, after)}")


if __name__ == "__main__":
    _cli()
