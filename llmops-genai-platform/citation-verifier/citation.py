"""Citation support: does the passage a claim points at actually say that?

A declared corpus (12 passages in 3 documents), a declared claim generator (one true fact, one of six edits,
one of four citation choices, verbatim or paraphrased), and five checkers that see only text. Truth is
structural: a claim is SUPPORTED when the cited passage states the same (entity, attribute, value, scope,
polarity). The generator has a finite discrete space, so every claim it can produce is enumerated with its
probability and each checker's rates are exact; a raw simulation only confirms them.

Seam with `hallucination-checker` (Day 88): that tool asks whether the claim is anywhere in the retrieved
context. This one asks whether it is in the passage the citation names. The context-level checker here
(`context_lexical`) is that method, as a baseline.
"""

from __future__ import annotations

import functools
import itertools
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED = 190
MC_REPS = 200_000
TAU = 0.4

# (doc, entity, attr, value, scope)
FACTS: List[Tuple[str, str, str, str, str]] = [
    ("billing", "Basic plan", "costs", "$12 per month", "when billed annually"),
    ("billing", "Pro plan", "costs", "$29 per month", "when billed annually"),
    ("billing", "Basic plan", "includes", "2 seats", ""),
    ("billing", "Pro plan", "includes", "5 seats", ""),
    ("refunds", "Basic plan", "offers refunds within", "14 days", "for first-time purchases"),
    ("refunds", "Pro plan", "offers refunds within", "30 days", "for first-time purchases"),
    ("refunds", "Enterprise plan", "offers refunds within", "60 days", "under the master agreement"),
    ("refunds", "Add-on packs", "offers refunds within", "7 days", "if unused"),
    ("support", "Basic plan", "gets a support response within", "48 hours", "on business days"),
    ("support", "Pro plan", "gets a support response within", "8 hours", "on business days"),
    ("support", "Enterprise plan", "gets a support response within", "1 hour", "around the clock"),
    ("support", "Add-on packs", "gets a support response within", "48 hours", "on business days"),
]
NEG = {"costs": "does not cost", "includes": "does not include",
       "offers refunds within": "does not offer refunds within",
       "gets a support response within": "does not get a support response within"}
PARA = {"costs": "is priced at", "includes": "comes with", "offers refunds within": "can be refunded for up to",
        "gets a support response within": "has a support turnaround of"}
PARA_NEG = {"costs": "is not priced at", "includes": "does not come with",
            "offers refunds within": "is not refundable within",
            "gets a support response within": "does not have a support turnaround of"}
PARA_SCOPE = {"when billed annually": "on the annual billing cycle", "for first-time purchases": "on a first purchase",
              "under the master agreement": "per the master agreement", "if unused": "as long as they are unused",
              "on business days": "during business hours", "around the clock": "at any hour", "": ""}
NOVEL = {"costs": "$19 per month", "includes": "3 seats", "offers refunds within": "45 days",
         "gets a support response within": "24 hours"}
OVERGEN = "on every plan"

EDITS = ["none", "polarity", "scope", "value_near", "value_far", "entity"]
CITES = ["source", "neighbour", "unrelated", "none"]
CHECKERS = ["has_citation", "context_lexical", "cited_lexical", "cited_lexical+numbers",
            "cited_lexical+numbers+negation"]
MODELS = {
    "v1 cites when sure": {"edit": [0.80, 0.04, 0.05, 0.04, 0.02, 0.05], "cite": [0.62, 0.08, 0.04, 0.26], "para": 0.35},
    "v2 always cite": {"edit": [0.80, 0.04, 0.05, 0.04, 0.02, 0.05], "cite": [0.52, 0.32, 0.14, 0.02], "para": 0.35},
}
# sklearn's English list drops not / no / cannot, so a lexical checker never sees polarity. Kept on purpose.
STOP = {"a", "an", "the", "is", "are", "of", "for", "on", "in", "at", "to", "per", "with", "and", "as", "they",
        "be", "up", "any", "every", "does", "do", "not", "no", "can", "cannot", "has", "have", "its", "it", "this",
        "that", "if", "when", "under", "within", "after", "during", "long", "first"}
NEGATORS = {"not", "no", "cannot", "never"}
Fact = Tuple[str, str, str, str, bool]  # entity, attr, value, scope, positive


def render(f: Fact, para: bool = False) -> str:
    entity, attr, value, scope, pos = f
    verb = (PARA[attr] if para else attr) if pos else (PARA_NEG[attr] if para else NEG[attr])
    sc = PARA_SCOPE.get(scope, scope) if para else scope
    return f"{entity} {verb} {value}" + (f" {sc}" if sc else "") + "."


TUPLES: List[Fact] = [(e, a, v, s, True) for _, e, a, v, s in FACTS]
PASSAGES: List[str] = [render(t) for t in TUPLES]
DOC_OF: List[str] = [f[0] for f in FACTS]
TRUE_SET = set(TUPLES)


def sibling(i: int) -> int:
    """The next passage with the same attribute (cyclic) - the neighbour a model confuses it with."""
    attr = FACTS[i][2]
    others = [j for j in range(len(FACTS)) if j != i and FACTS[j][2] == attr]
    return min(others, key=lambda j: (j - i) % len(FACTS))


def edit_claim(i: int, edit: str) -> Fact:
    e, a, v, s, _ = TUPLES[i]
    if edit == "polarity":
        return (e, a, v, s, False)
    if edit == "scope":
        return (e, a, v, OVERGEN, True)
    if edit == "value_near":
        return (e, a, TUPLES[sibling(i)][2], s, True)
    if edit == "value_far":
        return (e, a, NOVEL[a], s, True)
    if edit == "entity":
        return (TUPLES[sibling(i)][0], a, v, s, True)
    return TUPLES[i]


def cited_index(i: int, cite: str) -> Optional[int]:
    if cite == "source":
        return i
    if cite == "neighbour":
        return sibling(i)
    if cite == "unrelated":
        return (i + 4) % len(FACTS)
    return None


def _strip(text: str) -> str:
    """Citation markers are not content: a checker that reads [1] as the number 1 fails every cited claim."""
    return re.sub(r"\[\d+\]", " ", text)


def _content(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", _strip(text).lower())
    return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in toks if t not in STOP]


def coverage(claim: str, text: str) -> float:
    c, t = _content(claim), set(_content(text))
    return sum(w in t for w in c) / len(c) if c else 0.0


def numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", _strip(text)))


def negations(text: str) -> int:
    return sum(t in NEGATORS for t in re.findall(r"[a-z']+", text.lower()))


def check(claim: str, cited: Optional[str], context: str, tau: float = TAU) -> Dict[str, bool]:
    """Five text-only checkers. `cited` is the passage the claim's [n] names, or None."""
    cov = coverage(claim, cited) if cited is not None else 0.0
    lex = cited is not None and cov >= tau
    num = lex and numbers(claim) <= numbers(cited or "")
    return {"has_citation": cited is not None,
            "context_lexical": coverage(claim, context) >= tau and numbers(claim) <= numbers(context),
            "cited_lexical": lex, "cited_lexical+numbers": num,
            "cited_lexical+numbers+negation": num and negations(claim) == negations(cited or "")}


@functools.lru_cache(maxsize=None)
def build(i: int, edit: str, cite: str, para: bool) -> Dict[str, object]:
    """One claim as the generator would emit it, with its truth labels and checker verdicts (576 distinct)."""
    tup = edit_claim(i, edit)
    j = cited_index(i, cite)
    claim = render(tup, para) + (f" [{j + 1}]" if j is not None else "")
    docs = {DOC_OF[i]} | ({DOC_OF[j]} if j is not None else set())
    context = " ".join(p for p, d in zip(PASSAGES, DOC_OF) if d in docs)
    supported = j is not None and TUPLES[j] == tup
    is_true = tup in TRUE_SET
    klass = ("supported" if supported else "uncited true" if is_true and j is None
             else "misattributed" if is_true else edit)
    return {"claim": claim, "cited": j, "supported": supported, "true": is_true, "klass": klass,
            "edit": edit, "cite": cite, "para": para, "cov": coverage(claim, PASSAGES[j]) if j is not None else 0.0,
            "checks": check(claim, PASSAGES[j] if j is not None else None, context)}


def enumerate_claims(model: Dict) -> List[Dict[str, object]]:
    rows = []
    for i, (ei, e), (ci, c), para in itertools.product(range(len(FACTS)), enumerate(EDITS), enumerate(CITES),
                                                       (False, True)):
        r = dict(build(i, e, c, para))
        r["p"] = model["edit"][ei] * model["cite"][ci] * (model["para"] if para else 1 - model["para"]) / len(FACTS)
        rows.append(r)
    return rows


def summarise(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    P = np.array([r["p"] for r in rows])
    sup = np.array([r["supported"] for r in rows])
    out: Dict[str, object] = {"p_supported": float(P[sup].sum()), "p_true": float(sum(r["p"] for r in rows if r["true"])),
                              "p_cited": float(sum(r["p"] for r in rows if r["cited"] is not None)), "scorers": {}}
    for k in CHECKERS:
        ok = np.array([r["checks"][k] for r in rows])
        passed_bad = Counter()
        for r, p, s, o in zip(rows, P, sup, ok):
            if o and not s:
                passed_bad[r["klass"]] += p
        out["scorers"][k] = {"reported": float(P[ok].sum()), "false_pass": float(P[ok & ~sup].sum() / P[~sup].sum()),
                             "false_fail": float(P[~ok & sup].sum() / P[sup].sum()),
                             "bad_among_passes": float(P[ok & ~sup].sum() / P[ok].sum()),
                             "passed_bad_by_class": {c: float(v) for c, v in passed_bad.most_common()}}
    return out


def evaluate() -> Dict[str, Dict[str, object]]:
    return {name: summarise(enumerate_claims(m)) for name, m in MODELS.items()}


def threshold_sweep(model: Dict, taus: Sequence[float]) -> List[Dict[str, float]]:
    """cited_lexical at each tau: how often it fails a supported paraphrase vs passes a polarity flip."""
    rows = enumerate_claims(model)
    out = []
    for tau in taus:
        para_sup = [(r["p"], r["cov"] >= tau) for r in rows if r["supported"] and r["para"]]
        flips = [(r["p"], r["cov"] >= tau) for r in rows if r["edit"] == "polarity" and r["cite"] == "source"]
        out.append({"tau": tau,
                    "fails_supported_paraphrase": 1 - sum(p for p, ok in para_sup if ok) / sum(p for p, _ in para_sup),
                    "passes_polarity_flip": sum(p for p, ok in flips if ok) / sum(p for p, _ in flips)})
    return out


def flip_vs_paraphrase_order(model: Dict) -> Dict[str, float]:
    """Pair-weighted P(a source-cited polarity flip scores above / level with a supported paraphrase)."""
    rows = enumerate_claims(model)
    flips = [(r["p"], r["cov"]) for r in rows if r["edit"] == "polarity" and r["cite"] == "source"]
    paras = [(r["p"], r["cov"]) for r in rows if r["supported"] and r["para"]]
    tot = sum(p for p, _ in flips) * sum(p for p, _ in paras)
    above = sum(pf * pp * (cf > cp + 1e-12) for pf, cf in flips for pp, cp in paras) / tot
    tied = sum(pf * pp * (abs(cf - cp) <= 1e-12) for pf, cf in flips for pp, cp in paras) / tot
    verbatim = [(r["p"], r["cov"]) for r in rows if r["edit"] == "polarity" and r["cite"] == "source" and not r["para"]]
    return {"flip_above": above, "tied": tied, "flip_below": 1 - above - tied,
            "verbatim_flip_coverage_min": min(c for _, c in verbatim)}


def simulate(model: Dict, n: int, seed: int = SEED) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    fi = rng.integers(0, len(FACTS), n)
    ei = rng.choice(len(EDITS), n, p=model["edit"])
    ci = rng.choice(len(CITES), n, p=model["cite"])
    pa = rng.random(n) < model["para"]
    acc = Counter()
    for i, e, c, b in zip(fi, ei, ci, pa):
        r = build(int(i), EDITS[e], CITES[c], bool(b))
        acc["supported"] += r["supported"]
        for k, v in r["checks"].items():
            acc[k] += v
    return {k: acc[k] / n for k in ["supported"] + CHECKERS}


def wilson(p: float, n: int, z: float = 2.576) -> Tuple[float, float]:
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def calibrate(n: int = MC_REPS, seed: int = SEED) -> List[Dict[str, object]]:
    out = []
    for name, m in MODELS.items():
        ex = summarise(enumerate_claims(m))
        mc = simulate(m, n, seed)
        for k in ["supported"] + CHECKERS:
            e = ex["p_supported"] if k == "supported" else ex["scorers"][k]["reported"]
            lo, hi = wilson(mc[k], n)
            out.append({"model": name, "quantity": k, "exact": e, "mc": mc[k], "inside_99": bool(lo <= e <= hi)})
    return out


def parse_claims(passages_text: str, claims_text: str) -> Tuple[List[str], List[Tuple[str, Optional[int]]]]:
    """Passages one per line (an optional leading [n] is stripped); claims one per line, citing with [n]."""
    passages = [re.sub(r"^\s*\[\d+\]\s*", "", ln).strip() for ln in passages_text.strip().splitlines() if ln.strip()]
    if not passages:
        raise ValueError("paste at least one passage")
    claims = []
    for ln in [x.strip() for x in claims_text.strip().splitlines() if x.strip()]:
        m = re.search(r"\[(\d+)\]", ln)
        j = int(m.group(1)) - 1 if m else None
        if j is not None and not 0 <= j < len(passages):
            raise ValueError(f"citation [{j + 1}] names a passage that is not there: {ln[:60]!r}")
        claims.append((ln, j))
    if not claims:
        raise ValueError("paste at least one claim")
    return passages, claims


def audit(passages: List[str], claims: List[Tuple[str, Optional[int]]], tau: float = TAU) -> List[Dict[str, object]]:
    """Checker verdicts on the user's own claims, with the disagreements that point at a wrong citation."""
    context = " ".join(passages)
    out = []
    for claim, j in claims:
        cited = passages[j] if j is not None else None
        ch = check(claim, cited, context, tau)
        notes = []
        if j is None:
            notes.append("no citation")
        else:
            if ch["context_lexical"] and not ch["cited_lexical"]:
                notes.append("in the context, not in the cited passage")
            if ch["cited_lexical"] and not ch["cited_lexical+numbers"]:
                notes.append("a number the cited passage does not hold: " + ", ".join(sorted(numbers(claim) - numbers(cited or ""))))
            if ch["cited_lexical+numbers"] and not ch["cited_lexical+numbers+negation"]:
                notes.append("polarity differs from the cited passage")
            if ch["cited_lexical+numbers+negation"]:
                notes.append("lexically consistent - entity and scope are NOT checked by any of these")
        out.append({"claim": claim, "cited": j, "coverage": coverage(claim, cited) if cited else 0.0, "checks": ch,
                    "notes": notes})
    return out
