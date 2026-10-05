"""An index is a snapshot, the source keeps moving, and the queries land on the part that moves.

A RAG index holds one version of every document. Each source document changes as a Poisson process; some
changes alter a fact a chunk carries (fact-bearing), the rest are cosmetic. A reindex policy gives every document
a period T: it is re-read on a schedule, so at a random moment it was last indexed a uniform phase u ~ U(0, T)
ago, and the chunk served is STALE when at least one fact-bearing change landed in that window.

Everything is closed-form under the declared model and a raw simulation only checks it:

    stale(mu, T)  = 1 - (1 - exp(-mu T)) / (mu T)      mu = fact-bearing change rate, phase uniform in [0, T)
    stale_at(mu, d) = 1 - exp(-mu d)                   d days after the last reindex

Three numbers are reported for a policy and they are three different quantities:

    hash_freshness   doc-weighted share of documents whose indexed bytes equal the source (ALL changes count)
    fact_freshness   doc-weighted share of documents whose indexed facts equal the source
    answer_staleness query-weighted share of served answers whose chunk carries a changed fact

The dashboard shows the first. The user hits the third.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED = 191
MC_REPS = 200_000

# One declared corpus: three document classes. change_every = mean days between ANY edit; fact_share = the
# share of edits that change a fact a chunk carries; query_share = the share of served answers that retrieve
# from this class. Pricing and limits pages are few, edited often, and asked about constantly.
CLASSES: List[Dict] = [
    {"name": "hot", "docs": 10, "change_every": 14.0, "fact_share": 0.6, "query_share": 0.50,
     "what": "pricing, limits, policy"},
    {"name": "warm", "docs": 40, "change_every": 45.0, "fact_share": 0.5, "query_share": 0.35,
     "what": "how-to guides, release notes"},
    {"name": "cold", "docs": 150, "change_every": 365.0, "fact_share": 0.3, "query_share": 0.15,
     "what": "archive, legal, reference"},
]

# A policy is a reindex period in days per class.
POLICIES: Dict[str, Dict[str, float]] = {
    "v1 monthly full": {"hot": 30.0, "warm": 30.0, "cold": 30.0},
    "v2a weekly full": {"hot": 7.0, "warm": 7.0, "cold": 7.0},
    "v2b tiered 1/7/90": {"hot": 1.0, "warm": 7.0, "cold": 90.0},
}

# Twelve declared facts for the similarity question: a chunk is the sentence with the value, a query asks
# about the attribute and never contains the value. (entity, attribute phrase, old value, new value)
FACTS: List[Tuple[str, str, str, str]] = [
    ("Pro plan", "includes", "5 seats", "8 seats"),
    ("Pro plan", "allows API calls per minute of", "600", "1200"),
    ("Pro plan", "offers refunds within", "30 days", "two weeks"),
    ("Pro plan", "stores logs for", "90 days", "30 days"),
    ("Basic plan", "includes", "2 seats", "3 seats"),
    ("Basic plan", "allows API calls per minute of", "60", "100"),
    ("Basic plan", "offers refunds within", "14 days", "7 days"),
    ("Basic plan", "stores logs for", "30 days", "7 days"),
    ("Enterprise plan", "includes", "50 seats", "unlimited seats"),
    ("Enterprise plan", "guarantees uptime of", "99.9 percent", "99.95 percent"),
    ("Enterprise plan", "responds to support tickets within", "4 hours", "1 hour"),
    ("Enterprise plan", "stores logs for", "365 days", "730 days"),
]


# ---------------------------------------------------------------------------------------------- closed form
def stale(mu: float, period: float) -> float:
    """P(served chunk carries a changed fact) when the doc is reindexed every `period` days."""
    x = mu * period
    if x == 0:
        return 0.0
    return 1.0 + math.expm1(-x) / x  # expm1: 1 - (1 - e^-x)/x cancels catastrophically for small x


def stale_at(mu: float, days: float) -> float:
    """P(stale) `days` after the last reindex."""
    return -math.expm1(-mu * days)


def rates(c: Dict) -> Tuple[float, float]:
    """(lambda, mu): rate of any edit, rate of fact-bearing edits, per day."""
    lam = 1.0 / c["change_every"]
    return lam, lam * c["fact_share"]


def evaluate(policy: Dict[str, float], classes: Sequence[Dict] = CLASSES) -> Dict:
    """The three numbers plus cost, exactly."""
    n = sum(c["docs"] for c in classes)
    hash_stale = fact_stale = answer = cost = 0.0
    per: Dict[str, Dict[str, float]] = {}
    for c in classes:
        lam, mu = rates(c)
        T = policy[c["name"]]
        h, f = stale(lam, T), stale(mu, T)
        per[c["name"]] = {"period": T, "hash_stale": h, "fact_stale": f}
        hash_stale += c["docs"] * h
        fact_stale += c["docs"] * f
        answer += c["query_share"] * f
        cost += c["docs"] / T
    return {"hash_freshness": 1.0 - hash_stale / n, "fact_freshness": 1.0 - fact_stale / n,
            "answer_staleness": answer, "reindex_per_day": cost, "per_class": per, "docs": n}


def cycle(policy: Dict[str, float], days: int = 30, classes: Sequence[Dict] = CLASSES) -> List[Dict]:
    """Answer staleness on each day of a cycle, for classes whose period is >= `days` (others are at their
    own phase average)."""
    out = []
    for d in range(days + 1):
        a = 0.0
        for c in classes:
            _, mu = rates(c)
            T = policy[c["name"]]
            a += c["query_share"] * (stale_at(mu, min(d, T)) if T >= days else stale(mu, T))
        out.append({"day": d, "answer_staleness": a})
    return out


def with_archive(extra_docs: int, policy: Dict[str, float], period: float = 30.0,
                 classes: Sequence[Dict] = CLASSES) -> Tuple[List[Dict], Dict[str, float]]:
    """Add documents nobody asks about: cold change profile, zero query share, reindexed every `period`."""
    cls = list(classes) + [{"name": "archive", "docs": extra_docs, "change_every": 365.0, "fact_share": 0.3,
                            "query_share": 0.0, "what": "added, never retrieved"}]
    return cls, {**policy, "archive": period}


def audit_sample(policy: Dict[str, float], sample_from: str, classes: Sequence[Dict] = CLASSES) -> float:
    """Expected stale share in an audit sample: `docs` draws documents uniformly, `queries` draws them in
    proportion to query share. Each is an unbiased estimate of a different number."""
    n = sum(c["docs"] for c in classes)
    tot = 0.0
    for c in classes:
        _, mu = rates(c)
        w = c["docs"] / n if sample_from == "docs" else c["query_share"]
        tot += w * stale(mu, policy[c["name"]])
    return tot


# ---------------------------------------------------------------------------------------------- similarity
def tokens(s: str) -> List[str]:
    return [t for t in s.lower().replace(",", " ").split() if t]


def jaccard(a: str, b: str) -> float:
    A, B = set(tokens(a)), set(tokens(b))
    return len(A & B) / len(A | B) if A | B else 0.0


def chunk(fact: Tuple[str, str, str, str], new: bool) -> str:
    e, attr, old, cur = fact
    return f"{e} {attr} {cur if new else old}."


def query(fact: Tuple[str, str, str, str]) -> str:
    e, attr, _, _ = fact
    return f"{e} {attr}"


def similarity_gap(facts: Sequence[Tuple[str, str, str, str]] = FACTS) -> Dict:
    """For each fact: similarity of the query to the stale chunk and to the fresh chunk. A gate that keeps
    every fresh chunk passes every stale chunk whose score is >= the lowest fresh score."""
    rows = []
    for f in facts:
        q = query(f)
        s_old, s_new = jaccard(q, chunk(f, False)), jaccard(q, chunk(f, True))
        rows.append({"query": q, "stale_sim": s_old, "fresh_sim": s_new, "gap": s_new - s_old})
    floor = min(r["fresh_sim"] for r in rows)
    passed = sum(r["stale_sim"] >= floor for r in rows)
    return {"rows": rows, "max_abs_gap": max(abs(r["gap"]) for r in rows),
            "stale_at_least_as_similar": sum(r["gap"] <= 0 for r in rows), "n": len(rows),
            "stale_passed_by_gate_keeping_all_fresh": passed / len(rows)}


# ---------------------------------------------------------------------------------------------- calibration
def wilson(k: int, n: int, z: float = 2.5758) -> Tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def simulate(policy: Dict[str, float], reps: int = MC_REPS, seed: int = SEED,
             classes: Sequence[Dict] = CLASSES) -> Dict:
    """Raw simulation of the model: a uniform phase, Poisson edits in it, each fact-bearing with prob
    fact_share; a served answer draws its class by query share. Returns counts for a Wilson check."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Tuple[int, int]] = {}
    for c in classes:
        lam, _ = rates(c)
        T = policy[c["name"]]
        u = rng.uniform(0, T, reps)
        edits = rng.poisson(lam * u)
        fact_edits = rng.binomial(edits, c["fact_share"])
        out[f"{c['name']} hash_stale"] = (int((edits > 0).sum()), reps)
        out[f"{c['name']} fact_stale"] = (int((fact_edits > 0).sum()), reps)
    shares = np.array([c["query_share"] for c in classes])
    pick = rng.choice(len(classes), size=reps, p=shares / shares.sum())
    k = 0
    for i, c in enumerate(classes):
        lam, _ = rates(c)
        m = int((pick == i).sum())
        u = rng.uniform(0, policy[c["name"]], m)
        k += int((rng.binomial(rng.poisson(lam * u), c["fact_share"]) > 0).sum())
    out["answer_staleness"] = (k, reps)
    return out


def calibrate(reps: int = MC_REPS, seed: int = SEED, policies: Optional[Dict[str, Dict[str, float]]] = None,
              classes: Sequence[Dict] = CLASSES) -> List[Dict]:
    pols = policies or {k: POLICIES[k] for k in ("v1 monthly full", "v2b tiered 1/7/90")}
    rows = []
    for s, (pname, pol) in enumerate(pols.items()):
        ex = evaluate(pol, classes)
        sim = simulate(pol, reps, seed + s, classes)
        for key, (k, n) in sim.items():
            if key == "answer_staleness":
                exact = ex["answer_staleness"]
            else:
                cname, what = key.split(" ")
                exact = ex["per_class"][cname][what]
            lo, hi = wilson(k, n)
            rows.append({"policy": pname, "quantity": key, "exact": exact, "mc": k / n,
                         "inside_99": bool(lo <= exact <= hi)})
    return rows


# ---------------------------------------------------------------------------------------------- app parsing
def parse_classes(text: str) -> List[Dict]:
    """`name, docs, change_every_days, fact_share, query_share` per line. Query shares are normalised."""
    out = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"need 5 comma-separated fields, got {len(parts)}: {line!r}")
        name, docs, ce, fs, qs = parts
        try:
            c = {"name": name, "docs": int(docs), "change_every": float(ce), "fact_share": float(fs),
                 "query_share": float(qs), "what": ""}
        except ValueError:
            raise ValueError(f"numbers did not parse on {line!r}")
        if c["docs"] <= 0 or c["change_every"] <= 0 or not 0 <= c["fact_share"] <= 1 or c["query_share"] < 0:
            raise ValueError(f"out of range on {line!r}: docs > 0, change_every > 0, 0 <= fact_share <= 1, "
                             "query_share >= 0")
        out.append(c)
    if not out:
        raise ValueError("need at least one document class")
    if len({c["name"] for c in out}) != len(out):
        raise ValueError("class names must be unique")
    tot = sum(c["query_share"] for c in out)
    if tot <= 0:
        raise ValueError("query shares sum to zero - nobody asks anything")
    for c in out:
        c["query_share"] /= tot
    return out


def parse_policy(text: str, classes: Sequence[Dict]) -> Dict[str, float]:
    """A single number (every class) or `name=days` pairs, comma separated, one per class."""
    text = text.strip()
    names = [c["name"] for c in classes]
    try:
        T = float(text)
        if T <= 0:
            raise ValueError
        return {n: T for n in names}
    except ValueError:
        pass
    pol: Dict[str, float] = {}
    for part in text.split(","):
        if "=" not in part:
            raise ValueError(f"expected name=days, got {part.strip()!r}")
        n, d = (p.strip() for p in part.split("=", 1))
        try:
            pol[n] = float(d)
        except ValueError:
            raise ValueError(f"days did not parse in {part.strip()!r}")
        if pol[n] <= 0:
            raise ValueError(f"period must be > 0 in {part.strip()!r}")
    missing = [n for n in names if n not in pol]
    if missing:
        raise ValueError(f"policy gives no period for {missing}")
    return pol
