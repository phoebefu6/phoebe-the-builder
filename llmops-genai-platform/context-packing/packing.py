"""The fact was in the context, and the model did not read it.

A RAG answer is generated from a WINDOW: an instruction block, the retrieved chunks the packer kept under a
token budget, in the order the packer chose, then the question. The eval column that describes the window is
context recall - the share of the facts the answer needs that are physically present in it. The model does not
read every position equally well. Liu et al. (2023, "Lost in the Middle") measured a U: facts at the start and
the end of the context are recovered far more often than facts in the middle, and every position gets worse as
the context grows. An instruction placed once at the top is obeyed less often the more tokens sit between it and
the question. So three numbers describe one window and they are three quantities:

    context_recall    share of needed facts PRESENT in the window                (the dashboard)
    effective_recall  share of needed facts present AND at a position that is read
    answer_rate       P(every needed fact is read AND the instruction is obeyed)  (the user)

The query log is declared and seeded (Q queries, each with N candidate chunks, their token lengths, the ranks
at which the needed facts were retrieved). Given the log, every rate is an EXACT expectation over the reading
model - there is no sampling in the numbers, only in the calibration that checks them.

The reading model, declared:

    read(u, L)   = clip(curve(u) - LENGTH_SLOPE * (L - L0) / 1000, READ_FLOOR, READ_CEIL)
                   u = relative position of the chunk among the packed chunks (first = 0, last = 1),
                   L = total tokens in the window; curve is the quadratic through the three Liu points
    comply(d)    = clip(COMPLY_AT_ZERO - COMPLY_SLOPE * d / 1000, COMPLY_FLOOR, 1)
                   d = tokens between the nearest copy of the instruction and the question
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED = 192
MC_REPS = 200  # replicates per query; x 2,000 queries = 400,000 trials per rate

# ---------------------------------------------------------------------------------------------- the log
Q = 2_000          # queries in the declared log
N_CAND = 30        # candidate chunks the retriever returns per query, best first
LEN_RANGE = (120, 520)                       # chunk length in tokens, uniform
FACTS_NEEDED = {1: 0.60, 2: 0.30, 3: 0.10}   # how many distinct facts the answer needs
RANK_DECAY = 0.82                            # P(gold chunk at rank r) proportional to RANK_DECAY ** r
MISS_RATE = 0.06                             # the fact's chunk is not in the top N_CAND at all
INSTRUCTION_TOKENS = 180
QUESTION_TOKENS = 60

# ---------------------------------------------------------------------------------------------- reading model
L0 = 4_000
LIU_POINTS = (0.76, 0.54, 0.66)   # read probability at first / middle / last position when L = L0
LENGTH_SLOPE = 0.025              # every position loses this per 1,000 tokens of window beyond L0
READ_FLOOR, READ_CEIL = 0.05, 0.98
COMPLY_AT_ZERO, COMPLY_SLOPE, COMPLY_FLOOR = 0.98, 0.03, 0.30

# A policy is a budget, an order, and where the instruction goes.
#   order:  "rank" = best first (fill in retriever order)   "reorder" = best at both edges, worst in the middle
#           "reverse" = best last, next to the question
#   instruction: "top" once at the top of the window;  "both" at the top and again just before the question
POLICIES: Dict[str, Dict] = {
    "v1 rank, 4k":            {"budget": 4_000, "order": "rank", "instruction": "top"},
    "v2 rank, 12k":           {"budget": 12_000, "order": "rank", "instruction": "top"},
    "v3 reorder, 4k":         {"budget": 4_000, "order": "reorder", "instruction": "top"},
    "v4 reorder+repeat, 4k":  {"budget": 4_000, "order": "reorder", "instruction": "both"},
}


def make_log(q: int = Q, seed: int = SEED) -> List[Dict]:
    """The declared query log. Each query: candidate lengths (best rank first), and the rank of each needed
    fact's chunk (None when the retriever missed it). Ranks of distinct facts are distinct."""
    rng = np.random.default_rng(seed)
    ks = list(FACTS_NEEDED)
    pk = np.array([FACTS_NEEDED[k] for k in ks])
    pr = RANK_DECAY ** np.arange(1, N_CAND + 1)
    pr = pr / pr.sum()
    log = []
    for _ in range(q):
        lens = rng.integers(LEN_RANGE[0], LEN_RANGE[1] + 1, N_CAND).tolist()
        k = int(rng.choice(ks, p=pk))
        ranks: List[Optional[int]] = []
        taken: set = set()
        for _f in range(k):
            if rng.random() < MISS_RATE:
                ranks.append(None)
                continue
            r = int(rng.choice(N_CAND, p=pr)) + 1
            while r in taken:
                r = int(rng.choice(N_CAND, p=pr)) + 1
            taken.add(r)
            ranks.append(r)
        log.append({"lens": lens, "gold": ranks})
    return log


# ---------------------------------------------------------------------------------------------- reading model
def curve(u: float, pts: Tuple[float, float, float] = LIU_POINTS) -> float:
    """Quadratic through (0, first), (0.5, middle), (1, last)."""
    a, b, c = pts
    return a * (1 - u) * (1 - 2 * u) + 4 * b * u * (1 - u) + c * u * (2 * u - 1)


def read(u: float, total_tokens: int) -> float:
    p = curve(u) - LENGTH_SLOPE * (total_tokens - L0) / 1000.0
    return min(READ_CEIL, max(READ_FLOOR, p))


def comply(distance_tokens: int) -> float:
    p = COMPLY_AT_ZERO - COMPLY_SLOPE * distance_tokens / 1000.0
    return min(1.0, max(COMPLY_FLOOR, p))


# ---------------------------------------------------------------------------------------------- the packer
def select(lens: Sequence[int], budget: int, instruction: str) -> List[int]:
    """Ranks (1-based) kept: walk the candidates best first, keep each that still fits. Fixed costs come off
    the budget first."""
    fixed = INSTRUCTION_TOKENS * (2 if instruction == "both" else 1) + QUESTION_TOKENS
    room = budget - fixed
    kept = []
    for i, ln in enumerate(lens):
        if ln <= room:
            kept.append(i + 1)
            room -= ln
    return kept


def arrange(kept: Sequence[int], order: str) -> List[int]:
    """The order the kept ranks appear in the window, top to bottom."""
    k = list(kept)
    if order == "rank":
        return k
    if order == "reverse":
        return k[::-1]
    if order == "reorder":
        front, back = k[0::2], k[1::2]
        return front + back[::-1]
    raise ValueError(f"unknown order {order!r}")


def window(lens: Sequence[int], policy: Dict) -> Dict:
    """Everything about the packed window: the layout, each kept rank's relative position, total tokens, and
    the distance from the nearest instruction copy to the question."""
    kept = select(lens, policy["budget"], policy["instruction"])
    layout = arrange(kept, policy["order"])
    chunk_tokens = sum(lens[r - 1] for r in layout)
    copies = 2 if policy["instruction"] == "both" else 1
    total = INSTRUCTION_TOKENS * copies + chunk_tokens + QUESTION_TOKENS
    pos: Dict[int, float] = {}
    start = 0
    span = chunk_tokens - (lens[layout[-1] - 1] if layout else 0)   # first chunk starts at 0, last at span
    for r in layout:
        pos[r] = start / span if span > 0 else 0.0
        start += lens[r - 1]
    distance = 0 if copies == 2 else chunk_tokens
    return {"kept": kept, "layout": layout, "pos": pos, "total": total, "distance": distance}


# ---------------------------------------------------------------------------------------------- exact rates
def score_query(entry: Dict, policy: Dict) -> Dict:
    """Exact per-query expectations under the reading model."""
    w = window(entry["lens"], policy)
    present = [r is not None and r in w["pos"] for r in entry["gold"]]
    reads = [read(w["pos"][r], w["total"]) if ok else 0.0 for r, ok in zip(entry["gold"], present)]
    c = comply(w["distance"])
    all_present = all(present)
    all_read = math.prod(reads) if all_present else 0.0
    return {
        "present_sum": sum(present),
        "read_sum": sum(reads),
        "answer": c * all_read,
        "tokens": w["total"],
        "comply": c,
        # failure decomposition: the three parts sum to 1 - answer
        "fail_missing": 0.0 if all_present else 1.0,
        "fail_unread": (1.0 - all_read) if all_present else 0.0,
        "fail_ignored": all_read * (1.0 - c),
        "gold_positions": [w["pos"][r] for r, ok in zip(entry["gold"], present) if ok],
        "n_kept": len(w["kept"]),
    }


def evaluate(policy: Dict, log: Sequence[Dict]) -> Dict:
    """Recalls are pooled over FACTS (a query needing three facts contributes three); the answer rate and the
    failure classes are per QUERY. The first version averaged per-query recalls and the raw simulation pooled
    facts, so a quantity with no randomness in it read as outside its own interval - the two readers of one
    number disagreed. One definition now, and the calibration no longer checks the deterministic column."""
    rows = [score_query(e, policy) for e in log]
    n = len(rows)
    facts = sum(len(e["gold"]) for e in log)
    gp = [u for r in rows for u in r["gold_positions"]]
    mid = sum(1 / 3 <= u <= 2 / 3 for u in gp) / len(gp) if gp else 0.0
    present = sum(r["present_sum"] for r in rows)
    read_sum = sum(r["read_sum"] for r in rows)
    return {
        "context_recall": present / facts,
        "effective_recall": read_sum / facts,
        "answer_rate": sum(r["answer"] for r in rows) / n,
        "compliance": sum(r["comply"] for r in rows) / n,
        "tokens_per_query": sum(r["tokens"] for r in rows) / n,
        "chunks_per_query": sum(r["n_kept"] for r in rows) / n,
        "fail_missing": sum(r["fail_missing"] for r in rows) / n,
        "fail_unread": sum(r["fail_unread"] for r in rows) / n,
        "fail_ignored": sum(r["fail_ignored"] for r in rows) / n,
        "gold_in_middle_third": mid,
        "read_given_present": read_sum / present if present else 0.0,
        "facts": facts,
    }


def sweep(budgets: Sequence[int], log: Sequence[Dict], order: str = "rank", instruction: str = "top") -> List[Dict]:
    out = []
    for b in budgets:
        e = evaluate({"budget": b, "order": order, "instruction": instruction}, log)
        out.append({"budget": b, **{k: e[k] for k in ("context_recall", "effective_recall", "answer_rate",
                                                      "compliance", "tokens_per_query")}})
    return out


def same_set_different_order(log: Sequence[Dict], a: Dict, b: Dict) -> Dict:
    """Two policies with the same budget and instruction keep the SAME chunks for every query; only the order
    differs. Everything the retrieval log records is identical; the answer rate is not."""
    assert a["budget"] == b["budget"] and a["instruction"] == b["instruction"]
    same = all(window(e["lens"], a)["kept"] == window(e["lens"], b)["kept"] for e in log)
    ea, eb = evaluate(a, log), evaluate(b, log)
    return {"identical_sets": same, "context_recall_gap": eb["context_recall"] - ea["context_recall"],
            "tokens_gap": eb["tokens_per_query"] - ea["tokens_per_query"],
            "answer_rate_gap": eb["answer_rate"] - ea["answer_rate"],
            "gold_in_middle_a": ea["gold_in_middle_third"], "gold_in_middle_b": eb["gold_in_middle_third"]}


def by_rank(policy: Dict, log: Sequence[Dict], max_rank: int = 10) -> List[Dict]:
    """For a needed fact retrieved at rank r: how often it is kept, and how often it is read given kept."""
    out = []
    for r in range(1, max_rank + 1):
        kept = read_p = 0.0
        n = 0
        for e in log:
            if r in e["gold"]:
                n += 1
                w = window(e["lens"], policy)
                if r in w["pos"]:
                    kept += 1
                    read_p += read(w["pos"][r], w["total"])
        out.append({"rank": r, "n": n, "kept": kept / n if n else 0.0, "read_given_kept": read_p / kept if kept else 0.0,
                    "read": read_p / n if n else 0.0})
    return out


# ---------------------------------------------------------------------------------------------- calibration
def wilson(k: int, n: int, z: float = 2.5758) -> Tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def simulate(policy: Dict, log: Sequence[Dict], reps: int = MC_REPS, seed: int = SEED) -> Dict[str, Tuple[int, int]]:
    """Raw simulation: for every query and replicate, draw whether each present fact is read and whether the
    instruction is obeyed as Bernoullis. Returns (successes, trials) for a Wilson check of the two exact rates
    that have randomness in them; the present count is deterministic and is asserted instead."""
    rng = np.random.default_rng(seed)
    hits = eff = ans = 0
    facts = 0
    for e in log:
        w = window(e["lens"], policy)
        present = np.array([r is not None and r in w["pos"] for r in e["gold"]])
        p = np.array([read(w["pos"][r], w["total"]) if ok else 0.0 for r, ok in zip(e["gold"], present)])
        c = comply(w["distance"])
        rd = rng.random((reps, len(p))) < p            # present-and-read draws (p is 0 where absent)
        ok = rng.random(reps) < c
        facts += reps * len(p)
        hits += reps * int(present.sum())
        eff += int(rd.sum())
        ans += int((rd.all(axis=1) & ok).sum())
    assert hits == reps * sum(round(evaluate_present(e, policy)) for e in log)
    return {"effective_recall": (eff, facts), "answer_rate": (ans, reps * len(log))}


def evaluate_present(entry: Dict, policy: Dict) -> int:
    """Needed facts present in the window - deterministic, so it is asserted rather than interval-checked."""
    w = window(entry["lens"], policy)
    return sum(r is not None and r in w["pos"] for r in entry["gold"])


def calibrate(log: Sequence[Dict], reps: int = MC_REPS, seed: int = SEED,
              policies: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    rows = []
    for s, (name, pol) in enumerate((policies or POLICIES).items()):
        ex = evaluate(pol, log)
        sim = simulate(pol, log, reps, seed + s)
        for key, (k, n) in sim.items():
            lo, hi = wilson(k, n)
            rows.append({"policy": name, "quantity": key, "exact": ex[key], "mc": k / n, "inside_99": bool(lo <= ex[key] <= hi)})
    return rows


# ---------------------------------------------------------------------------------------------- app parsing
def parse_policy(text: str) -> Dict:
    """`budget, order, instruction` - e.g. `4000, reorder, both`. Order in rank/reorder/reverse, instruction in
    top/both."""
    parts = [p.strip().lower() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"need 3 comma-separated fields (budget, order, instruction), got {len(parts)}: {text!r}")
    b, order, instr = parts
    try:
        budget = int(b.replace("k", "000").replace("_", ""))
    except ValueError:
        raise ValueError(f"budget did not parse: {b!r}")
    fixed = 2 * INSTRUCTION_TOKENS + QUESTION_TOKENS
    if budget < fixed + LEN_RANGE[0]:
        raise ValueError(f"budget {budget} cannot hold the instruction, the question and one chunk ({fixed + LEN_RANGE[0]})")
    if order not in ("rank", "reorder", "reverse"):
        raise ValueError(f"order must be rank, reorder or reverse, got {order!r}")
    if instr not in ("top", "both"):
        raise ValueError(f"instruction must be top or both, got {instr!r}")
    return {"budget": budget, "order": order, "instruction": instr}
