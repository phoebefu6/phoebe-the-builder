"""When should a human take over? Escalation thresholds priced by the cost of the error they prevent.

A declared, constructed world: an AI support agent handles a day of tickets in three segments that differ in what a
wrong autonomous action costs. For each ticket the agent reports a confidence. A policy is one threshold per segment:
below it, the ticket goes to a human. Humans cost money per review, have a fixed daily capacity, make their own
mistakes, and make MORE of them as the queue fills. Escalations past capacity time out and go back to the agent.

Everything is exact: confidence lives on a grid of bins, and expected cost is a sum over bins. A Monte Carlo of whole
days checks the exact error rates inside 99% Wilson intervals.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

SEED = 195
MC_DAYS = 40
BINS = 40
CENTERS = (np.arange(BINS) + 0.5) / BINS  # stated confidence of each bin

# segment: share of tickets, cost of a wrong action ($), the model's underlying score x ~ Beta(a, b), calibration
# gamma, and inflate k. True P(correct | x) = x ** gamma; the confidence the agent REPORTS is x ** (1 / k).
# gamma = 1 and k = 1 is calibrated; gamma > 1 is overconfident; k > 1 reports higher numbers for the same skill.
SEGMENTS: Dict[str, Dict[str, float]] = {
    "refund": {"share": 0.15, "cost": 120.0, "a": 14.0, "b": 1.8, "gamma": 1.9, "inflate": 1.0},
    "account": {"share": 0.25, "cost": 40.0, "a": 16.0, "b": 1.8, "gamma": 1.3, "inflate": 1.0},
    "faq": {"share": 0.60, "cost": 3.0, "a": 20.0, "b": 1.8, "gamma": 1.0, "inflate": 1.0},
}
SEGS = list(SEGMENTS)

BASE: Dict[str, float] = {
    "tickets": 10_000.0,  # per day
    "review_cost": 6.0,  # $ of agent time per human review
    "capacity": 700.0,  # reviews a day the team can do
    "human_err": 0.03,  # human error rate with a calm queue
    "human_err_full": 0.12,  # human error rate at 100% utilisation
    "calm_until": 0.70,  # utilisation below which humans stay at human_err
}


FINE = 50


_CACHE: Dict[tuple, Dict[str, Dict[str, np.ndarray]]] = {}


def segment_tables(segments: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    seg = segments or SEGMENTS
    key = tuple((s, tuple(sorted(v.items()))) for s, v in seg.items())
    if key not in _CACHE:
        _CACHE[key] = _tables(seg)
    return _CACHE[key]


def _tables(seg: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Per segment: mass of REPORTED confidence on the bin grid, and true accuracy within each bin.

    The underlying score is integrated on a fine midpoint grid; each fine point reports x ** (1 / inflate) and is
    correct with probability x ** gamma, so a bin's accuracy is the mass-weighted mean over the points it receives."""
    x = (np.arange(BINS * FINE) + 0.5) / (BINS * FINE)
    out = {}
    for s, v in seg.items():
        logd = (v["a"] - 1) * np.log(x) + (v["b"] - 1) * np.log1p(-x)
        d = np.exp(logd - logd.max())
        d /= d.sum()
        reported = x ** (1 / v.get("inflate", 1.0))
        b = np.minimum((reported * BINS).astype(int), BINS - 1)
        pmf = np.bincount(b, weights=d, minlength=BINS)
        hit = np.bincount(b, weights=d * x ** v["gamma"], minlength=BINS)
        p_true = np.divide(hit, pmf, out=np.zeros(BINS), where=pmf > 0)
        out[s] = {"pmf": pmf, "p_true": p_true}
    return out


def human_error(util: float, prm: Dict[str, float]) -> float:
    """Calm until calm_until, then rising linearly to human_err_full at 100% and staying there."""
    lo, hi, k = prm["human_err"], prm["human_err_full"], prm["calm_until"]
    if util <= k:
        return lo
    return lo + (hi - lo) * min(1.0, (util - k) / (1 - k))


def evaluate(thr: Dict[str, float], prm: Optional[Dict[str, float]] = None,
             segments: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Exact expected day under a threshold policy: escalate when stated confidence < thr[segment]."""
    prm = prm or BASE
    seg = segments or SEGMENTS
    tab = segment_tables(seg)
    n = prm["tickets"]
    esc = {s: float(n * seg[s]["share"] * tab[s]["pmf"][CENTERS < thr[s]].sum()) for s in seg}
    e_total = sum(esc.values())
    reviewed = min(1.0, prm["capacity"] / e_total) if e_total > 0 else 1.0
    util = min(e_total, prm["capacity"]) / prm["capacity"]
    h = human_error(util, prm)
    cost = review = 0.0
    errors = {}
    by_seg = {}
    for s in seg:
        c, pm, pt = seg[s]["cost"], tab[s]["pmf"], tab[s]["p_true"]
        mask = CENTERS < thr[s]
        ns = n * seg[s]["share"]
        auto_err = ns * float((pm * (1 - pt))[~mask].sum())
        esc_model_err = ns * float((pm * (1 - pt))[mask].sum())  # if these went back to the agent
        human_cases = esc[s] * reviewed
        err = auto_err + human_cases * h + (1 - reviewed) * esc_model_err
        errors[s] = err
        by_seg[s] = err * c
        cost += err * c
        review += human_cases * prm["review_cost"]
    return {
        "cost": cost + review, "error_cost": cost, "review_cost": review, "escalated": e_total,
        "escalation_rate": e_total / n, "reviewed_share": reviewed, "utilisation": util, "human_err": h,
        "errors": errors, "error_cost_by_seg": by_seg, "esc_by_seg": esc,
        "error_rate": sum(errors.values()) / n,
    }


def chow_threshold(cost: float, prm: Dict[str, float], gamma: float = 1.0) -> float:
    """Escalate iff (1 - p) * cost > review_cost + human_err * cost, with p = c ** gamma.

    gamma = 1 trusts the stated confidence; the segment's true gamma uses the calibrated probability.
    Returns the stated-confidence cutoff (escalate below it)."""
    p_star = 1 - (prm["review_cost"] + prm["human_err"] * cost) / cost
    if p_star <= 0:
        return 0.0
    return float(p_star ** (1 / gamma))


def snap(t: float) -> float:
    """Thresholds act on bin centres; snap to the bin edge at or above t so policies are comparable."""
    return float(min(1.0, math.ceil(t * BINS - 1e-9) / BINS))


EDGES = [i / BINS for i in range(BINS + 1)]


def best_global(prm: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    prm = prm or BASE
    best = min(EDGES, key=lambda t: evaluate({s: t for s in SEGS}, prm)["cost"])
    return {s: best for s in SEGS}


def best_per_segment(prm: Optional[Dict[str, float]] = None,
                     segments: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, float]:
    """Joint grid search over the three thresholds (capacity couples them, so they cannot be optimised apart)."""
    prm = prm or BASE
    best, best_c = None, float("inf")
    for a in EDGES:
        for b in EDGES:
            for c in EDGES[: BINS // 2 + 1]:
                thr = {"refund": a, "account": b, "faq": c}
                v = evaluate(thr, prm, segments)["cost"]
                if v < best_c:
                    best, best_c = thr, v
    return best


def rate_target(rate: float, prm: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """One global cutoff chosen so the escalation rate is as close as possible to `rate` (a staffing target)."""
    prm = prm or BASE
    t = min(EDGES, key=lambda t: abs(evaluate({s: t for s in SEGS}, prm)["escalation_rate"] - rate))
    return {s: t for s in SEGS}


def policies(prm: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    prm = prm or BASE
    return {
        "never escalate": {s: 0.0 for s in SEGS},
        "accuracy rule: conf < 0.90": {s: 0.90 for s in SEGS},
        "staffing rule: fill capacity": rate_target(prm["capacity"] / prm["tickets"], prm),
        "best single threshold": best_global(prm),
        "priced, stated confidence": {s: snap(chow_threshold(SEGMENTS[s]["cost"], prm)) for s in SEGS},
        "priced, calibrated": {s: snap(chow_threshold(SEGMENTS[s]["cost"], prm, SEGMENTS[s]["gamma"])) for s in SEGS},
        "priced + capacity (joint)": best_per_segment(prm),
    }


def policy_table(prm: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    prm = prm or BASE
    rows = []
    for name, thr in policies(prm).items():
        r = evaluate(thr, prm)
        rows.append({"policy": name, "thr": thr, **{k: r[k] for k in (
            "cost", "error_cost", "review_cost", "escalated", "escalation_rate", "reviewed_share", "utilisation",
            "human_err", "error_rate")}, "error_cost_by_seg": r["error_cost_by_seg"], "esc_by_seg": r["esc_by_seg"]})
    return rows


def load_curve(prm: Optional[Dict[str, float]] = None, points: int = 13) -> List[Dict[str, float]]:
    """Scale the calibrated priced thresholds up and down: does escalating MORE always make the day safer?"""
    prm = prm or BASE
    base = {s: chow_threshold(SEGMENTS[s]["cost"], prm, SEGMENTS[s]["gamma"]) for s in SEGS}
    rows = []
    for i in range(points):
        k = 0.6 + 0.6 * i / (points - 1)  # 0.6x .. 1.2x of each priced cutoff
        thr = {s: min(1.0, base[s] * k) for s in SEGS}
        r = evaluate(thr, prm)
        rows.append({"scale": k, "escalated": r["escalated"], "utilisation_demand": r["escalated"] / prm["capacity"],
                     "cost": r["cost"], "errors": sum(r["errors"].values()), "human_err": r["human_err"]})
    return rows


INFLATE_V2 = 2.0  # the new model reports sqrt-shaped confidence: 0.81 skill now reads 0.90


def upgrade(prm: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """A model upgrade with identical skill that REPORTS higher confidence (every segment inflated).

    Same thresholds before and after (the policy did not change, only the model did)."""
    prm = prm or BASE
    v2 = {s: dict(v) for s, v in SEGMENTS.items()}
    for s in v2:
        v2[s]["inflate"] = INFLATE_V2
    thr = best_per_segment(prm)
    before = evaluate(thr, prm)
    after = evaluate(thr, prm, v2)
    thr2 = best_per_segment(prm, v2)
    refit = evaluate(thr2, prm, v2)
    keep = ("cost", "escalation_rate", "error_rate", "escalated")

    def row(r: Dict[str, object]) -> Dict[str, float]:
        out = {k: float(r[k]) for k in keep}
        out["refund_err_cost"] = float(r["error_cost_by_seg"]["refund"])
        return out

    return {"before": row(before), "after": row(after), "refit": row(refit), "thr": thr, "thr_refit": thr2}


def value_curve(prm: Optional[Dict[str, float]] = None) -> Dict[str, List[float]]:
    """Expected dollars one human review saves, per segment and reported-confidence bin, with a calm queue:
    (1 - p) * cost  -  (review_cost + human_err * cost). Positive means the review pays for itself."""
    prm = prm or BASE
    tab = segment_tables()
    out = {}
    for s in SEGS:
        c = SEGMENTS[s]["cost"]
        v = (1 - tab[s]["p_true"]) * c - (prm["review_cost"] + prm["human_err"] * c)
        out[s] = [float(x) if m > 0 else float("nan") for x, m in zip(v, tab[s]["pmf"])]
    return out


def simulate(thr: Dict[str, float], prm: Optional[Dict[str, float]] = None, days: int = MC_DAYS,
             seed: int = SEED) -> Dict[str, Dict[str, float]]:
    """Whole days, ticket by ticket. Overflow picks escalations to time out uniformly at random."""
    prm = prm or BASE
    rng = np.random.default_rng(seed)
    tab = segment_tables()
    n = int(prm["tickets"])
    tot = {s: [0, 0] for s in SEGS}  # errors, tickets
    for _ in range(days):
        seg_idx = rng.choice(len(SEGS), size=n, p=[SEGMENTS[s]["share"] for s in SEGS])
        bins = np.empty(n, dtype=int)
        for i, s in enumerate(SEGS):
            m = seg_idx == i
            bins[m] = rng.choice(BINS, size=int(m.sum()), p=tab[s]["pmf"])
        conf = CENTERS[bins]
        ptab = np.stack([tab[s]["p_true"] for s in SEGS])
        model_ok = rng.random(n) < ptab[seg_idx, bins]
        cut = np.array([thr[s] for s in SEGS])
        esc = conf < cut[seg_idx]
        e = int(esc.sum())
        reviewed = np.zeros(n, dtype=bool)
        if e:
            idx = np.flatnonzero(esc)
            k = min(e, int(prm["capacity"]))
            reviewed[rng.choice(idx, size=k, replace=False)] = True
        util = min(e, prm["capacity"]) / prm["capacity"]
        h = human_error(util, prm)
        human_ok = rng.random(n) >= h
        wrong = np.where(reviewed, ~human_ok, ~model_ok)
        for i, s in enumerate(SEGS):
            m = seg_idx == i
            tot[s][0] += int(wrong[m].sum())
            tot[s][1] += int(m.sum())
    return {s: {"errors": tot[s][0], "tickets": tot[s][1]} for s in SEGS}


def wilson(k: int, n: int, z: float = 2.5758) -> tuple:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def calibrate(prm: Optional[Dict[str, float]] = None) -> List[Dict[str, object]]:
    """Exact per-segment error rates vs the simulation, for every policy."""
    prm = prm or BASE
    rows = []
    for name, thr in policies(prm).items():
        ex = evaluate(thr, prm)
        mc = simulate(thr, prm)
        for s in SEGS:
            exact = ex["errors"][s] / (prm["tickets"] * SEGMENTS[s]["share"])
            k, n = mc[s]["errors"], mc[s]["tickets"]
            lo, hi = wilson(k, n)
            rows.append({"policy": name, "segment": s, "exact": exact, "mc": k / n, "inside": bool(lo <= exact <= hi)})
    return rows


def parse_params(text: str) -> Dict[str, float]:
    """`key=value, key=value` overrides of BASE. Unknown keys and bad numbers raise ValueError."""
    prm = dict(BASE)
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" not in part:
            raise ValueError(f"expected key=value, got {part!r}")
        k, v = [x.strip() for x in part.split("=", 1)]
        if k not in BASE:
            raise ValueError(f"unknown parameter {k!r}; known: {', '.join(BASE)}")
        try:
            prm[k] = float(v)
        except ValueError:
            raise ValueError(f"{k} needs a number, got {v!r}") from None
    if prm["capacity"] <= 0 or prm["tickets"] <= 0:
        raise ValueError("capacity and tickets must be positive")
    if not 0 <= prm["human_err"] <= prm["human_err_full"] <= 1:
        raise ValueError("need 0 <= human_err <= human_err_full <= 1")
    return prm
