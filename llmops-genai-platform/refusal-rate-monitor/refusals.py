"""A refusal rate is one number made of two errors, a traffic mix, and a detector - and three of the four move
without the model changing.

The dashboard reports REFUSAL RATE: the share of responses a refusal detector flags. Behind it sit four declared
objects. The traffic MIX (benign / benign-but-sensitive / harmful requests). The model's refusal probability per
class - which is really two error rates, OVER-refusal P(refuse | benign) and UNDER-refusal P(comply | harmful).
The model's refusal PHRASING, which changes with every version. And the DETECTOR, a keyword list written against
one version's phrasing, with a recall that is a property of the pair (detector, version), not of the detector.

    true_rate(v, mix)       = sum_c share_c * refuse_vc
    measured(v, mix, det)   = sum_c share_c * (refuse_vc * recall(det, v) + (1 - refuse_vc) * fpr(det))
    recall(det, v)          = sum_p phrasing_v[p] * [det matches p]

Everything is closed-form under the declared model; raw simulation only checks it. Binomial tails are exact
(log-space pmf), not normal approximations - the monitored quantities are rare events.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEED = 193
MC_REPS = 200_000

# ---------------------------------------------------------------------------------------------- traffic
# share of requests, and what the class is. `benign` and `sensitive` are both legitimate requests; the second is
# where refusals of legitimate requests concentrate (medical, legal, security questions).
CLASSES: List[Dict] = [
    {"name": "benign", "share": 0.80, "legit": True, "what": "plain requests"},
    {"name": "sensitive", "share": 0.15, "legit": True, "what": "legitimate but sensitive: medical, legal, security"},
    {"name": "harmful", "share": 0.05, "legit": False, "what": "requests the policy says to refuse"},
]
CAMPAIGN_MIX = {"benign": 0.74, "sensitive": 0.14, "harmful": 0.12}   # a jailbreak campaign week, model unchanged

# ---------------------------------------------------------------------------------------------- phrasing
PHRASES = ["I cannot", "I can't", "I'm not able to", "I won't be able to", "I'm unable to", "I must decline"]

# A version is a refusal probability per class and a distribution over refusal phrasings.
VERSIONS: Dict[str, Dict] = {
    "v1 baseline": {
        "refuse": {"benign": 0.006, "sensitive": 0.12, "harmful": 0.90},
        "phrasing": {"I cannot": 0.55, "I can't": 0.30, "I'm not able to": 0.10, "I won't be able to": 0.02,
                     "I'm unable to": 0.02, "I must decline": 0.01},
    },
    "v2 upgraded model": {
        "refuse": {"benign": 0.012, "sensitive": 0.24, "harmful": 0.97},
        "phrasing": {"I cannot": 0.12, "I can't": 0.08, "I'm not able to": 0.45, "I won't be able to": 0.20,
                     "I'm unable to": 0.10, "I must decline": 0.05},
    },
    "v3 prompt tuned down": {
        "refuse": {"benign": 0.003, "sensitive": 0.06, "harmful": 0.70},
        "phrasing": {"I cannot": 0.55, "I can't": 0.30, "I'm not able to": 0.10, "I won't be able to": 0.02,
                     "I'm unable to": 0.02, "I must decline": 0.01},
    },
}

# A detector is the set of phrasings it matches, plus a false-positive rate on compliant responses (which say
# "I cannot stress enough how..." and "I can't wait to show you..." more often than you would think).
DETECTORS: Dict[str, Dict] = {
    "keyword list (written on v1)": {"matches": {"I cannot", "I can't"}, "fpr": 0.004},
    "keyword list, extended": {"matches": set(PHRASES), "fpr": 0.009},
    "LLM judge": {"matches": None, "recall": 0.93, "fpr": 0.010},
}


# ---------------------------------------------------------------------------------------------- closed form
def mix_of(classes: Sequence[Dict] = CLASSES, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    m = {c["name"]: c["share"] for c in classes}
    if override:
        m.update(override)
    tot = sum(m.values())
    return {k: v / tot for k, v in m.items()}


def true_rates(version: Dict, mix: Dict[str, float], classes: Sequence[Dict] = CLASSES) -> Dict:
    """The aggregate and the two errors it is made of."""
    legit = {c["name"]: c["legit"] for c in classes}
    agg = sum(mix[c] * version["refuse"][c] for c in mix)
    legit_share = sum(mix[c] for c in mix if legit[c])
    over = sum(mix[c] * version["refuse"][c] for c in mix if legit[c]) / legit_share
    harm_share = sum(mix[c] for c in mix if not legit[c])
    under = sum(mix[c] * (1 - version["refuse"][c]) for c in mix if not legit[c]) / harm_share if harm_share else 0.0
    benign_refusals = sum(mix[c] * version["refuse"][c] for c in mix if legit[c])
    return {"refusal_rate": agg, "over_refusal": over, "under_refusal": under,
            "legit_share_of_refusals": benign_refusals / agg if agg else 0.0,
            "per_class": {c: version["refuse"][c] for c in mix}}


def recall(detector: Dict, version: Dict) -> float:
    if detector["matches"] is None:
        return detector["recall"]
    return sum(p for ph, p in version["phrasing"].items() if ph in detector["matches"])


def measured_rate(version: Dict, mix: Dict[str, float], detector: Dict) -> float:
    r = recall(detector, version)
    return sum(mix[c] * (version["refuse"][c] * r + (1 - version["refuse"][c]) * detector["fpr"]) for c in mix)


def flat_aggregate_twin(base: Dict, mix: Dict[str, float], over_multiplier: float) -> Dict:
    """The version that multiplies every legitimate class's refusal rate by `over_multiplier` and keeps the
    AGGREGATE exactly flat. At a fixed mix the harmful refusal rate is forced - and it must move the other way."""
    agg = true_rates(base, mix)["refusal_rate"]
    refuse = {c: base["refuse"][c] * (over_multiplier if c != "harmful" else 1.0) for c in mix}
    legit_part = sum(mix[c] * refuse[c] for c in mix if c != "harmful")
    refuse["harmful"] = (agg - legit_part) / mix["harmful"]
    return {"refuse": refuse, "phrasing": base["phrasing"]}


# ---------------------------------------------------------------------------------------------- exact binomials
def log_pmf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    if p <= 0:
        return np.where(k == 0, 0.0, -np.inf)
    if p >= 1:
        return np.where(k == n, 0.0, -np.inf)
    lc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    return lc + k * math.log(p) + (n - k) * math.log1p(-p)


def upper_tail(k_from: int, n: int, p: float) -> float:
    """P(X >= k_from), X ~ Binomial(n, p), exact."""
    if k_from <= 0:
        return 1.0
    if k_from > n:
        return 0.0
    ks = np.arange(k_from, n + 1)
    return float(np.exp(log_pmf(ks, n, p)).sum())


def critical_k(n: int, p0: float, alpha: float) -> int:
    """Smallest k with P(X >= k | p0) <= alpha."""
    ks = np.arange(0, n + 1)
    tail = np.cumsum(np.exp(log_pmf(ks, n, p0))[::-1])[::-1]   # tail[k] = P(X >= k)
    idx = np.nonzero(tail <= alpha)[0]
    return int(idx[0]) if len(idx) else n + 1


def audit_power(p0: float, p1: float, n: int, alpha: float = 0.05) -> Dict:
    k = critical_k(n, p0, alpha)
    return {"n": n, "p0": p0, "p1": p1, "critical_k": k, "size": upper_tail(k, n, p0), "power": upper_tail(k, n, p1)}


def audit_designs(base: Dict, new: Dict, mix: Dict[str, float], ns: Sequence[int], detector: Dict,
                  classes: Sequence[Dict] = CLASSES) -> Dict[str, List[Dict]]:
    """Two ways to spend n labels a week. TRAFFIC: label n random responses, estimate over-refusal from the
    legitimate ones (so ~95% of n is usable and the event is rare). REFUSALS: label n detected refusals and
    estimate the share that were legitimate requests (every label is on-target, the event is common)."""
    legit = {c["name"]: c["legit"] for c in classes}
    legit_share = sum(mix[c] for c in mix if legit[c])
    r0, r1 = true_rates(base, mix), true_rates(new, mix)
    traffic = [audit_power(r0["over_refusal"], r1["over_refusal"], int(round(n * legit_share))) | {"n": n} for n in ns]
    # share of DETECTED refusals that came from legitimate requests: detection is phrasing-based, phrasing is
    # independent of class, so recall cancels - but the detector's false positives do not.
    def legit_share_detected(v: Dict) -> float:
        r = recall(detector, v)
        num = sum(mix[c] * (v["refuse"][c] * r + (1 - v["refuse"][c]) * detector["fpr"]) for c in mix if legit[c])
        return num / measured_rate(v, mix, detector)
    refusals = [audit_power(legit_share_detected(base), legit_share_detected(new), n) for n in ns]
    return {"traffic": traffic, "refusals": refusals}


# ---------------------------------------------------------------------------------------------- the monitor
WEEKS = 26
WEEKLY_N = 20_000
AUDIT_N = 300
CAMPAIGN_WEEKS = (9, 10, 11)
UPGRADE_WEEK = 19


def week_state(week: int) -> Tuple[Dict, Dict[str, float]]:
    version = VERSIONS["v2 upgraded model"] if week >= UPGRADE_WEEK else VERSIONS["v1 baseline"]
    mix = mix_of(override=CAMPAIGN_MIX) if week in CAMPAIGN_WEEKS else mix_of()
    return version, mix


def monitor(detector: Dict, weeks: int = WEEKS, n: int = WEEKLY_N, audit_n: int = AUDIT_N, z: float = 3.0) -> List[Dict]:
    """Two charts on the same 26 weeks. AGGREGATE: the measured refusal count out of n, alarm above the v1
    baseline mean + z sigma (and flagged below mean - z sigma as a 'drop'). AUDIT: of audit_n detected refusals,
    the number from legitimate requests, alarm above its baseline + z sigma. P(alarm) is an exact binomial tail
    at each week's true state."""
    base_v, base_mix = VERSIONS["v1 baseline"], mix_of()
    legit = {c["name"]: c["legit"] for c in CLASSES}
    p_base = measured_rate(base_v, base_mix, detector)
    hi_k = math.ceil(n * p_base + z * math.sqrt(n * p_base * (1 - p_base)))
    lo_k = math.floor(n * p_base - z * math.sqrt(n * p_base * (1 - p_base)))

    def legit_share_detected(v: Dict, mix: Dict[str, float]) -> float:
        r = recall(detector, v)
        num = sum(mix[c] * (v["refuse"][c] * r + (1 - v["refuse"][c]) * detector["fpr"]) for c in mix if legit[c])
        return num / measured_rate(v, mix, detector)

    q_base = legit_share_detected(base_v, base_mix)
    audit_k = math.ceil(audit_n * q_base + z * math.sqrt(audit_n * q_base * (1 - q_base)))
    rows = []
    for w in range(1, weeks + 1):
        v, mix = week_state(w)
        p = measured_rate(v, mix, detector)
        tr = true_rates(v, mix)
        rows.append({
            "week": w, "campaign": w in CAMPAIGN_WEEKS, "upgraded": w >= UPGRADE_WEEK,
            "true_rate": tr["refusal_rate"], "measured": p, "over_refusal": tr["over_refusal"],
            "under_refusal": tr["under_refusal"], "recall": recall(detector, v),
            "p_alarm_high": upper_tail(hi_k, n, p), "p_alarm_low": max(0.0, 1 - upper_tail(lo_k + 1, n, p)),
            "legit_share_detected": legit_share_detected(v, mix),
            "p_audit_alarm": upper_tail(audit_k, audit_n, legit_share_detected(v, mix)),
        })
    return rows


# ---------------------------------------------------------------------------------------------- calibration
def wilson(k: int, n: int, z: float = 2.5758) -> Tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def simulate(version: Dict, mix: Dict[str, float], detector: Dict, reps: int = MC_REPS, seed: int = SEED) -> Dict:
    """Raw simulation of one response at a time: draw the class, whether the model refuses, which phrasing it
    uses, whether the detector fires. Returns (successes, trials) for a Wilson check."""
    rng = np.random.default_rng(seed)
    names = list(mix)
    cls = rng.choice(len(names), size=reps, p=[mix[c] for c in names])
    p_ref = np.array([version["refuse"][c] for c in names])[cls]
    refused = rng.random(reps) < p_ref
    if detector["matches"] is None:
        fires = np.where(refused, rng.random(reps) < detector["recall"], rng.random(reps) < detector["fpr"])
    else:
        ph = rng.choice(len(PHRASES), size=reps, p=[version["phrasing"].get(p, 0.0) for p in PHRASES])
        hit = np.array([PHRASES[i] in detector["matches"] for i in range(len(PHRASES))])[ph]
        fires = np.where(refused, hit, rng.random(reps) < detector["fpr"])
    legit = np.array([c != "harmful" for c in names])[cls]
    out = {"refusal_rate": (int(refused.sum()), reps), "measured": (int(fires.sum()), reps),
           "over_refusal": (int((refused & legit).sum()), int(legit.sum())),
           "under_refusal": (int((~refused & ~legit).sum()), int((~legit).sum())),
           "legit_share_detected": (int((fires & legit).sum()), int(fires.sum()))}
    return out


def calibrate(reps: int = MC_REPS, seed: int = SEED) -> List[Dict]:
    rows = []
    det = DETECTORS["keyword list (written on v1)"]
    cases = [("v1 baseline", mix_of()), ("v2 upgraded model", mix_of()), ("v1 baseline", mix_of(override=CAMPAIGN_MIX))]
    for s, (vname, mix) in enumerate(cases):
        v = VERSIONS[vname]
        tr = true_rates(v, mix)
        exact = {**{k: tr[k] for k in ("refusal_rate", "over_refusal", "under_refusal")},
                 "measured": measured_rate(v, mix, det)}
        legit = [c for c in mix if c != "harmful"]
        r = recall(det, v)
        exact["legit_share_detected"] = sum(mix[c] * (v["refuse"][c] * r + (1 - v["refuse"][c]) * det["fpr"])
                                            for c in legit) / exact["measured"]
        sim = simulate(v, mix, det, reps, seed + s)
        label = vname + (" (campaign mix)" if mix["harmful"] > 0.1 else "")
        for key, (k, n) in sim.items():
            lo, hi = wilson(k, n)
            rows.append({"case": label, "quantity": key, "exact": exact[key], "mc": k / n, "inside_99": bool(lo <= exact[key] <= hi)})
    # the exact binomial tail against a raw draw
    rng = np.random.default_rng(seed + 10)
    n, p0, p1 = 300, 0.33, 0.49
    k = critical_k(n, p0, 0.05)
    draws = rng.binomial(n, p1, reps)
    kk = int((draws >= k).sum())
    lo, hi = wilson(kk, reps)
    ex = upper_tail(k, n, p1)
    rows.append({"case": "binomial tail", "quantity": f"P(X>={k} | n={n}, p={p1})", "exact": ex, "mc": kk / reps,
                 "inside_99": bool(lo <= ex <= hi)})
    return rows


# ---------------------------------------------------------------------------------------------- app parsing
def parse_version(text: str) -> Dict:
    """`benign, sensitive, harmful` refusal rates, e.g. `0.006, 0.12, 0.90`; phrasing defaults to v1's."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"need 3 comma-separated refusal rates (benign, sensitive, harmful), got {len(parts)}: {text!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"rates did not parse: {text!r}")
    if not all(0 <= v <= 1 for v in vals):
        raise ValueError(f"rates must be between 0 and 1: {text!r}")
    return {"refuse": dict(zip(("benign", "sensitive", "harmful"), vals)), "phrasing": VERSIONS["v1 baseline"]["phrasing"]}


def parse_mix(text: str) -> Dict[str, float]:
    """`benign, sensitive, harmful` traffic shares; normalised."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"need 3 comma-separated traffic shares (benign, sensitive, harmful), got {len(parts)}: {text!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"shares did not parse: {text!r}")
    if any(v < 0 for v in vals) or sum(vals) <= 0 or vals[2] <= 0:
        raise ValueError(f"shares must be >= 0, sum > 0, harmful > 0: {text!r}")
    tot = sum(vals)
    return dict(zip(("benign", "sensitive", "harmful"), [v / tot for v in vals]))
