"""A golden set is a sample of production traffic. Which sample, and how it goes stale.

Production traffic falls into K intents with a mix w_k(t) that drifts month by month, and the model passes
intent k with probability q_k. The number the dashboard shows is a pass rate on a labelled golden set of m
cases, built ONCE at month 0. This module measures:

  * how far each allocation design's headline sits from the true production pass rate sum_k w_k(t) q_k,
    at month 0 and after 12 months of drift - exactly, from binomial variances, where the design fixes
    the per-intent counts (Monte Carlo only for simple random sampling, where the counts are random);
  * how much of the staleness a free fix removes: reweighting per-intent pass rates by TODAY's traffic mix
    (unlabelled counts) - and the part it cannot remove, intents that did not exist when the set was built;
  * whether a set can catch a regression confined to one small intent, given that an LLM's pass/fail on a
    case also changes between runs (paired exact McNemar, aggregate vs per-intent with Bonferroni). Run-to-run
    noise is modelled per CASE: a flaky case passes each run with probability 1/2, a stable one always does
    the same thing, so a re-run of the same model has the same pass rate. The first version flipped every
    outcome with probability 0.03 instead; that silently lowers the re-run's pass rate (at q = 0.9 there are
    nine passes to flip for every fail) and the aggregate test then flagged the SAME model 58% of the time.

The traffic is synthetic and its structure is declared here, not discovered.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

ALPHA = 0.05
M = 240  # labelling budget: cases in the golden set
MONTHS = 12
INTENTS = ["billing", "password reset", "order status", "refund", "shipping", "account update",
           "product question", "cancel", "bulk export", "API errors", "voice agent", "invoice split"]
EXISTING = 10  # the last two intents launch after month 0: zero traffic when the set is built
_Z = 1 / np.arange(1, EXISTING + 1) ** 1.1
W0 = np.r_[_Z / _Z.sum(), 0.0, 0.0]
NEW_SHARE = np.array([0.08, 0.05])  # traffic share of the two new intents at month 12
W12 = np.r_[W0[:EXISTING] * (1 - NEW_SHARE.sum()) * np.linspace(0.8, 1.6, EXISTING)
            / np.sum(W0[:EXISTING] * np.linspace(0.8, 1.6, EXISTING)), NEW_SHARE]
Q = np.array([0.94, 0.92, 0.95, 0.83, 0.90, 0.88, 0.78, 0.86, 0.72, 0.70, 0.55, 0.60])
DESIGNS = ("random", "proportional", "sqrt", "equal")
FLAKY = 0.08  # share of cases whose outcome is a coin flip on every run (the rest are deterministic)
REG_INTENT = 8  # "bulk export": 2.9% of month-0 traffic
REG_FLIP = 0.5  # v2 breaks half the cases v1 passed there: 0.72 -> 0.36
MC_REPS = 20000


def mix(t: float) -> np.ndarray:
    """Traffic mix at month t: linear from W0 to W12."""
    f = min(max(t / MONTHS, 0.0), 1.0)
    return (1 - f) * W0 + f * W12


def truth(t: float, q: np.ndarray = Q) -> float:
    return float(mix(t) @ q)


def allocate(design: str, m: int = M, w: np.ndarray = W0) -> np.ndarray:
    """Per-intent case counts for a fixed-allocation design (largest remainder, every known intent >= 1)."""
    known = w > 0
    if design == "proportional":
        target = w
    elif design == "sqrt":
        target = np.sqrt(w)
    elif design == "equal":
        target = known.astype(float)
    else:
        raise ValueError(f"{design!r} has no fixed allocation")
    target = target / target.sum() * m
    n = np.floor(target).astype(int)
    n[known & (n == 0)] = 1
    rem = m - n.sum()
    order = np.argsort(-(target - np.floor(target)))
    for i in order:
        if rem == 0:
            break
        if known[i]:
            n[i] += int(np.sign(rem))
            rem -= int(np.sign(rem))
    assert n.sum() == m and np.all(n[~known] == 0)
    return n


def moments(n: np.ndarray, t: float, reweight: bool, q: np.ndarray = Q) -> Dict[str, float]:
    """Exact bias, SD and RMSE of the headline for fixed counts n, against the truth at month t.

    raw:       sum_k (n_k / m) phat_k            (what a dashboard over the golden set shows)
    reweight:  sum_{k covered} v_k phat_k,  v = today's mix renormalised over the intents the set covers
    """
    m = n.sum()
    cov = n > 0
    wt = n / m if not reweight else np.where(cov, mix(t), 0) / np.where(cov, mix(t), 0).sum()
    mean = float(wt @ q)
    var = float(np.sum(np.where(cov, wt ** 2 * q * (1 - q) / np.maximum(n, 1), 0)))
    bias = mean - truth(t, q)
    return {"bias": bias, "sd": float(np.sqrt(var)), "rmse": float(np.sqrt(bias ** 2 + var))}


def moments_random(t: float, reweight: bool, m: int = M, reps: int = MC_REPS, seed: int = 5,
                   q: np.ndarray = Q) -> Dict[str, float]:
    """Simple random sampling from month-0 traffic. Raw is exact (a binomial); reweighted is Monte Carlo."""
    if not reweight:
        p0 = float(W0 @ q)
        bias = p0 - truth(t, q)
        var = p0 * (1 - p0) / m
        return {"bias": bias, "sd": float(np.sqrt(var)), "rmse": float(np.sqrt(bias ** 2 + var)), "exact": True}
    rng = np.random.default_rng(seed)
    n = rng.multinomial(m, W0, size=reps)
    k = rng.binomial(n, q)
    cov = n > 0
    v = np.where(cov, mix(t), 0)
    v = v / v.sum(axis=1, keepdims=True)
    est = np.sum(v * np.where(cov, k / np.maximum(n, 1), 0), axis=1)
    err = est - truth(t, q)
    return {"bias": float(err.mean()), "sd": float(est.std()), "rmse": float(np.sqrt(np.mean(err ** 2))), "exact": False}


def headline(design: str, t: float, reweight: bool, m: int = M) -> Dict[str, float]:
    if design == "random":
        return moments_random(t, reweight, m)
    return {**moments(allocate(design, m), t, reweight), "exact": True}


def coverage_gap(t: float) -> Dict[str, float]:
    """What reweighting cannot fix: the traffic share with no cases, and the bias it leaves."""
    cov = W0 > 0
    w = mix(t)
    v = np.where(cov, w, 0) / w[cov].sum()
    return {"uncovered_share": float(w[~cov].sum()), "bias_left": float(v @ Q - truth(t))}


def refresh(n: np.ndarray, add_per_new: int) -> np.ndarray:
    out = n.copy()
    out[W0 == 0] += add_per_new
    return out


def mcnemar_p(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """One-sided exact McNemar: P(Binomial(b + c, 1/2) >= b). b = pass->fail, c = fail->pass."""
    return stats.binom.sf(np.asarray(b) - 1, np.asarray(b) + np.asarray(c), 0.5)


def regression_power(n: np.ndarray, flip: float = REG_FLIP, flaky: float = FLAKY, reps: int = MC_REPS,
                     seed: int = 9) -> Dict[str, float]:
    """P(flag) for v1 vs v2 on the same golden cases. With flip = 0 this is the false-alarm rate.

    Each case is flaky with probability `flaky` (passes each run w.p. 1/2) or stable (passes every run w.p.
    s_k, chosen so the marginal pass rate is exactly q_k). In the regressed intent v2 additionally breaks
    a case v1 passed with probability `flip`. Aggregate: one McNemar on all m cases at ALPHA. Per-intent:
    McNemar in each covered intent at ALPHA / (number covered), flag if any fires.
    """
    rng = np.random.default_rng(seed)
    cov = np.flatnonzero(n > 0)
    B = np.zeros((reps, len(Q)), dtype=int)
    C = np.zeros_like(B)
    for k in cov:
        shape = (reps, n[k])
        is_flaky = rng.random(shape) < flaky
        stable_pass = rng.random(shape) < (Q[k] - flaky / 2) / (1 - flaky)
        v1 = np.where(is_flaky, rng.random(shape) < 0.5, stable_pass)
        v2 = np.where(is_flaky, rng.random(shape) < 0.5, stable_pass)
        if k == REG_INTENT and flip > 0:
            v2 &= ~(v1 & (rng.random(shape) < flip))
        B[:, k] = np.sum(v1 & ~v2, axis=1)
        C[:, k] = np.sum(~v1 & v2, axis=1)
    agg = mcnemar_p(B.sum(axis=1), C.sum(axis=1)) < ALPHA
    per = (mcnemar_p(B[:, cov], C[:, cov]) < ALPHA / len(cov)).any(axis=1)
    return {"aggregate": float(agg.mean()), "per_intent": float(per.mean()), "cases_in_regressed": int(n[REG_INTENT])}


def audit(rows: Sequence[Dict], min_share: float = 0.02) -> Dict:
    """The app's core: a golden set's per-intent (cases, passes) against today's traffic counts."""
    names = [str(r["intent"]) for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("each intent must appear once")
    n = np.array([int(r["cases"]) for r in rows])
    k = np.array([int(r["passes"]) for r in rows])
    traffic = np.array([float(r["traffic"]) for r in rows])
    if np.any(n < 0) or np.any(k < 0) or np.any(k > n) or np.any(traffic < 0):
        raise ValueError("need 0 <= passes <= cases and traffic >= 0")
    if n.sum() == 0 or traffic.sum() == 0:
        raise ValueError("need at least one labelled case and some traffic")
    w = traffic / traffic.sum()
    cov = n > 0
    raw = float(k.sum() / n.sum())
    v = np.where(cov, w, 0)
    rew = float(np.sum(v / v.sum() * np.where(cov, k / np.maximum(n, 1), 0)))
    rows_out = []
    for i, name in enumerate(names):
        lo, hi = wilson(int(k[i]), int(n[i])) if n[i] else (float("nan"), float("nan"))
        rows_out.append({"intent": name, "traffic share": float(w[i]), "set share": float(n[i] / n.sum()),
                         "cases": int(n[i]), "pass rate": float(k[i] / n[i]) if n[i] else float("nan"),
                         "99% low": lo, "99% high": hi})
    gaps = [names[i] for i in range(len(names)) if not cov[i] and w[i] >= min_share]
    return {"raw": raw, "reweighted": rew, "uncovered_share": float(w[~cov].sum()), "gaps": gaps,
            "thin": [names[i] for i in range(len(names)) if cov[i] and n[i] < 10 and w[i] >= min_share],
            "rows": rows_out}


def wilson(k: int, n: int, z: float = float(stats.norm.isf(0.005))) -> List[float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]


def calibrate(reps: int = MC_REPS, seed: int = 21) -> Dict:
    """The exact moments against a raw simulation of drawing and labelling the set."""
    rng = np.random.default_rng(seed)
    out = []
    for design in ("proportional", "equal"):
        n = allocate(design)
        for t, rw in ((0, False), (12, False), (12, True)):
            k = rng.binomial(n, Q, size=(reps, len(Q)))
            cov = n > 0
            wt = n / n.sum() if not rw else np.where(cov, mix(t), 0) / np.where(cov, mix(t), 0).sum()
            est = np.sum(wt * np.where(cov, k / np.maximum(n, 1), 0), axis=1)
            ex = moments(n, t, rw)
            se_mean = est.std() / np.sqrt(reps)
            out.append({"design": design, "month": t, "reweight": rw, "exact_bias": ex["bias"],
                        "mc_bias": float(est.mean() - truth(t)), "exact_sd": ex["sd"], "mc_sd": float(est.std()),
                        "bias_z": float((est.mean() - truth(t) - ex["bias"]) / se_mean),
                        "sd_ratio": float(est.std() / ex["sd"])})
    # the McNemar p-value against scipy's exact binomial test
    mism = sum(abs(float(mcnemar_p(b, c)) - stats.binomtest(b, b + c, 0.5, alternative="greater").pvalue) > 1e-12
               for b in range(0, 15) for c in range(0, 15) if b + c > 0)
    return {"moments": out, "mcnemar_vs_binomtest_mismatches": int(mism), "mcnemar_cases": 224}
