"""No significant difference is not evidence of no difference. TOST, measured exactly.

The usual report: "the new pricing engine made no significant difference to basket value (p = 0.31)".
That sentence is read as "the two are the same". It is not what the test said. The test was built to
detect a difference and it failed to; how often it fails on a difference that is REAL depends on n.

The procedure that answers "are these the same, within a margin that matters?" is the two one-sided
tests (TOST): pick a margin D, reject both "delta <= -D" and "delta >= +D" at alpha, and only then call
the two equivalent. That is identical to the 90% confidence interval sitting wholly inside (-D, +D).

Everything here is EXACT, not simulated. For a two-sample Student t with equal n and normal data, the
difference of means and the pooled SD are independent, so conditioning on the SD makes every decision
region an interval in the mean difference. One numerical integral over the chi-square distribution of
the variance gives the probability of each of the four outcomes

    equivalent & not different | equivalent & different | inconclusive | different & not equivalent

at any true delta and n. Monte Carlo on raw data appears only as the cross-check (section 1).

What separates this from the built `sample-size-calc` (power to DETECT a difference) and
`t-test-variants` (which t-test): nothing in the catalog measures the claim of NO difference.

Note on duplication: the Wilson interval also appears in sibling builds. Deliberate - every build must
run standalone from a bare Colab link. One copy WITHIN this build.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import integrate, stats

ALPHA = 0.05
MARGIN = 0.5  # equivalence margin in SD units: the smallest difference the business says matters
NS = (10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500)
DELTAS = (0.0, 0.2, 0.5, 1.0, 1.5)  # true difference as a MULTIPLE of the margin
OUTCOMES = ("equiv_only", "both", "inconclusive", "diff_only")
LABEL = {"equiv_only": "equivalent, not different", "both": "equivalent AND different",
         "inconclusive": "inconclusive (neither)", "diff_only": "different, not equivalent"}
Z99 = 2.5758293035489
MC_REPS = 20_000
# Seeds are INDICES into this list, never hash() of a design (Day 178: string hashing is salted).
MC_DESIGNS: List[Tuple[int, float]] = [(10, 0.0), (20, 1.0), (50, 0.0), (50, 1.0), (100, 0.5),
                                       (200, 0.0), (500, 0.2)]


def wilson(hits: int, n: int, z: float = Z99) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # At 0 (or n) hits the formula's bound is 0 (or 1) only up to float noise: 2.7e-20, which
    # failed an exact 0.0 outcome on the first run. Clamp to the true bound.
    lo = 0.0 if hits == 0 else float(mid - half)
    hi = 1.0 if hits == n else float(mid + half)
    return (lo, hi)


# --------------------------------------------------------------------------- 1. the tests on data

def analyse(x: Sequence[float], y: Sequence[float], margin: float, alpha: float = ALPHA) -> Dict:
    """Both analyses on one dataset. delta = mean(y) - mean(x); Student pooled SD, as in the study."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("each group needs at least 2 values")
    if margin <= 0:
        raise ValueError("the equivalence margin must be positive")
    n1, n2 = len(x), len(y)
    df = n1 + n2 - 2
    sp2 = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / df
    if sp2 == 0:
        raise ValueError("both groups have zero spread")
    se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    d = float(y.mean() - x.mean())
    p_diff = float(2 * stats.t.sf(abs(d) / se, df))
    p_lower = float(stats.t.sf((d + margin) / se, df))  # H0: delta <= -margin
    p_upper = float(stats.t.cdf((d - margin) / se, df))  # H0: delta >= +margin
    t90 = stats.t.ppf(1 - alpha, df)
    ci90 = (d - t90 * se, d + t90 * se)
    p_tost = max(p_lower, p_upper)
    different, equivalent = p_diff < alpha, p_tost < alpha
    outcome = ("both" if different and equivalent else "equiv_only" if equivalent
               else "diff_only" if different else "inconclusive")
    return {"delta": d, "se": float(se), "df": df, "p_diff": p_diff, "p_tost": p_tost,
            "p_lower": p_lower, "p_upper": p_upper, "ci90": (float(ci90[0]), float(ci90[1])),
            "outcome": outcome}


def tost_batch(x: np.ndarray, y: np.ndarray, margin: float, alpha: float = ALPHA) -> Dict:
    """Vectorised decisions for (reps, n) arrays, same formulas as analyse()."""
    n = x.shape[1]
    df = 2 * n - 2
    sp2 = (x.var(axis=1, ddof=1) + y.var(axis=1, ddof=1)) / 2
    se = np.sqrt(sp2 * 2 / n)
    d = y.mean(axis=1) - x.mean(axis=1)
    diff = np.abs(d) / se > stats.t.ppf(1 - alpha / 2, df)
    equiv = (d - stats.t.ppf(1 - alpha, df) * se > -margin) & (d + stats.t.ppf(1 - alpha, df) * se < margin)
    return {"different": diff, "equivalent": equiv}


# --------------------------------------------------------------------------- 2. the exact engine

def outcome_probs(n: int, delta: float, margin: float = MARGIN, alpha: float = ALPHA) -> Dict[str, float]:
    """P(each of the four outcomes) at n per group, true difference delta, sigma = 1. EXACT.

    Given the pooled SD s the mean difference D ~ N(delta, 2/n) is independent of s, and with
    se = s*sqrt(2/n):  equivalent  <=>  |D| < margin - t90*se ;  different  <=>  |D| > t95*se.
    Integrate the conditional normal probabilities over V = df*s^2 ~ chi2(df).
    """
    df = 2 * n - 2
    sd = np.sqrt(2.0 / n)
    t_eq, t_df = stats.t.ppf(1 - alpha, df), stats.t.ppf(1 - alpha / 2, df)
    Phi = stats.norm.cdf

    def band(lo: float, hi: float) -> float:  # P(lo < |D| < hi), hi may be inf
        if hi <= lo:
            return 0.0
        return float(Phi((hi - delta) / sd) - Phi((lo - delta) / sd)
                     + Phi((-lo - delta) / sd) - Phi((-hi - delta) / sd))

    def cond(v: float) -> np.ndarray:
        se = np.sqrt(v / df) * sd
        e, c = margin - t_eq * se, t_df * se  # equivalence radius, difference cutoff
        if e <= 0:
            return np.array([0.0, 0.0, band(0, c), band(c, np.inf)])
        return np.array([band(0, min(e, c)), band(c, e), band(e, c), band(max(e, c), np.inf)])

    pdf = stats.chi2(df).pdf
    hi = stats.chi2.ppf(1 - 1e-13, df)
    kink = df * (margin / ((t_eq + t_df) * sd)) ** 2  # where e = c: the integrand changes form
    v_eq = df * (margin / (t_eq * sd)) ** 2  # beyond this, equivalence is impossible
    pts = sorted(p for p in (kink, v_eq) if 0 < p < hi)
    out = np.zeros(4)
    for i in range(4):
        out[i] = integrate.quad(lambda v: cond(v)[i] * pdf(v), 0, hi, points=pts or None,
                                limit=200, epsabs=1e-12, epsrel=1e-10)[0]
    return dict(zip(OUTCOMES, out.tolist()))


def p_equivalent(n: int, delta: float, margin: float = MARGIN, alpha: float = ALPHA) -> float:
    p = outcome_probs(n, delta, margin, alpha)
    return p["equiv_only"] + p["both"]


def p_not_significant(n: int, delta: float, alpha: float = ALPHA) -> float:
    """Closed form from the noncentral t - used to CHECK the integral, not by the study."""
    df = 2 * n - 2
    tc, nc = stats.t.ppf(1 - alpha / 2, df), delta / np.sqrt(2.0 / n)
    return float(stats.nct.cdf(tc, df, nc) - stats.nct.cdf(-tc, df, nc))


def n_for_power(target: float, delta: float, margin: float = MARGIN, alpha: float = ALPHA) -> int:
    """Smallest n per group where P(equivalent) >= target at true delta. Power rises in n (checked)."""
    lo, hi = 2, 4
    while p_equivalent(hi, delta, margin, alpha) < target:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if p_equivalent(mid, delta, margin, alpha) >= target else (mid, hi)
    return hi


def n_normal_approx(target: float, delta_frac: float, margin: float = MARGIN, alpha: float = ALPHA) -> float:
    """The textbook shortcut. At delta = 0 the two tails share beta: z_(1-beta/2); else z_(1-beta)."""
    zb = stats.norm.ppf(1 - (1 - target) / 2) if delta_frac == 0 else stats.norm.ppf(target)
    return 2 * (stats.norm.ppf(1 - alpha) + zb) ** 2 / (margin * (1 - delta_frac)) ** 2


# --------------------------------------------------------------------------- 3. the study

def grid() -> List[Dict]:
    rows = []
    for n in NS:
        for f in DELTAS:
            p = outcome_probs(n, f * MARGIN)
            rows.append({"n": n, "delta_frac": f, **p,
                         "equivalent": p["equiv_only"] + p["both"],
                         "not_significant": p["equiv_only"] + p["inconclusive"]})
    return rows


def calibrate(seed: int = 0) -> Dict:
    """Four checks, each against something independent of the integral."""
    rng = np.random.default_rng(seed)
    # (a) vectorised decisions == scipy's own one-sided and two-sided t-tests on the same raw data
    mism, total = 0, 0
    for n in (8, 25, 60):
        x = rng.normal(0, 1, (400, n))
        y = rng.normal(0.3, 1, (400, n))
        dec = tost_batch(x, y, MARGIN)
        lo = stats.ttest_ind(y + MARGIN, x, axis=1, alternative="greater").pvalue
        up = stats.ttest_ind(y - MARGIN, x, axis=1, alternative="less").pvalue
        two = stats.ttest_ind(y, x, axis=1).pvalue
        mism += int(np.sum(dec["equivalent"] != (np.maximum(lo, up) < ALPHA)))
        mism += int(np.sum(dec["different"] != (two < ALPHA)))
        total += 2 * 400
    # (b) the integral's P(not significant) vs the noncentral t closed form
    gap = sum_gap = 0.0
    for n in (10, 30, 100, 500):
        for f in DELTAS:
            p = outcome_probs(n, f * MARGIN)
            gap = max(gap, abs(p["equiv_only"] + p["inconclusive"] - p_not_significant(n, f * MARGIN)))
            # (c) the four outcomes sum to one
            sum_gap = max(sum_gap, abs(sum(p.values()) - 1))
    # (d) TOST == 90% CI inside the margin, on single datasets through analyse()
    ci_mism = 0
    for i in range(300):
        x, y = rng.normal(0, 1, 30), rng.normal(0.2, 1, 30)
        r = analyse(x, y, MARGIN)
        ci_mism += int((r["p_tost"] < ALPHA) != (r["ci90"][0] > -MARGIN and r["ci90"][1] < MARGIN))
    return {"scipy_decision_mismatches": mism, "decisions_compared": total,
            "max_gap_vs_noncentral_t": float(gap), "max_sum_gap": float(sum_gap),
            "tost_vs_ci90_mismatches": ci_mism}


def monte_carlo(reps: int = MC_REPS) -> List[Dict]:
    """Raw-data replicates vs the exact numbers. Each outcome must sit inside its 99% Wilson interval."""
    rows = []
    for i, (n, f) in enumerate(MC_DESIGNS):
        rng = np.random.default_rng(10_000 + i)
        x, y = rng.normal(0, 1, (reps, n)), rng.normal(f * MARGIN, 1, (reps, n))
        dec = tost_batch(x, y, MARGIN)
        e, d = dec["equivalent"], dec["different"]
        hits = {"equiv_only": int(np.sum(e & ~d)), "both": int(np.sum(e & d)),
                "inconclusive": int(np.sum(~e & ~d)), "diff_only": int(np.sum(~e & d))}
        exact = outcome_probs(n, f * MARGIN)
        inside = all(wilson(hits[o], reps)[0] <= exact[o] <= wilson(hits[o], reps)[1] for o in OUTCOMES)
        rows.append({"n": n, "delta_frac": f, "mc": {o: hits[o] / reps for o in OUTCOMES},
                     "exact": exact, "inside_99": inside})
    return rows


def exemplar(n: int = 20, f: float = 1.0) -> Dict:
    """First seed where the TRUE difference equals the margin, the t-test says 'no significant
    difference' (p > 0.25, so nobody would call it borderline) and TOST is inconclusive."""
    for seed in range(10_000):
        rng = np.random.default_rng(seed)
        x = np.round(rng.normal(100, 20, n), 1)  # basket value, $; SD 20, margin $10 = 0.5 SD
        y = np.round(rng.normal(100 + f * MARGIN * 20, 20, n), 1)
        r = analyse(x, y, MARGIN * 20)
        if r["p_diff"] > 0.25 and r["outcome"] == "inconclusive":
            return {"seed": seed, "x": x.tolist(), "y": y.tolist(), "margin": MARGIN * 20,
                    "true_delta": f * MARGIN * 20, **r}
    raise RuntimeError("no exemplar found")
