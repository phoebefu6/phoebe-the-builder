"""The error bars overlap, so it is not significant.

It is the most common thing said about a chart with error bars on it, and for two independent
means at the same confidence level it is provably WRONG IN ONE DIRECTION: non-overlap implies
significance, but significance does NOT imply non-overlap. The rule throws away real results.
Then, on paired or otherwise correlated measurements, it fails in the OPPOSITE direction, and
the bars carry nothing that would tell a reader which regime they are in.

The whole build rests on one line of algebra. Two independent estimates with standard errors
`s1`, `s2` and intervals drawn at multiplier `z`:

    the bars stop overlapping when   |d| > z * (s1 + s2)
    the test rejects when            |d| > z_alpha * sqrt(s1^2 + s2^2)

and `s1 + s2 >= sqrt(s1^2 + s2^2)` for every non-negative pair, with equality only when one of
them is zero. So at the same multiplier non-overlap is the STRICTER event, always. The ratio

    R = (s1 + s2) / sqrt(s1^2 + s2^2)

runs from `sqrt(2) = 1.414` at equal standard errors down to `1.0` as they diverge - which means
the rule is at its MOST wrong exactly where it looks most reasonable, on two groups of the same
size and spread, and is nearly right in the lopsided case nobody trusts.

What separates this build from its six siblings in this domain:

- `t-test-variants` (Day 174), `assumption-pretest-cost` (Day 176) and `nonparametric-swap`
  (Day 179) interrogate which TEST to run. Nothing here changes the test; the test is correct
  throughout and the reader's eye is the thing being measured.
- `effect-size-reader` (Day 178) measured an ESTIMATE rather than a test. This measures a
  PICTURE of an estimate, which is a third object again: the interval is drawn correctly, and
  the inference drawn off it is still wrong.
- `prediction-interval` (Day 173) is a different interval about a different thing - a future
  observation, not a parameter.

What gets measured:

1. **The geometry, in closed form and checked exhaustively.** `matching_level()` returns the
   confidence level at which non-overlap and significance become the SAME event. At equal
   standard errors it is 83.4%, not 95%, and the value is `z_alpha / sqrt(2)` exactly.
2. **The containment theorem, verified by simulation** - non-overlap implies significance in
   100% of replicates when the intervals use a known sigma, and the converse fails at a
   measured rate. Then the corner where containment genuinely BREAKS: t-intervals at unequal n,
   where the two groups get different multipliers.
3. **The dead zone**, the share of real, significant results whose 95% bars overlap.
4. **The overlap fraction that corresponds to significance** - 58.6% of one arm at equal
   standard errors, and a value that falls to 0 as they diverge, so the folk correction "a
   little overlap is fine" has no single number to be.
5. **The reversal on correlated data.** Paired measurements at correlation rho, where the test
   is overwhelmingly significant while the bars sit almost on top of each other.
6. **Measured coverage of four interval methods for a proportion** - Wald, Wilson,
   Agresti-Coull, Clopper-Pearson - computed EXACTLY by enumerating all n+1 outcomes rather
   than simulated, so those numbers carry no Monte Carlo error at all. Then the overlap rule
   run on each method's bars against a real two-proportion test, also exactly, over all
   (n1+1)(n2+1) outcome pairs.

Note on duplication: `wilson()`, `mc_interval()` and `verdict()` also appear in sibling builds.
Deliberate - every build must run standalone from a bare Colab link, so it cannot import a
neighbour. Duplicated ACROSS builds, exactly one copy WITHIN this one; the chart, the app and
the notebook read this module's numbers rather than recomputing them.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
Z99 = 2.5758293035489004
CHUNK = 4000

# A nominal 5% rate is called broken only when its whole 99% Wilson interval clears this band.
# Carried from Day 175, where comparing a point estimate to a fixed band made the harness flag
# its own exactly-calibrated cell as broken.
BAND_LO, BAND_HI = 0.045, 0.055


def wilson(rate: float, n: int, z: float = Z99) -> Tuple[float, float]:
    """Wilson score interval for a measured proportion - used for every rate verdict here."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mc_interval(values: np.ndarray, z: float = Z99) -> Tuple[float, float]:
    """99% interval on a Monte Carlo mean, so no 'difference' is read off two point estimates."""
    m = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return m - z * se, m + z * se


def verdict(rate: float, n: int) -> str:
    """INFLATED / CONSERVATIVE / ok - the whole interval must clear the band, or it is ok."""
    lo, hi = wilson(rate, n)
    if lo > BAND_HI:
        return "INFLATED"
    if hi < BAND_LO:
        return "CONSERVATIVE"
    return "ok"


# --------------------------------------------------------------------------- 1. the geometry
# Everything in this section is closed form. Nothing is estimated, so nothing here has an
# interval on it, and the tests assert equalities rather than bands.

def z_for_level(level: float) -> float:
    """Two-sided normal multiplier for a confidence level, e.g. 0.95 -> 1.959964."""
    return float(stats.norm.ppf(0.5 + level / 2.0))


def level_for_z(z: float) -> float:
    return float(2.0 * stats.norm.cdf(z) - 1.0)


Z_ALPHA = z_for_level(1.0 - ALPHA)


def se_ratio(s1: float, s2: float) -> float:
    """R = (s1 + s2) / sqrt(s1^2 + s2^2). The single number the whole fallacy turns on.

    Bounded in [1, sqrt(2)]: sqrt(2) when the two standard errors are equal, approaching 1 as
    one of them dominates. It is exactly the factor by which the non-overlap threshold exceeds
    the significance threshold at a common multiplier.
    """
    if s1 < 0 or s2 < 0 or (s1 == 0 and s2 == 0):
        raise ValueError("standard errors must be non-negative and not both zero")
    return (s1 + s2) / math.sqrt(s1 * s1 + s2 * s2)


def matching_level(s1: float, s2: float, alpha: float = ALPHA) -> float:
    """The confidence level whose non-overlap event IS the alpha-level test's rejection event.

    Solve z * (s1 + s2) = z_alpha * sqrt(s1^2 + s2^2)  ->  z = z_alpha / R.
    At equal standard errors this is z_alpha / sqrt(2) = 1.3859, i.e. 83.4% intervals - which
    is why "83% error bars" keeps turning up in the methods sections of people who have thought
    about this, and why it is never 95%.
    """
    return level_for_z(z_for_level(1.0 - alpha) / se_ratio(s1, s2))


MATCHING_LEVEL_EQUAL_SE = matching_level(1.0, 1.0)


def overlap_fraction_at_significance(s1: float, s2: float, level: float = 0.95,
                                     alpha: float = ALPHA) -> float:
    """How much of an arm the bars may still share while the test is exactly at its boundary.

    Overlap is measured as a fraction of the MEAN arm length (an arm being one half-width):

        f = overlap_length / ((arm1 + arm2) / 2)

    At the significance boundary d = z_alpha * sqrt(s1^2 + s2^2), so

        f = 2 - 2 * z_alpha / (z_level * R)

    which is 2 - sqrt(2) = 0.5858 for equal standard errors and 95% bars, and falls to 0 as the
    standard errors diverge. There is no universal "a bit of overlap is fine" number: the
    permitted overlap is a function of the design, not of the picture.
    """
    return 2.0 - 2.0 * z_for_level(1.0 - alpha) / (z_for_level(level) * se_ratio(s1, s2))


def rule_alpha(s1: float, s2: float, level: float = 0.95) -> float:
    """The false-positive rate of the OVERLAP RULE itself, read as a test.

    Under a true null the difference is `N(0, sqrt(s1^2 + s2^2))`, and the rule fires when
    `|d| > z_level * (s1 + s2)`. So the rule is a two-sided test at `z_level * R` standard
    errors, and its size is `2 * (1 - Phi(z_level * R))`. At equal standard errors and 95% bars
    that is 0.0056 - the rule is a 0.6% test wearing a 5% label, which is the whole dead zone
    restated in one number.
    """
    return float(2.0 * stats.norm.sf(z_for_level(level) * se_ratio(s1, s2)))





def paired_se(s1: float, s2: float, rho: float) -> float:
    """SE of the difference when the two estimates are correlated. The bars cannot show rho."""
    v = s1 * s1 + s2 * s2 - 2.0 * rho * s1 * s2
    return math.sqrt(max(v, 0.0))


RULE_ALPHA_EQUAL_SE = rule_alpha(1.0, 1.0)


def paired_slack(s1: float, s2: float, rho: float, level: float = 0.95,
                 alpha: float = ALPHA) -> float:
    """How many times further apart the bars demand the means be than the test does.

    Above 1 the rule is conservative (it discards real results); it rises without bound as rho
    approaches 1. This is the number the independent-groups algebra caps at sqrt(2) and that
    correlation removes the cap from.
    """
    se = paired_se(s1, s2, rho)
    if se <= 0:
        return float("inf")
    return z_for_level(level) * (s1 + s2) / (z_for_level(1.0 - alpha) * se)


# --------------------------------------------------------------- 2. intervals and the rule

def overlaps(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return (lo_a <= hi_b) and (lo_b <= hi_a)


def overlap_share(lo_a: np.ndarray, hi_a: np.ndarray, lo_b: np.ndarray,
                  hi_b: np.ndarray) -> np.ndarray:
    """Vectorised overlap length as a fraction of the mean arm. Negative means a visible gap."""
    inter = np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b)
    mean_arm = ((hi_a - lo_a) + (hi_b - lo_b)) / 4.0
    return inter / mean_arm


def ci_known_sigma(mean: np.ndarray, se: float, level: float = 0.95
                   ) -> Tuple[np.ndarray, np.ndarray]:
    z = z_for_level(level)
    return mean - z * se, mean + z * se


def ci_t(mean: np.ndarray, sd: np.ndarray, n: int, level: float = 0.95
         ) -> Tuple[np.ndarray, np.ndarray]:
    t = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
    half = t * sd / math.sqrt(n)
    return mean - half, mean + half


def welch_p(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Two-sided Welch p-value, rows are replicates. The test is never the thing at fault here."""
    n1, n2 = a.shape[1], b.shape[1]
    m1, m2 = a.mean(axis=1), b.mean(axis=1)
    v1, v2 = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    se2 = v1 / n1 + v2 / n2
    t = (m1 - m2) / np.sqrt(se2)
    df = se2 ** 2 / (v1 ** 2 / (n1 ** 2 * (n1 - 1)) + v2 ** 2 / (n2 ** 2 * (n2 - 1)))
    return 2.0 * stats.t.sf(np.abs(t), df)


def z_test_p(a: np.ndarray, b: np.ndarray, sd1: float, sd2: float) -> np.ndarray:
    """Two-sided z-test using the SAME known standard errors the bars are drawn from.

    This exists to make the containment theorem checkable as an exact logical statement. The
    theorem is about intervals and a test that share their standard errors; the moment the bars
    are drawn from one SE and the test computed from another, it is no longer guaranteed - which
    is measured, not assumed, in `run_means(..., test="welch", known_sigma=True)`.
    """
    n1, n2 = a.shape[1], b.shape[1]
    se = math.sqrt(sd1 * sd1 / n1 + sd2 * sd2 / n2)
    z = (a.mean(axis=1) - b.mean(axis=1)) / se
    return 2.0 * stats.norm.sf(np.abs(z))


def paired_p(d: np.ndarray) -> np.ndarray:
    """Two-sided paired t-test on the differences, rows are replicates."""
    n = d.shape[1]
    t = d.mean(axis=1) / (d.std(axis=1, ddof=1) / math.sqrt(n))
    return 2.0 * stats.t.sf(np.abs(t), n - 1)


def _chunks(reps: int, size: int = CHUNK) -> List[int]:
    out = []
    left = reps
    while left > 0:
        out.append(min(size, left))
        left -= size
    return out


# ------------------------------------------------------------------ 3. the means simulations

def run_means(n1: int, n2: int, sd1: float, sd2: float, delta: float, reps: int, seed: int,
              level: float = 0.95, known_sigma: bool = False,
              test: str = "welch") -> Dict[str, float]:
    """One design, replicated. Returns the rates the whole argument is made of.

    `sig` is the Welch t-test. `gap` is the reader's rule: the two intervals do not touch.
    `sig_and_overlap` is the dead zone - a real result the picture tells you to discard.
    `gap_and_not_sig` is the reverse failure, which the algebra forbids for known-sigma
    intervals and does NOT forbid once each group gets its own t multiplier.
    """
    rng = np.random.default_rng(seed)
    sig = gap = both = sig_ov = gap_ns = 0
    shares: List[np.ndarray] = []
    for m in _chunks(reps):
        a = rng.normal(0.0, sd1, size=(m, n1))
        b = rng.normal(delta, sd2, size=(m, n2))
        p = z_test_p(a, b, sd1, sd2) if test == "z" else welch_p(a, b)
        if known_sigma:
            lo_a, hi_a = ci_known_sigma(a.mean(axis=1), sd1 / math.sqrt(n1), level)
            lo_b, hi_b = ci_known_sigma(b.mean(axis=1), sd2 / math.sqrt(n2), level)
        else:
            lo_a, hi_a = ci_t(a.mean(axis=1), a.std(axis=1, ddof=1), n1, level)
            lo_b, hi_b = ci_t(b.mean(axis=1), b.std(axis=1, ddof=1), n2, level)
        share = overlap_share(lo_a, hi_a, lo_b, hi_b)
        is_sig = p < ALPHA
        is_gap = share < 0.0
        sig += int(is_sig.sum())
        gap += int(is_gap.sum())
        both += int((is_sig & is_gap).sum())
        sig_ov += int((is_sig & ~is_gap).sum())
        gap_ns += int((is_gap & ~is_sig).sum())
        shares.append(share[is_sig])
    sig_shares = np.concatenate(shares) if shares else np.array([0.0])
    return {
        "n1": n1, "n2": n2, "sd1": sd1, "sd2": sd2, "delta": delta, "level": level,
        "reps": reps, "known_sigma": known_sigma, "test": test,
        "sig": sig / reps,
        "gap": gap / reps,
        "sig_and_overlap": sig_ov / reps,
        "gap_and_not_sig": gap_ns / reps,
        "disagree": (sig_ov + gap_ns) / reps,
        "overlap_given_sig": (sig_ov / sig) if sig else float("nan"),
        "median_share_when_sig": float(np.median(sig_shares)) if sig else float("nan"),
        "se_ratio": se_ratio(sd1 / math.sqrt(n1), sd2 / math.sqrt(n2)),
        "matching_level": matching_level(sd1 / math.sqrt(n1), sd2 / math.sqrt(n2)),
    }


def run_paired(n: int, rho: float, delta: float, sd: float, reps: int,
               seed: int, level: float = 0.95) -> Dict[str, float]:
    """The reversal. Bars are drawn per arm, as they always are; the test knows about rho."""
    rng = np.random.default_rng(seed)
    sig = sig_ov = 0
    for m in _chunks(reps):
        z1 = rng.normal(0.0, 1.0, size=(m, n))
        z2 = rho * z1 + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.normal(0.0, 1.0, size=(m, n))
        a = sd * z1
        b = delta + sd * z2
        p = paired_p(b - a)
        lo_a, hi_a = ci_t(a.mean(axis=1), a.std(axis=1, ddof=1), n, level)
        lo_b, hi_b = ci_t(b.mean(axis=1), b.std(axis=1, ddof=1), n, level)
        share = overlap_share(lo_a, hi_a, lo_b, hi_b)
        is_sig = p < ALPHA
        sig += int(is_sig.sum())
        sig_ov += int((is_sig & (share >= 0.0)).sum())
    se = sd / math.sqrt(n)
    return {
        "n": n, "rho": rho, "delta": delta, "reps": reps,
        "sig": sig / reps,
        "sig_and_overlap": sig_ov / reps,
        "overlap_given_sig": (sig_ov / sig) if sig else float("nan"),
        "slack": paired_slack(se, se, rho, level),
    }


# ------------------------------------------------------- 4. proportions, computed exactly
# No Monte Carlo below this line. Coverage and every rule rate are summed over the complete
# outcome space with its exact binomial weights, so these numbers have no sampling error and
# no interval - a real strengthening over the siblings, which could only simulate.

METHODS = ("wald", "wilson", "agresti_coull", "clopper_pearson")


def prop_ci(method: str, x: np.ndarray, n: int, level: float = 0.95
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Interval for a binomial proportion. x may be an array of counts."""
    x = np.asarray(x, dtype=float)
    z = z_for_level(level)
    if method == "wald":
        p = x / n
        half = z * np.sqrt(p * (1.0 - p) / n)
        return np.clip(p - half, 0.0, 1.0), np.clip(p + half, 0.0, 1.0)
    if method == "wilson":
        p = x / n
        denom = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
        return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)
    if method == "agresti_coull":
        n_t = n + z * z
        p_t = (x + z * z / 2.0) / n_t
        half = z * np.sqrt(p_t * (1.0 - p_t) / n_t)
        return np.clip(p_t - half, 0.0, 1.0), np.clip(p_t + half, 0.0, 1.0)
    if method == "clopper_pearson":
        a = 1.0 - level
        lo = np.where(x > 0, stats.beta.ppf(a / 2.0, x, n - x + 1), 0.0)
        hi = np.where(x < n, stats.beta.ppf(1.0 - a / 2.0, x + 1, n - x), 1.0)
        return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)
    raise ValueError(f"unknown method {method!r}")


def exact_coverage(method: str, p: float, n: int, level: float = 0.95) -> float:
    """Exact coverage: enumerate every count 0..n and weight by its binomial probability."""
    x = np.arange(n + 1)
    lo, hi = prop_ci(method, x, n, level)
    w = stats.binom.pmf(x, n, p)
    return float(w[(lo <= p) & (p <= hi)].sum())


def exact_interval_width(method: str, p: float, n: int, level: float = 0.95) -> float:
    x = np.arange(n + 1)
    lo, hi = prop_ci(method, x, n, level)
    return float((stats.binom.pmf(x, n, p) * (hi - lo)).sum())


def two_prop_z(x1: np.ndarray, n1: int, x2: np.ndarray, n2: int) -> np.ndarray:
    """Pooled two-proportion z statistic. Its SE is NOT the one the two bars are drawn from,
    which is why the containment theorem can fail here and cannot fail for two means."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p1 - p2) / np.where(se > 0, se, 1.0), 0.0)
    return z


def exact_prop_rule(method: str, p1: float, p2: float, n1: int, n2: int,
                    level: float = 0.95) -> Dict[str, float]:
    """Every outcome pair, exactly weighted. Returns the two rule failures and the test's rate."""
    x1 = np.arange(n1 + 1)[:, None]
    x2 = np.arange(n2 + 1)[None, :]
    w = stats.binom.pmf(x1, n1, p1) * stats.binom.pmf(x2, n2, p2)
    lo1, hi1 = prop_ci(method, np.arange(n1 + 1), n1, level)
    lo2, hi2 = prop_ci(method, np.arange(n2 + 1), n2, level)
    gap = (lo1[:, None] > hi2[None, :]) | (lo2[None, :] > hi1[:, None])
    z = two_prop_z(x1, n1, x2, n2)
    sig = 2.0 * stats.norm.sf(np.abs(z)) < ALPHA
    return {
        "method": method, "p1": p1, "p2": p2, "n1": n1, "n2": n2,
        "sig": float(w[sig].sum()),
        "gap": float(w[gap].sum()),
        "sig_and_overlap": float(w[sig & ~gap].sum()),
        "gap_and_not_sig": float(w[gap & ~sig].sum()),
    }


# ----------------------------------------------------------------------- 5. the study design
# Seeds are INDICES into these lists, never hashes of their contents: Python salts string
# hashing per process, so a hash-derived seed makes a study that cannot reproduce itself
# (the defect Day 178 shipped and fixed). The notebook imports these same lists.

MEANS_REPS = 40_000
PAIRED_REPS = 40_000
CONTAINMENT_REPS = 200_000

# (n1, n2, sd1, sd2, delta) - delta is chosen per row to land at a stated power, see evidence.py
MEANS_DESIGNS: List[Tuple[int, int, float, float, float]] = [
    (30, 30, 1.0, 1.0, 0.60),
    (30, 30, 1.0, 1.0, 0.80),
    (30, 30, 1.0, 1.0, 1.00),
    (60, 60, 1.0, 1.0, 0.45),
    (60, 60, 1.0, 1.0, 0.60),
    (100, 100, 1.0, 1.0, 0.40),
    (30, 30, 1.0, 3.0, 1.60),
    (30, 30, 1.0, 6.0, 3.00),
    (100, 20, 1.0, 1.0, 0.70),
    (100, 20, 1.0, 3.0, 2.20),
    (20, 20, 1.0, 1.0, 1.00),
    (200, 200, 1.0, 1.0, 0.28),
]

CONTAINMENT_DESIGNS: List[Tuple[int, int, float, float, float]] = [
    (30, 30, 1.0, 1.0, 0.70),
    (100, 20, 1.0, 1.0, 0.70),
    (100, 5, 1.0, 1.0, 1.20),
    (200, 6, 1.0, 2.5, 2.40),
    (40, 8, 1.0, 1.0, 1.00),
]

RATIO_GRID: List[float] = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]
RHO_GRID: List[float] = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
PAIRED_N = 30
PAIRED_DELTA = 0.35
PAIRED_SD = 1.0

COVERAGE_PS: List[float] = [0.02, 0.05, 0.10, 0.25, 0.50]
COVERAGE_NS: List[int] = [20, 50, 100, 400]

PROP_DESIGNS: List[Tuple[float, float, int, int]] = [
    (0.50, 0.62, 100, 100),
    (0.10, 0.18, 200, 200),
    (0.03, 0.08, 150, 150),
    (0.50, 0.66, 60, 60),
    (0.20, 0.34, 80, 80),
]

MEANS_SEED_BASE = 71_000
CONTAINMENT_SEED_BASE = 72_000
PAIRED_SEED_BASE = 73_000
LEVEL_SEED_BASE = 74_000


def delta_for_power(n1: int, n2: int, sd1: float, sd2: float, power: float = 0.80) -> float:
    """Normal-approximation delta hitting a target power. Used only to LABEL designs."""
    se = math.sqrt(sd1 * sd1 / n1 + sd2 * sd2 / n2)
    return (Z_ALPHA + z_for_level(2.0 * power - 1.0)) * se


def approx_power(n1: int, n2: int, sd1: float, sd2: float, delta: float) -> float:
    se = math.sqrt(sd1 * sd1 / n1 + sd2 * sd2 / n2)
    lam = delta / se
    return float(stats.norm.sf(Z_ALPHA - lam) + stats.norm.cdf(-Z_ALPHA - lam))


def geometry_table(ratios: Sequence[float] = tuple(RATIO_GRID)) -> List[Dict[str, float]]:
    """Closed-form only: what the two rules demand, as the standard errors diverge."""
    rows = []
    for r in ratios:
        s1, s2 = 1.0, float(r)
        rows.append({
            "sd_ratio": float(r),
            "se_ratio": se_ratio(s1, s2),
            "matching_level": matching_level(s1, s2),
            "overlap_at_sig": overlap_fraction_at_significance(s1, s2),
            "threshold_inflation": se_ratio(s1, s2),
        })
    return rows
