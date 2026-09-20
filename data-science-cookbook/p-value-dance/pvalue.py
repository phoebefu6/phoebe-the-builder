"""A p-value is a random variable, and the spread of that variable is the result.

The three sibling builds in this domain (`t-test-variants`, `normality-test-trap`,
`assumption-pretest-cost`) all live under a TRUE NULL and measure how often a procedure rejects
when it should not. This build goes to the other side: the effect is real, the test is correct,
and the question is what the p-value itself does.

Everything below runs Student's two-sample t on equal-n, equal-variance, normally distributed
groups. That is deliberate, and it is the single most important line in this module: Day 174
verified that exact configuration is the one cell where Student's t is EXACT - its Type I error
sits at 0.0500 and its power matches the noncentral-t formula to the replicate noise. So nothing
measured here can be blamed on a violated assumption, a wrong test, or a shaky approximation. The
test is right. The dance is the p-value's own.

Four things get measured:

1. **Calibration.** Under d=0 the p-value is Uniform(0,1) by construction, and measured power
   under d>0 must match the analytic noncentral-t power. Both are checked before any finding is
   read, because a harness that cannot reproduce a known truth cannot be trusted on an unknown
   one (the rule that `normality-test-trap` learned the hard way).
2. **The dance.** Replay ONE fixed true effect at one fixed n and look at the spread of p across
   replicates, in orders of magnitude.
3. **The winner's curse.** Conditional on p < 0.05, how inflated is the observed effect (type M)
   and how often does it point the wrong way (type S).
4. **What a significant result is worth.** The share of significant findings that are false,
   simulated from a mixture of true and null studies rather than quoted from the formula - then
   checked against the formula.

Note on duplication: `wilson()` also appears in `../normality-test-trap/normality.py` and
`../assumption-pretest-cost/pretest.py`. Deliberate - each build must run standalone from a bare
Colab link, so it cannot import a sibling. Duplicated ACROSS builds, exactly one copy WITHIN this
one; the chart, the app and the notebook read this module's verdicts rather than recomputing them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
# The band a nominal-0.05 rate is entitled to sit in: +/-10% relative.
BAND_LO, BAND_HI = 0.045, 0.055
Z99 = 2.5758293035489004

# Drawing in chunks keeps the largest allocation at chunk*n*2 floats rather than reps*n*2.
CHUNK = 2000


def wilson(rate: float, n: int, z: float = Z99) -> Tuple[float, float]:
    """Wilson score interval for a measured proportion.

    Wilson rather than the normal approximation because several of these rates sit near 0.05 or
    near 1.0, where p +/- z*sqrt(p(1-p)/n) is noticeably wrong and can leave [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_differs(rate_a: float, n_a: int, rate_b: float, n_b: int) -> bool:
    """True only when two measured rates' 99% Wilson intervals do not overlap.

    Any comparison of two measured numbers needs a materiality rule or coin-flips masquerade as
    findings (the error `assumption-pretest-cost` shipped and then fixed).
    """
    lo_a, hi_a = wilson(rate_a, n_a)
    lo_b, hi_b = wilson(rate_b, n_b)
    return hi_a < lo_b or hi_b < lo_a


def analytic_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Exact two-sided power of Student's two-sample t at effect d, n per group.

    Noncentral t with df = 2n-2 and noncentrality d*sqrt(n/2). This is the known truth the
    simulation is calibrated against, not a rule of thumb.
    """
    df = 2 * n - 2
    nc = d * math.sqrt(n / 2.0)
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))


def required_n(d: float, target: float = 0.80, alpha: float = ALPHA) -> int:
    """Smallest n per group reaching `target` power at effect d. Brute force; the grid is small."""
    for n in range(4, 20001):
        if analytic_power(d, n, alpha) >= target:
            return n
    return -1


# ---------------------------------------------------------------------------
# One cell: reps replications of the same study
# ---------------------------------------------------------------------------


def _one_chunk(d: float, n: int, reps: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return (p, d_hat) for `reps` studies. Group B is shifted by exactly d population SDs."""
    a = rng.standard_normal((reps, n))
    b = rng.standard_normal((reps, n)) + d
    t, p = stats.ttest_ind(b, a, axis=1, equal_var=True)
    # Cohen's d on the pooled sample SD - the estimate a reader would actually report.
    va = a.var(axis=1, ddof=1)
    vb = b.var(axis=1, ddof=1)
    pooled = np.sqrt((va + vb) / 2.0)
    d_hat = (b.mean(axis=1) - a.mean(axis=1)) / pooled
    return np.asarray(p, dtype=float), np.asarray(d_hat, dtype=float)


def simulate(d: float, n: int, reps: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Replay the SAME study `reps` times. Returns (p-values, observed effect sizes)."""
    rng = np.random.default_rng(seed)
    ps: List[np.ndarray] = []
    ds: List[np.ndarray] = []
    done = 0
    while done < reps:
        take = min(CHUNK, reps - done)
        p, dh = _one_chunk(d, n, take, rng)
        ps.append(p)
        ds.append(dh)
        done += take
    return np.concatenate(ps), np.concatenate(ds)


@dataclass
class Cell:
    """Everything one (true effect, sample size) design has to say."""

    d_true: float
    n: int
    reps: int
    power: float
    power_analytic: float
    power_ci: Tuple[float, float]
    calibrated: bool
    p_pcts: Dict[str, float]
    log10_spread: float
    ks_stat: float
    ks_p: float
    uniform_ok: Optional[bool]
    type_m: Optional[float]
    type_s: Optional[float]
    type_s_ci: Optional[Tuple[float, float]]
    d_hat_median_sig: Optional[float]
    seed: int

    @property
    def label(self) -> str:
        return f"d={self.d_true:g}, n={self.n}"


def analyse(d_true: float, n: int, reps: int, seed: int) -> Cell:
    p, d_hat = simulate(d_true, n, reps, seed)
    sig = p < ALPHA
    power = float(sig.mean())
    exact = analytic_power(d_true, n) if d_true > 0 else ALPHA

    pcts = {f"p{q}": float(np.percentile(p, q)) for q in (1, 5, 25, 50, 75, 95, 99)}
    # The dance, stated in the unit people actually argue in: orders of magnitude.
    spread = float(math.log10(pcts["p95"]) - math.log10(max(pcts["p5"], 1e-300)))

    # Under the null p must be Uniform(0,1) exactly. Under an alternative it must not be.
    ks = stats.kstest(p, "uniform")
    ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    uniform_ok = bool(ks_p > 0.01) if d_true == 0 else None

    type_m = type_s = d_hat_med = None
    type_s_ci = None
    if d_true > 0 and sig.sum() > 0:
        winners = d_hat[sig]
        # Type M (exaggeration): how big the published estimate is, given it got published.
        type_m = float(np.mean(np.abs(winners)) / d_true)
        # Type S: significant AND pointing the wrong way.
        wrong = int((winners < 0).sum())
        type_s = wrong / int(sig.sum())
        type_s_ci = wilson(type_s, int(sig.sum()))
        d_hat_med = float(np.median(winners))

    lo, hi = wilson(power, reps)
    # Calibrated when the analytic truth lies inside the measured 99% interval.
    calibrated = bool(lo <= exact <= hi)

    return Cell(
        d_true=d_true, n=n, reps=reps, power=power, power_analytic=exact, power_ci=(lo, hi),
        calibrated=calibrated, p_pcts=pcts, log10_spread=spread, ks_stat=ks_stat, ks_p=ks_p,
        uniform_ok=uniform_ok, type_m=type_m, type_s=type_s, type_s_ci=type_s_ci,
        d_hat_median_sig=d_hat_med, seed=seed,
    )


# ---------------------------------------------------------------------------
# The grid. The seed is the INDEX into DESIGNS, so the list order is load-bearing.
# ---------------------------------------------------------------------------

NS: Tuple[int, ...] = (10, 20, 50, 100, 500)
DS: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.8)
DESIGNS: Tuple[Tuple[float, int], ...] = tuple((d, n) for d in DS for n in NS)
REPS = 40000


def run_grid(reps: int = REPS, designs: Sequence[Tuple[float, int]] = DESIGNS) -> List[Cell]:
    return [analyse(d, n, reps, seed=i) for i, (d, n) in enumerate(designs)]


# ---------------------------------------------------------------------------
# The replication question: you saw p = 0.05 once. What does the rerun say?
# ---------------------------------------------------------------------------


@dataclass
class Replication:
    d_true: float
    n: int
    window: Tuple[float, float]
    hits: int
    rep_p_pcts: Dict[str, float]
    rep_sig_rate: float
    rep_sig_ci: Tuple[float, float]
    rep_above_10pct: float
    power_analytic: float
    differs_from_power: bool

    @property
    def label(self) -> str:
        return f"d={self.d_true:g}, n={self.n}"


def replication_study(
    d_true: float, n: int, reps: int, seed: int, window: Tuple[float, float] = (0.045, 0.055)
) -> Replication:
    """Draw studies; for every one landing in the p-window, rerun it from the SAME truth.

    This is the honest version of "would it replicate": the truth is held fixed and only the
    sample is redrawn, so nothing here is contaminated by uncertainty about the effect itself.
    """
    rng = np.random.default_rng(seed)
    keep: List[np.ndarray] = []
    done = 0
    while done < reps:
        take = min(CHUNK, reps - done)
        p, _ = _one_chunk(d_true, n, take, rng)
        hit = int(((p >= window[0]) & (p <= window[1])).sum())
        if hit:
            rp, _ = _one_chunk(d_true, n, hit, rng)
            keep.append(rp)
        done += take
    rep_p = np.concatenate(keep) if keep else np.array([])
    hits = int(rep_p.size)
    pcts = {f"p{q}": float(np.percentile(rep_p, q)) for q in (5, 25, 50, 75, 95)} if hits else {}
    rate = float((rep_p < ALPHA).mean()) if hits else 0.0
    above = float((rep_p > 0.10).mean()) if hits else 0.0
    exact = analytic_power(d_true, n) if d_true > 0 else ALPHA
    lo, hi = wilson(rate, hits)
    # The claim under test: conditioning on "you saw p = 0.05" buys NOTHING. With the true effect
    # held fixed, replicate p-values are independent, so the replication significance rate should
    # be the plain unconditional power. `differs` is True only if the 99% interval excludes it.
    differs = bool(hits > 0 and not (lo <= exact <= hi))
    return Replication(
        d_true=d_true, n=n, window=window, hits=hits, rep_p_pcts=pcts,
        rep_sig_rate=rate, rep_sig_ci=(lo, hi), rep_above_10pct=above,
        power_analytic=exact, differs_from_power=differs,
    )


# ---------------------------------------------------------------------------
# What a "significant" result is worth, given that most hypotheses are wrong
# ---------------------------------------------------------------------------


@dataclass
class Ppv:
    prior: float
    d_true: float
    n: int
    studies: int
    n_sig: int
    false_share: float
    false_share_ci: Tuple[float, float]
    formula: float
    power: float
    matches_formula: bool


def ppv_study(prior: float, d_true: float, n: int, studies: int, seed: int) -> Ppv:
    """Run a MIXTURE of studies: a share `prior` are real, the rest are exactly null.

    Simulated rather than quoted, then checked against
    alpha*(1-prior) / (alpha*(1-prior) + prior*power) - so the formula is a test of the
    simulation, not a substitute for it.
    """
    rng = np.random.default_rng(seed)
    is_real_all: List[np.ndarray] = []
    sig_all: List[np.ndarray] = []
    done = 0
    while done < studies:
        take = min(CHUNK, studies - done)
        real = rng.random(take) < prior
        a = rng.standard_normal((take, n))
        b = rng.standard_normal((take, n)) + np.where(real, d_true, 0.0)[:, None]
        _, p = stats.ttest_ind(b, a, axis=1, equal_var=True)
        is_real_all.append(real)
        sig_all.append(np.asarray(p, dtype=float) < ALPHA)
        done += take
    real = np.concatenate(is_real_all)
    sig = np.concatenate(sig_all)
    n_sig = int(sig.sum())
    false_share = float((sig & ~real).sum() / n_sig) if n_sig else 0.0
    power = analytic_power(d_true, n)
    formula = ALPHA * (1 - prior) / (ALPHA * (1 - prior) + prior * power)
    lo, hi = wilson(false_share, n_sig)
    return Ppv(
        prior=prior, d_true=d_true, n=n, studies=studies, n_sig=n_sig,
        false_share=false_share, false_share_ci=(lo, hi), formula=formula,
        power=power, matches_formula=bool(lo <= formula <= hi),
    )


# ---------------------------------------------------------------------------
# Predicates the findings are stated as - each must be able to return either way
# ---------------------------------------------------------------------------


def dance_is_wide(cell: Cell, orders: float = 2.0) -> bool:
    """True when the central 90% of p spans more than `orders` orders of magnitude."""
    return cell.log10_spread > orders


def dance_has_stopped(cell: Cell, ceiling: float = 1e-4) -> bool:
    """True when even the WORST of the central 90% of replicates is decisively significant.

    The negative result this build is looking for: the dance is a property of underpowered
    designs, not of p-values as such. If no cell satisfies this, that claim is unsupported.
    """
    return cell.p_pcts.get("p95", 1.0) < ceiling


def dance_crosses_threshold(cell: Cell) -> bool:
    """True when the central 90% of replicates straddles 0.05 - i.e. an identical rerun of this
    same study routinely lands on the OTHER side of the verdict.

    This, not the raw spread, is the decision-relevant statistic. The raw spread turns out to be
    widest at HIGH power (see evidence.txt), which is exactly where it changes nothing.

    Under the null the straddle is degenerate - p is Uniform(0,1), so its 5th percentile IS alpha
    and the predicate flips on replicate noise. It is defined as False there rather than left to
    be read as a finding.

    Worth stating plainly because it is the build's sharpest negative result: for d > 0 this
    predicate is EXACTLY `power < 0.95`, since p95 < alpha if and only if P(p < alpha) > 0.95.
    The decision-relevant dance is not an extra hazard sitting on top of low power. It is low
    power, restated in a more alarming vocabulary.
    """
    if cell.d_true == 0:
        return False
    return cell.p_pcts["p5"] < ALPHA < cell.p_pcts["p95"]
