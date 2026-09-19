"""A normality test and a t-test do not ask the same question, and one does not gate the other.

The everyday move is: run Shapiro-Wilk, and if it rejects, abandon the t-test. This module
measures what that move actually buys, by running both procedures on the SAME simulated data
under a null that is exactly true, so every rejection observed is a false positive by
construction.

Two facts make the move unsound, and both are measurable rather than arguable:

1. A normality test's rejection rate is a POWER curve in n. On a fixed non-normal population
   it misses at small n and rejects with certainty at large n. The population never changed;
   only the sample size did.
2. The t-test does not require the DATA to be normal. It requires the sampling distribution of
   the mean to be close enough to normal - which the CLT delivers for large n on exactly the
   populations where Shapiro is most certain to reject.

So the two curves run in OPPOSITE directions in n, and the gate fires hardest where it is least
needed. Everything here is a measured rejection rate, not a claim.

Deliberately NOT here: the variance pretest (Levene/Bartlett/F then pick Student or Welch).
That is its own backlog item, `assumption-pretest-cost`, with its own finding. This build owns
the NORMALITY pretest only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05

# The band a nominal-0.05 test is entitled to sit in: +/-10% relative.
BAND_LO, BAND_HI = 0.045, 0.055
# Two-sided 99% normal quantile, used for the Wilson interval on each measured rate.
Z99 = 2.5758293035489004

# Every population is standardised to mean 0 and sd 1 before use. The null tested is
# "the population mean is 0", so it is EXACTLY true in every cell and any rejection is a
# false positive. Standardising also means the only thing that varies across populations
# is SHAPE - skew and tail weight - which is the whole subject.
DISTRIBUTIONS: Tuple[str, ...] = (
    "normal",
    "uniform",
    "t5",
    "contaminated",
    "lognormal-mild",
    "exponential",
    "lognormal-strong",
)

# Skewness and excess kurtosis of each population, computed analytically where a closed form
# exists. Used to label the charts and to sanity-check the samplers in the test suite.
SHAPE: Dict[str, Tuple[float, float]] = {
    "normal": (0.0, 0.0),
    "uniform": (0.0, -1.2),
    "t5": (0.0, 6.0),
    "contaminated": (0.0, 16.9587),
    "lognormal-mild": (1.7502, 5.8984),
    "exponential": (2.0, 6.0),
    "lognormal-strong": (6.1849, 110.9364),
}


def draw(dist: str, n: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `reps` samples of size `n`, standardised to population mean 0 and sd 1.

    Returns shape (reps, n). Standardisation uses the POPULATION moments, never the sample
    ones - dividing by a sample sd would make the null true only on average and would quietly
    change the test being run.
    """
    if dist == "normal":
        return rng.standard_normal((reps, n))
    if dist == "uniform":
        # Uniform(0,1): mean 1/2, sd 1/sqrt(12).
        return (rng.random((reps, n)) - 0.5) * np.sqrt(12.0)
    if dist == "t5":
        # t with 5 df: variance nu/(nu-2) = 5/3.
        return rng.standard_t(5, (reps, n)) / np.sqrt(5.0 / 3.0)
    if dist == "contaminated":
        # 95% N(0,1) + 5% N(0,5): symmetric, but a fat tail. Variance = .95 + .05*25 = 2.2.
        base = rng.standard_normal((reps, n))
        wide = rng.standard_normal((reps, n)) * 5.0
        pick = rng.random((reps, n)) < 0.05
        return np.where(pick, wide, base) / np.sqrt(2.2)
    if dist in ("lognormal-mild", "lognormal-strong"):
        s = 0.5 if dist == "lognormal-mild" else 1.0
        raw = rng.lognormal(0.0, s, (reps, n))
        mean = np.exp(s * s / 2.0)
        sd = np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s))
        return (raw - mean) / sd
    if dist == "exponential":
        # Exponential(1): mean 1, sd 1.
        return rng.exponential(1.0, (reps, n)) - 1.0
    raise ValueError(f"unknown distribution: {dist!r}")


# ---------------------------------------------------------------------------
# The two procedures, implemented so the notebook can carry a self-contained copy.
# ---------------------------------------------------------------------------


def shapiro_reject(samples: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Shapiro-Wilk on each row. Returns a boolean array: True where normality is REJECTED.

    scipy caps the exact null distribution at n=5000 and warns above it, so the caller is
    responsible for staying inside that range; `GRID_N` does.
    """
    out = np.empty(samples.shape[0], dtype=bool)
    for i in range(samples.shape[0]):
        out[i] = stats.shapiro(samples[i]).pvalue < alpha
    return out


def one_sample_t_p(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One-sample t against mu=0 on each row, vectorised. Returns (t statistic, two-sided p).

    Written from the formula rather than looped through scipy because the grid runs millions
    of tests; test_normality.py checks it against scipy.stats.ttest_1samp to 1e-12.
    """
    n = samples.shape[1]
    mean = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / (sd / np.sqrt(n))
    t = np.where(np.isfinite(t), t, 0.0)
    p = 2.0 * stats.t.sf(np.abs(t), n - 1)
    return t, p


def wilcoxon_p(samples: np.ndarray) -> np.ndarray:
    """Wilcoxon signed-rank against a centre of 0 on each row.

    This is the test people switch TO when Shapiro rejects, so the conditional procedure needs
    it. It is looped because scipy's exact/asymptotic switch is not vectorised.
    """
    out = np.empty(samples.shape[0], dtype=float)
    for i in range(samples.shape[0]):
        try:
            out[i] = stats.wilcoxon(samples[i]).pvalue
        except ValueError:
            # All-zero differences: the statistic does not exist. p=1.0 is the conservative
            # reading, and it is the same convention the sibling t-test-variants build used.
            out[i] = 1.0
    return out


def wilson(rate: float, n: int, z: float = Z99) -> Tuple[float, float]:
    """Wilson score interval for a measured proportion.

    Wilson rather than the normal approximation because these rates sit near 0.05, where the
    textbook p +/- z*sqrt(p(1-p)/n) interval is noticeably wrong and can run below zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# One grid cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellResult:
    """Everything one (distribution, n) cell measured, under a null that is exactly true."""

    dist: str
    n: int
    reps: int
    alpha: float
    shapiro_reject_rate: float
    t_error: float
    t_error_lower_tail: float
    t_error_upper_tail: float
    wilcoxon_error: float
    conditional_error: float
    t_error_given_shapiro_passed: float
    t_error_given_shapiro_rejected: float

    @property
    def t_error_interval(self) -> Tuple[float, float]:
        """99% Wilson interval on this cell's measured error rate, given ITS replicate count.

        Replicates are tiered across the grid, so a raw point estimate is not comparable from
        one cell to the next: 0.043 on 20,000 reps is a real departure, and 0.043 on 1,500 reps
        is a coin landing slightly oddly. The interval carries that difference.
        """
        return wilson(self.t_error, self.reps)

    @property
    def t_is_broken(self) -> bool:
        """Is the t-test's real error rate outside [0.045, 0.055], by a margin this cell can see?

        A cell counts as broken only when its whole 99% interval clears the band. The first
        version of this property compared the point estimate to the band and immediately flagged
        the CALIBRATION cell - a one-sample t on normal data, which is exact - as broken at
        n=5000, purely because that tier runs 1,500 replicates. A detector that fires on its own
        known-good cell is not measuring the thing it claims to measure.
        """
        lo, hi = self.t_error_interval
        return hi < BAND_LO or lo > BAND_HI

    @property
    def resolvable(self) -> bool:
        """Could this cell be flagged broken at all, at its replicate count?

        If a perfectly nominal 0.05 result already has an interval overlapping the band edges by
        more than the band is wide, the cell can only ever report "not broken" - which is an
        absence of evidence, not evidence of absence. Reported rather than hidden.
        """
        lo, hi = wilson(BAND_HI, self.reps)
        return (hi - lo) < 2 * (BAND_HI - BAND_LO)

    @property
    def tail_asymmetry(self) -> float:
        """Ratio of the larger one-sided error to the smaller.

        A two-sided rate of 0.05 can be 0.045 in one tail and 0.005 in the other. That test is
        not a 5% test; it is a 4.5% test pointing one way. The two-sided number hides it.
        """
        lo, hi = self.t_error_lower_tail, self.t_error_upper_tail
        small, large = min(lo, hi), max(lo, hi)
        return float("inf") if small == 0 else large / small


def run_cell(
    dist: str,
    n: int,
    reps: int,
    alpha: float = ALPHA,
    seed: int = 0,
    include_wilcoxon: bool = True,
) -> CellResult:
    """Measure both procedures on the same simulated data, under a true null.

    The SAME samples feed Shapiro, the t-test and Wilcoxon. That is the point: the conditional
    procedure picks its test using the data it then tests, so the selection and the statistic
    are correlated, and only a shared-sample simulation can see it.
    """
    rng = np.random.default_rng(seed)
    samples = draw(dist, n, reps, rng)

    sh_rej = shapiro_reject(samples, alpha)
    t_stat, t_p = one_sample_t_p(samples)
    t_rej = t_p < alpha

    if include_wilcoxon:
        w_p = wilcoxon_p(samples)
        w_rej = w_p < alpha
    else:
        w_rej = np.zeros(reps, dtype=bool)

    # The move under audit: if Shapiro rejects, use Wilcoxon; otherwise use the t-test.
    conditional = np.where(sh_rej, w_rej, t_rej)

    passed, rejected = ~sh_rej, sh_rej
    return CellResult(
        dist=dist,
        n=n,
        reps=reps,
        alpha=alpha,
        shapiro_reject_rate=float(sh_rej.mean()),
        t_error=float(t_rej.mean()),
        t_error_lower_tail=float(((t_p < alpha) & (t_stat < 0)).mean()),
        t_error_upper_tail=float(((t_p < alpha) & (t_stat > 0)).mean()),
        wilcoxon_error=float(w_rej.mean()) if include_wilcoxon else float("nan"),
        conditional_error=float(conditional.mean()),
        t_error_given_shapiro_passed=float(t_rej[passed].mean()) if passed.any() else float("nan"),
        t_error_given_shapiro_rejected=float(t_rej[rejected].mean()) if rejected.any() else float("nan"),
    )


# ---------------------------------------------------------------------------
# The grid, and the diagnostic scorecard built from it
# ---------------------------------------------------------------------------

# Replicates are tiered, but only slightly, and the tiering is set by RESOLUTION rather than by
# cost. Every tier here is chosen so that `CellResult.resolvable` is True - i.e. a 99% Wilson
# interval on a nominal 0.05 result is narrower than twice the [0.045, 0.055] band, so the cell
# is capable of reporting "broken" if it is. The first version of this table dropped to 1,500
# replicates at n=5000 and the harness promptly flagged its own exact calibration cell as broken.
GRID_N: Tuple[int, ...] = (10, 20, 30, 50, 100, 200, 500, 1000, 5000)
REPS_FOR_N: Dict[int, int] = {
    10: 20_000,
    20: 20_000,
    30: 20_000,
    50: 20_000,
    100: 20_000,
    200: 20_000,
    500: 20_000,
    1000: 16_000,
    5000: 8_000,
}


def reps_for(n: int) -> int:
    return REPS_FOR_N.get(n, 4_000)


def run_grid(
    dists: Optional[Tuple[str, ...]] = None,
    ns: Optional[Tuple[int, ...]] = None,
    alpha: float = ALPHA,
    seed: int = 0,
) -> List[CellResult]:
    """Every (distribution, n) cell. Each cell gets its own seed so a cell can be re-run alone."""
    dists = dists or DISTRIBUTIONS
    ns = ns or GRID_N
    out: List[CellResult] = []
    for di, d in enumerate(dists):
        for ni, n in enumerate(ns):
            out.append(run_cell(d, n, reps_for(n), alpha, seed=seed + 1000 * di + ni))
    return out


def noise_floor(n: int = 50, runs: int = 10, alpha: float = ALPHA) -> Tuple[float, float]:
    """The harness measuring itself: same known-truth cell, different seeds.

    A one-sample t-test on standard normal data is EXACT - its error rate is alpha with no
    approximation anywhere. So the spread of this cell across seeds is pure Monte-Carlo noise,
    and no later verdict is allowed to be narrower than it. The sibling build (Day 174) learned
    this the hard way: without a measured floor, a Wilson interval at 40k reps will happily
    flag a 1.05x departure as inflated.
    """
    vals = [run_cell("normal", n, reps_for(n), alpha, seed=9000 + i, include_wilcoxon=False).t_error for i in range(runs)]
    return min(vals), max(vals)


@dataclass(frozen=True)
class DiagnosticScore:
    """Shapiro-Wilk scored as what people actually use it for: a gate on the t-test.

    Positive = "Shapiro rejects, so do not use the t-test". Truth = "the t-test's real error
    rate is outside [0.045, 0.055]". Sensitivity and specificity are then the ordinary
    diagnostic quantities, computed over the grid cells.
    """

    cells: int
    broken_cells: int
    sensitivity: float
    specificity: float
    false_alarm_cells: int
    missed_cells: int
    threshold: float

    @property
    def summary(self) -> str:
        return (
            f"Over {self.cells} cells: sensitivity {self.sensitivity:.2f}, "
            f"specificity {self.specificity:.2f} "
            f"({self.false_alarm_cells} false alarms, {self.missed_cells} misses)"
        )


def score_as_diagnostic(cells: List[CellResult], threshold: float = 0.50) -> DiagnosticScore:
    """Score the gate. `threshold` is the Shapiro rejection rate above which the gate 'fires'.

    A single analyst runs Shapiro once, not 20,000 times - so a cell is treated as one where
    the gate fires when it fires on the majority of samples from that population at that n.
    """
    fires = [c.shapiro_reject_rate >= threshold for c in cells]
    broken = [c.t_is_broken for c in cells]
    tp = sum(1 for f, b in zip(fires, broken) if f and b)
    fp = sum(1 for f, b in zip(fires, broken) if f and not b)
    fn = sum(1 for f, b in zip(fires, broken) if not f and b)
    tn = sum(1 for f, b in zip(fires, broken) if not f and not b)
    return DiagnosticScore(
        cells=len(cells),
        broken_cells=sum(broken),
        sensitivity=tp / (tp + fn) if (tp + fn) else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        false_alarm_cells=fp,
        missed_cells=fn,
        threshold=threshold,
    )


def crossover_n(cells: List[CellResult], dist: str) -> Optional[int]:
    """The smallest n at which Shapiro rejects the majority of samples while the t-test is FINE.

    This is the trap in one number: from here up, the gate fires on a test that works.
    """
    rows = sorted((c for c in cells if c.dist == dist), key=lambda c: c.n)
    for c in rows:
        if c.shapiro_reject_rate >= 0.50 and not c.t_is_broken:
            return c.n
    return None


def blind_spot_n(cells: List[CellResult], dist: str) -> List[int]:
    """Every n where the t-test IS broken but Shapiro would not have warned you."""
    return [c.n for c in sorted((x for x in cells if x.dist == dist), key=lambda x: x.n) if c.t_is_broken and c.shapiro_reject_rate < 0.50]
