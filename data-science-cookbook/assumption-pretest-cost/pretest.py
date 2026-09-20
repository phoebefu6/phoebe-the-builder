"""Testing an assumption and then choosing a test is one procedure, and it has its own error rate.

The move under audit: run Levene (or Bartlett, or an F-test) on the two groups; if it rejects
equal variances use Welch's t, otherwise use Student's pooled t. It is taught as due diligence.
It is actually a single two-stage procedure, and the only honest way to describe it is to measure
what IT does - not what its two branches do separately.

Two things make it worse than it looks, and both are measurable rather than arguable:

1. **The pretest reads the same numbers the t-test does.** Levene's decision is a function of the
   sample variances, and Student's pooled standard error is a function of the same sample
   variances. So "Levene passed" is not a random subset of samples - it is the subset whose
   variance ratio happened to look small, which is exactly the subset where the pooled estimate
   is most flattering. The selection is correlated with the statistic it selects.
2. **The pretest has a power curve.** At the small n where Student's is most fragile, Levene has
   the least power to notice, so the gate waves through precisely the samples that needed Welch.

Deliberately NOT here: the normality pretest (Shapiro, then switch to a rank test). That is the
sibling build `normality-test-trap`, which measured that gate at sensitivity 0.81 / specificity
0.38. This build owns the VARIANCE pretest only.

Note on duplication: `wilson()` and the interval-based verdict rule also appear in
`../normality-test-trap/normality.py`. That is deliberate - each build has to run standalone from
a bare Colab link, so it cannot import a sibling. The rule is duplicated ACROSS builds and has
exactly one copy WITHIN this one; the chart and the notebook read this module's verdict rather
than recomputing it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
# The band a nominal-0.05 procedure is entitled to sit in: +/-10% relative.
BAND_LO, BAND_HI = 0.045, 0.055
Z99 = 2.5758293035489004

PRETESTS: Tuple[str, ...] = ("levene", "brown-forsythe", "bartlett", "f-test")
DISTRIBUTIONS: Tuple[str, ...] = ("normal", "t5", "lognormal", "uniform")


def wilson(rate: float, n: int, z: float = Z99) -> Tuple[float, float]:
    """Wilson score interval for a measured proportion.

    Wilson rather than the normal approximation because these rates sit near 0.05, where
    p +/- z*sqrt(p(1-p)/n) is noticeably wrong and can run below zero.
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def is_broken(rate: float, reps: int) -> bool:
    """True only when the WHOLE 99% interval clears the band - never the point estimate alone."""
    lo, hi = wilson(rate, reps)
    return hi < BAND_LO or lo > BAND_HI


# ---------------------------------------------------------------------------
# Drawing data
# ---------------------------------------------------------------------------


def draw(
    dist: str, n1: int, n2: int, sd1: float, sd2: float, reps: int, shift: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Two groups, each standardised to unit scale then multiplied by its own sd.

    `shift` is the TRUE difference in means: 0 makes the null exactly true, so every rejection
    counted is a false positive. Standardisation uses population moments, never sample ones -
    dividing by a sample sd would make the null true only on average.
    """
    def _unit(n: int) -> np.ndarray:
        if dist == "normal":
            return rng.standard_normal((reps, n))
        if dist == "t5":
            return rng.standard_t(5, (reps, n)) / np.sqrt(5.0 / 3.0)
        if dist == "uniform":
            return (rng.random((reps, n)) - 0.5) * np.sqrt(12.0)
        if dist == "lognormal":
            s = 0.5
            raw = rng.lognormal(0.0, s, (reps, n))
            mean = np.exp(s * s / 2.0)
            sd = np.sqrt((np.exp(s * s) - 1.0) * np.exp(s * s))
            return (raw - mean) / sd
        raise ValueError(f"unknown distribution: {dist!r}")

    return _unit(n1) * sd1, _unit(n2) * sd2 + shift


# ---------------------------------------------------------------------------
# The two t-tests, vectorised, from the formulas
# ---------------------------------------------------------------------------


def student_p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sample t with a POOLED variance. Assumes the two populations share one sigma."""
    n1, n2 = x.shape[1], y.shape[1]
    v1, v2 = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
    df = n1 + n2 - 2
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    se = np.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (x.mean(axis=1) - y.mean(axis=1)) / se
    t = np.where(np.isfinite(t), t, 0.0)
    return 2.0 * stats.t.sf(np.abs(t), df)


def welch_p(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Two-sample t with SEPARATE variances and Satterthwaite degrees of freedom."""
    n1, n2 = x.shape[1], y.shape[1]
    a, b = x.var(axis=1, ddof=1) / n1, y.var(axis=1, ddof=1) / n2
    se2 = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        df = se2**2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
        t = (x.mean(axis=1) - y.mean(axis=1)) / np.sqrt(se2)
    t = np.where(np.isfinite(t), t, 0.0)
    df = np.where(np.isfinite(df), df, 1.0)
    return 2.0 * stats.t.sf(np.abs(t), df)


# ---------------------------------------------------------------------------
# The four pretests
# ---------------------------------------------------------------------------


def pretest_p(name: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p-value of the equal-variance pretest on each row.

    Four procedures that all answer "are the variances equal", and they do not agree:

    * `levene` - ANOVA on absolute deviations from each group's MEAN (scipy center='mean')
    * `brown-forsythe` - the same on deviations from each group's MEDIAN; this is what
      `scipy.stats.levene` actually does by default, and it is the robust one
    * `bartlett` - the normal-theory likelihood-ratio test; famously a normality detector
      wearing a variance test's name
    * `f-test` - the textbook ratio of sample variances against an F distribution
    """
    reps = x.shape[0]
    out = np.empty(reps)
    if name == "f-test":
        # Vectorised: the two-sided F test on the variance ratio.
        n1, n2 = x.shape[1], y.shape[1]
        v1, v2 = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = v1 / v2
        f = np.where(np.isfinite(f) & (f > 0), f, 1.0)
        lo = stats.f.cdf(f, n1 - 1, n2 - 1)
        return np.clip(2.0 * np.minimum(lo, 1.0 - lo), 0.0, 1.0)

    center = {"levene": "mean", "brown-forsythe": "median"}.get(name)
    for i in range(reps):
        try:
            if name == "bartlett":
                out[i] = stats.bartlett(x[i], y[i]).pvalue
            else:
                out[i] = stats.levene(x[i], y[i], center=center).pvalue
        except ValueError:
            # Degenerate sample (zero variance in both groups): the statistic does not exist.
            # p=1.0 is the conservative reading - "no evidence of unequal variance".
            out[i] = 1.0
    return np.nan_to_num(out, nan=1.0)


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    """One experimental design: the thing an analyst actually has in front of them."""

    n1: int
    n2: int
    sd_ratio: float = 1.0
    dist: str = "normal"
    shift: float = 0.0

    @property
    def balanced(self) -> bool:
        return self.n1 == self.n2

    @property
    def bigger_group_has_bigger_sd(self) -> Optional[bool]:
        """Which way the pairing runs. This, not the ratio alone, is what breaks Student's.

        Returns None on a balanced design or equal variances, where the question is meaningless.
        """
        if self.sd_ratio == 1.0 or self.balanced:
            return None
        return (self.n1 > self.n2) == (1.0 > self.sd_ratio)

    @property
    def label(self) -> str:
        return f"n={self.n1}/{self.n2} sd=1:{self.sd_ratio:g}"


@dataclass(frozen=True)
class CellResult:
    """Every rate one design produced, all from the SAME simulated samples."""

    design: Design
    pretest: str
    reps: int
    alpha: float
    pretest_alpha: float
    pretest_reject_rate: float
    student_error: float
    welch_error: float
    gated_error: float
    # The mechanism: what the t-tests do on the two halves the pretest carved out.
    student_error_given_pass: float
    student_error_given_reject: float
    share_passed: float

    @property
    def gated_interval(self) -> Tuple[float, float]:
        return wilson(self.gated_error, self.reps)

    @property
    def gated_is_broken(self) -> bool:
        return is_broken(self.gated_error, self.reps)

    @property
    def welch_is_broken(self) -> bool:
        return is_broken(self.welch_error, self.reps)

    @property
    def student_is_broken(self) -> bool:
        return is_broken(self.student_error, self.reps)

    @property
    def gated_beats_welch(self) -> bool:
        """Is the two-stage procedure closer to nominal than just using Welch unconditionally?

        This is the question the whole build exists to answer, and it is deliberately phrased so
        that the answer CAN be yes. A bench that cannot report the good outcome is not a bench.
        """
        return abs(self.gated_error - self.alpha) < abs(self.welch_error - self.alpha)

    @property
    def gated_beats_welch_materially(self) -> bool:
        """The same comparison, but only counted when the gap is bigger than the measurement.

        Two rates differing by 0.0006 at 20,000 replicates are not two different rates - the 99%
        interval here is ~0.008 wide. Reporting a bare win/loss tally over a grid would let
        coin-flips masquerade as wins in both directions, so the headline claim uses this one.
        """
        half_width = (wilson(self.alpha, self.reps)[1] - wilson(self.alpha, self.reps)[0]) / 2
        return abs(self.gated_error - self.alpha) + half_width < abs(self.welch_error - self.alpha)

    @property
    def welch_controls_type_one(self) -> bool:
        """Does Welch hold its nominal rate here? Gates whether a power comparison is meaningful."""
        return not is_broken(self.welch_error, self.reps)

    @property
    def student_controls_type_one(self) -> bool:
        return not is_broken(self.student_error, self.reps)

    @property
    def student_type_one_direction(self) -> str:
        """"ok" | "inflated" | "conservative" - which decides how to read a power number.

        Three outcomes, not two, because the two failures mean opposite things for power:
        an INFLATED test's extra detections are the same inflation showing up again and prove
        nothing, while a CONSERVATIVE test's missed detections are a real, honestly-paid cost.
        Collapsing both into "not comparable" throws away the one that is a finding.
        """
        if not is_broken(self.student_error, self.reps):
            return "ok"
        return "inflated" if self.student_error > self.alpha else "conservative"


def run_cell(
    design: Design,
    pretest: str = "brown-forsythe",
    reps: int = 20_000,
    alpha: float = ALPHA,
    pretest_alpha: float = ALPHA,
    seed: int = 0,
) -> CellResult:
    """Run all three procedures on ONE set of simulated samples.

    Sharing the samples is the entire point. The gated procedure selects its test using the same
    variances the pooled test then uses, so the selection and the statistic are correlated -
    and only a shared-sample simulation can see that. Running the pretest and the t-tests as
    separate simulations would average the correlation away and make the procedure look fine.
    """
    rng = np.random.default_rng(seed)
    sd1, sd2 = 1.0, design.sd_ratio
    x, y = draw(design.dist, design.n1, design.n2, sd1, sd2, reps, design.shift, rng)

    pre_rejects = pretest_p(pretest, x, y) < pretest_alpha
    sp, wp = student_p(x, y), welch_p(x, y)
    s_rej, w_rej = sp < alpha, wp < alpha
    gated = np.where(pre_rejects, w_rej, s_rej)

    passed = ~pre_rejects
    return CellResult(
        design=design,
        pretest=pretest,
        reps=reps,
        alpha=alpha,
        pretest_alpha=pretest_alpha,
        pretest_reject_rate=float(pre_rejects.mean()),
        student_error=float(s_rej.mean()),
        welch_error=float(w_rej.mean()),
        gated_error=float(gated.mean()),
        student_error_given_pass=float(s_rej[passed].mean()) if passed.any() else float("nan"),
        student_error_given_reject=float(s_rej[pre_rejects].mean()) if pre_rejects.any() else float("nan"),
        share_passed=float(passed.mean()),
    )


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

# The designs an analyst actually meets. Both pairings of unequal n with unequal sd are here,
# because Day 174 found the DIRECTION of the pairing matters more than the ratio: the same 3x
# variance gap is nearly harmless when the big group has the big variance and severe when it
# does not. A grid that only contains one direction would miss half the subject.
DESIGNS: Tuple[Design, ...] = (
    # Calibration: Student's is EXACT here, so this cell must read 0.0500.
    Design(20, 20, 1.0),
    # Balanced but unequal variance - Day 174 found Student's barely cares (1.13x at a 6x gap).
    Design(20, 20, 3.0),
    Design(50, 50, 3.0),
    Design(15, 15, 5.0),
    # Unequal n, equal variance - nothing wrong, so any pretest rejection here is pure false alarm.
    Design(50, 10, 1.0),
    # The dangerous pairing: big group, SMALL variance. Student's inflates. The ratios matter more
    # than the extremes, because a moderate gap is where the pretest has the least power AND the
    # analyst feels most reassured by a p > 0.05.
    Design(50, 10, 1.5),
    Design(50, 10, 2.0),
    Design(50, 10, 3.0),
    Design(30, 10, 1.5),
    Design(30, 10, 2.0),
    Design(100, 20, 5.0),
    # The other pairing: big group, BIG variance. Student's goes ultra-conservative, which is a
    # power catastrophe rather than a false-positive one - and it is invisible to a Type I audit.
    Design(10, 50, 3.0),
    Design(10, 30, 3.0),
    Design(20, 100, 5.0),
)

REPS = 20_000


def run_grid(
    pretest: str = "brown-forsythe",
    designs: Optional[Tuple[Design, ...]] = None,
    reps: int = REPS,
    pretest_alpha: float = ALPHA,
    seed: int = 0,
) -> List[CellResult]:
    designs = designs or DESIGNS
    return [run_cell(d, pretest, reps, ALPHA, pretest_alpha, seed=seed + 17 * i) for i, d in enumerate(designs)]


def noise_floor(runs: int = 10, reps: int = REPS) -> Tuple[float, float]:
    """The harness measuring itself on the one cell whose answer is known exactly.

    Student's t on a balanced, equal-variance, normal design is EXACT - its error rate is alpha
    with no approximation anywhere. The spread of this cell across seeds is therefore pure
    Monte-Carlo noise, and no verdict below may rest on a gap narrower than it.
    """
    vals = [
        run_cell(Design(20, 20, 1.0), "brown-forsythe", reps, seed=9000 + i).student_error
        for i in range(runs)
    ]
    return min(vals), max(vals)


# ---------------------------------------------------------------------------
# Scoring the gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateScore:
    """The pretest scored as what it is used for: a decision about whether Student's is safe.

    Positive = the pretest rejects equal variances on the majority of samples from this design.
    Condition = Student's real error rate is outside [0.045, 0.055].
    """

    cells: int
    unsafe_cells: int
    sensitivity: float
    specificity: float
    false_alarm_cells: int
    missed_cells: int


def score_gate(cells: List[CellResult], threshold: float = 0.50) -> GateScore:
    fires = [c.pretest_reject_rate >= threshold for c in cells]
    unsafe = [c.student_is_broken for c in cells]
    tp = sum(1 for f, u in zip(fires, unsafe) if f and u)
    fp = sum(1 for f, u in zip(fires, unsafe) if f and not u)
    fn = sum(1 for f, u in zip(fires, unsafe) if not f and u)
    tn = sum(1 for f, u in zip(fires, unsafe) if not f and not u)
    return GateScore(
        cells=len(cells),
        unsafe_cells=sum(unsafe),
        sensitivity=tp / (tp + fn) if (tp + fn) else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        false_alarm_cells=fp,
        missed_cells=fn,
    )


def worst_cell(cells: List[CellResult]) -> CellResult:
    """The design where the two-stage procedure is furthest from what it promised."""
    return max(cells, key=lambda c: abs(c.gated_error - c.alpha))


def power_cost(
    design: Design, shift: float, pretest: str = "brown-forsythe", reps: int = REPS, seed: int = 7
) -> Dict[str, float]:
    """Power of each procedure at a real effect. A safer test that never detects anything is not safer."""
    d = Design(design.n1, design.n2, design.sd_ratio, design.dist, shift)
    c = run_cell(d, pretest, reps, seed=seed)
    return {"student": c.student_error, "welch": c.welch_error, "gated": c.gated_error}
