"""Mann-Whitney is not a robust t-test. It is a test of a different hypothesis.

"The data is not normal, just use Mann-Whitney" is the single most-repeated piece of statistical
advice in applied work, and it quietly substitutes one question for another. A two-sample t-test
asks about `mu_B - mu_A`. The Mann-Whitney U test asks about `P(B > A)`. Under a pure location
shift of one distribution those two agree in sign, and the swap is harmless. Off that model they
can point in OPPOSITE directions, both correctly, on the same data - and at a large enough sample
both will be significant while disagreeing.

What separates this build from its four siblings in this domain:

- `normality-test-trap` (Day 175) measured the ONE-sample signed-rank swap, and found Wilcoxon
  running at 0.98 on skewed data under a true mean-null because it tests symmetry, not the mean.
  This is the TWO-sample case, where the substituted hypothesis is stochastic dominance.
- `assumption-pretest-cost` (Day 176) measured the Student-vs-Welch choice, including the
  conditional version. Nothing here is about pooled variance.
- `effect-size-reader` (Day 178) measured `P(X > Y)` as an ESTIMATE and found it spread 0.0827
  across shapes at an identical Cohen's d. That spread is the mechanism this build exploits: if
  P(X>Y) can move while d is held fixed, it can also move ACROSS 0.5 while the mean difference
  stays positive. Here that is pushed to its conclusion and handed to the two TESTS.

What gets measured:

1. **Calibration.** The asymptotic relative efficiency of Mann-Whitney to the t-test under a
   location shift is `12 * sigma^2 * (integral of f^2)^2`. That integral is computed numerically
   for six standardised populations and checked against the three shapes where it has a closed
   form: normal gives exactly 3/pi, uniform exactly 1, centred exponential exactly 3. Nothing
   else is reported until those three land.
2. **The headline: sign disagreement.** A design whose true mean difference is `+0.80` and whose
   true `P(B > A)` is `0.3551`, both in CLOSED FORM - so the disagreement is not a simulation
   artefact, it is arithmetic. Then the rate at which the two tests are both significant and
   point opposite ways, as a function of n.
3. **Power where the comparison is legitimate.** Under a pure location shift the two hypotheses
   coincide, so power can be compared - but only in cells where both tests control Type I, the
   `comparable` gate carried forward from Day 176 and Day 178.
4. **Type I error under unequal spread.** `P(B > A) = 0.5` holds exactly for two symmetric
   populations with the same centre and different variances, so the Mann-Whitney null is TRUE -
   and the test still misses its nominal rate, because its variance formula assumes the two
   distributions are identical, not merely balanced. Welch's t-test is the comparator.
5. **Ties.** A rank test on a 5-point scale. The tie correction is not a rounding detail.

Note on duplication: `wilson()`, `mc_interval()` and `_standard()` also appear in sibling builds.
Deliberate - every build has to run standalone from a bare Colab link, so it cannot import a
neighbour. Duplicated ACROSS builds, exactly one copy WITHIN this one; the chart, the app and the
notebook read this module's numbers rather than recomputing them.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import integrate, stats

ALPHA = 0.05
Z99 = 2.5758293035489004
CHUNK = 2000

# A nominal 5% test is called broken only when its whole 99% Wilson interval clears this band.
# Carried from Day 175, where comparing a point estimate to a fixed band made the harness flag
# its own exactly-calibrated cell.
BAND_LO, BAND_HI = 0.045, 0.055

REFERENCE_DRAWS = 20_000_000


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


# ---------------------------------------------------------------------------
# Populations, standardised by their ANALYTIC moments
# ---------------------------------------------------------------------------

LOGNORMAL_S = 0.75
CONTAM_SCALE = 5.0
CONTAM_SHARE = 0.05


def _standard(name: str, size: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw from `name`, already standardised to population mean 0 and population SD 1.

    Analytic moments, never sample ones. Under a location shift of `d` the population Cohen's d
    is then exactly `d` for every shape, which is what makes the power rows in section 3 a
    comparison of TESTS rather than a comparison of effect sizes.
    """
    if name == "normal":
        return rng.standard_normal(size)
    if name == "t5":
        return rng.standard_t(5, size) / math.sqrt(5.0 / 3.0)
    if name == "uniform":
        return (rng.random(size) - 0.5) * math.sqrt(12.0)
    if name == "exponential":
        return rng.exponential(1.0, size) - 1.0
    if name == "lognormal":
        s = LOGNORMAL_S
        raw = np.exp(s * rng.standard_normal(size))
        mean = math.exp(s * s / 2.0)
        sd = math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
        return (raw - mean) / sd
    if name == "contaminated":
        pick = rng.random(size) < CONTAM_SHARE
        base = rng.standard_normal(size)
        sd = math.sqrt((1 - CONTAM_SHARE) + CONTAM_SHARE * CONTAM_SCALE ** 2)
        return np.where(pick, base * CONTAM_SCALE, base) / sd
    raise ValueError(f"unknown population: {name}")


SHAPES: Tuple[str, ...] = ("normal", "uniform", "t5", "contaminated", "lognormal", "exponential")
SYMMETRIC: Tuple[str, ...] = ("normal", "uniform", "t5", "contaminated")


def density(name: str) -> Tuple[Callable[[float], float], Tuple[float, float]]:
    """The standardised density of `name`, plus an integration range that contains its mass.

    Needed for the efficiency calibration: the asymptotic relative efficiency of Mann-Whitney to
    the t-test under a location shift is 12 * sigma^2 * (integral f^2)^2, and here sigma = 1 by
    construction, so the whole thing is that one integral.
    """
    if name == "normal":
        return (lambda x: float(stats.norm.pdf(x)), (-14.0, 14.0))
    if name == "uniform":
        h = math.sqrt(3.0)
        return (lambda x: (1.0 / (2.0 * h)) if abs(x) <= h else 0.0, (-h, h))
    if name == "t5":
        c = math.sqrt(5.0 / 3.0)
        return (lambda x: float(c * stats.t.pdf(c * x, 5)), (-90.0, 90.0))
    if name == "exponential":
        return (lambda x: math.exp(-(x + 1.0)) if x > -1.0 else 0.0, (-1.0, 90.0))
    if name == "lognormal":
        s = LOGNORMAL_S
        mean = math.exp(s * s / 2.0)
        sd = math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
        lo = -mean / sd
        return (lambda x: float(sd * stats.lognorm.pdf(x * sd + mean, s)) if x > lo else 0.0,
                (lo + 1e-12, 90.0))
    if name == "contaminated":
        sd = math.sqrt((1 - CONTAM_SHARE) + CONTAM_SHARE * CONTAM_SCALE ** 2)
        return (lambda x: float(sd * ((1 - CONTAM_SHARE) * stats.norm.pdf(x * sd)
                                      + CONTAM_SHARE * stats.norm.pdf(x * sd, scale=CONTAM_SCALE))),
                (-120.0, 120.0))
    raise ValueError(f"unknown population: {name}")


def integral_f_squared(name: str) -> float:
    """Numeric integral of f^2 over the standardised density."""
    f, (lo, hi) = density(name)
    value, _err = integrate.quad(lambda x: f(x) ** 2, lo, hi, limit=500)
    return float(value)


def density_mass(name: str) -> float:
    """Integral of f - must be 1. A shape whose density does not integrate to 1 would make its
    efficiency number meaningless, so this is checked before the efficiency is reported."""
    f, (lo, hi) = density(name)
    value, _err = integrate.quad(f, lo, hi, limit=500)
    return float(value)


def asymptotic_are(name: str) -> float:
    """ARE of Mann-Whitney to the t-test under a location shift: 12 * sigma^2 * (int f^2)^2.

    Above 1 means Mann-Whitney needs FEWER observations for the same power. Closed forms exist
    for three of the six shapes and `test_nonparam.py` checks them: normal 3/pi = 0.954929,
    uniform exactly 1, centred exponential exactly 3.
    """
    return 12.0 * integral_f_squared(name) ** 2


ARE_CLOSED_FORM: Dict[str, float] = {
    "normal": 3.0 / math.pi,
    "uniform": 1.0,
    "exponential": 3.0,
}


# ---------------------------------------------------------------------------
# The sign-disagreement design, entirely in closed form
# ---------------------------------------------------------------------------

# Control: N(0, MIX_SD^2). Treated: a two-component normal mixture with the same component SD.
# "Most treated cases get a little worse, a few get a lot better" - a shape that turns up in
# every treatment-effect setting anyone has ever worked in.
MIX_SD = 0.5
MIX_WEIGHTS: Tuple[float, ...] = (0.7, 0.3)
MIX_MEANS: Tuple[float, ...] = (-1.0, 5.0)


def mix_mean() -> float:
    """True mean of the treated population. Positive: the t-test's target says treated is HIGHER."""
    return float(sum(w * m for w, m in zip(MIX_WEIGHTS, MIX_MEANS)))


def mix_sd() -> float:
    """True SD of the treated population - large, because the mixture is what makes it large."""
    second = sum(w * (MIX_SD ** 2 + m * m) for w, m in zip(MIX_WEIGHTS, MIX_MEANS))
    return float(math.sqrt(second - mix_mean() ** 2))


def mix_prob_superiority() -> float:
    """True P(treated > control), in closed form.

    For normals, P(B > A) = Phi((mu_B - mu_A) / sqrt(sd_A^2 + sd_B^2)), and a mixture in B makes
    it the weighted sum of those. Below 0.5: Mann-Whitney's target says treated is LOWER.
    """
    spread = math.sqrt(2.0) * MIX_SD
    return float(sum(w * stats.norm.cdf(m / spread) for w, m in zip(MIX_WEIGHTS, MIX_MEANS)))


def mix_cohens_d() -> float:
    """Population Cohen's d for the disagreement design, pooled-SD convention."""
    pooled = math.sqrt((MIX_SD ** 2 + mix_sd() ** 2) / 2.0)
    return float(mix_mean() / pooled)


def draw_disagreement(n1: int, n2: int, reps: int,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Control and treated samples from the sign-disagreement design."""
    a = rng.standard_normal((reps, n1)) * MIX_SD
    which = rng.random((reps, n2)) < MIX_WEIGHTS[0]
    centres = np.where(which, MIX_MEANS[0], MIX_MEANS[1])
    b = centres + rng.standard_normal((reps, n2)) * MIX_SD
    return a, b


# ---------------------------------------------------------------------------
# The two tests, run on stacks of replicate samples
# ---------------------------------------------------------------------------


def welch(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Welch's two-sample t on each row. Returns (p two-sided, signed direction).

    Welch rather than Student's because Day 176 established unconditional Welch as the one
    procedure that held its nominal rate on every design in that grid - so when a Type I number
    misbehaves in this build, it is not the pooled-variance problem wearing a new hat.
    """
    res = stats.ttest_ind(b, a, axis=1, equal_var=False)
    return np.asarray(res.pvalue, dtype=float), np.sign(np.asarray(res.statistic, dtype=float))


def mannwhitney(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U on each row, asymptotic with SciPy's tie correction.

    Returns (p two-sided, signed direction), where the direction is the sign of
    `U / (n1 * n2) - 0.5`, i.e. whether the sample says P(B > A) is above or below a half. That
    statistic IS the sample probability of superiority - the same quantity Day 178 measured as an
    effect size. `method="asymptotic"` is pinned rather than left on "auto" so every cell in this
    build uses one procedure; the exact test is a different procedure with a different Type I
    profile and mixing them would make the grid unreadable.
    """
    res = stats.mannwhitneyu(b, a, axis=1, alternative="two-sided", method="asymptotic")
    u = np.asarray(res.statistic, dtype=float)
    phat = u / float(b.shape[1] * a.shape[1])
    return np.asarray(res.pvalue, dtype=float), np.sign(phat - 0.5)


def mannwhitney_uncorrected(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mann-Whitney's normal approximation WITHOUT the tie correction, two-sided p.

    Var(U) = n1 n2 (N + 1) / 12 is the no-ties formula. With ties the true variance is smaller,
    by n1 n2 / (12 N (N-1)) * sum(t^3 - t) over tie groups - so dropping the correction divides
    by too large a standard error and the test becomes conservative. That is the whole point of
    section 5 and it needs a hand-rolled version, because SciPy always applies the correction.
    """
    n1, n2 = a.shape[1], b.shape[1]
    reps = a.shape[0]
    n = n1 + n2
    both = np.concatenate([b, a], axis=1)
    ranks = np.apply_along_axis(stats.rankdata, 1, both)
    r_b = ranks[:, :n2].sum(axis=1)
    u = r_b - n2 * (n2 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = math.sqrt(n1 * n2 * (n + 1) / 12.0)
    z = (u - mu) / sd
    # Continuity correction matching SciPy's asymptotic branch, so the ONLY difference between
    # this and `mannwhitney` is the tie term.
    z = np.sign(z) * np.maximum(np.abs(z) - 0.5 / sd, 0.0)
    p = 2.0 * stats.norm.sf(np.abs(z))
    assert p.shape == (reps,)
    return np.minimum(p, 1.0)


def _chunks(reps: int, size: int = CHUNK) -> List[int]:
    out = []
    left = reps
    while left > 0:
        take = min(size, left)
        out.append(take)
        left -= take
    return out


# ---------------------------------------------------------------------------
# Section 2: the sign-disagreement study
# ---------------------------------------------------------------------------


def run_disagreement(n: int, reps: int, seed: int) -> Dict[str, float]:
    """Both tests on the disagreement design at n per group.

    Returns the rate at which each test is significant in each direction, plus the headline:
    `opposite` - both significant, pointing opposite ways.
    """
    rng = np.random.default_rng(seed)
    counts = {"t_up": 0, "t_down": 0, "mw_up": 0, "mw_down": 0,
              "opposite": 0, "both_sig": 0, "t_only": 0, "mw_only": 0, "neither": 0}
    for take in _chunks(reps):
        a, b = draw_disagreement(n, n, take, rng)
        pt, dt = welch(a, b)
        pm, dm = mannwhitney(a, b)
        t_sig, m_sig = pt < ALPHA, pm < ALPHA
        counts["t_up"] += int(np.sum(t_sig & (dt > 0)))
        counts["t_down"] += int(np.sum(t_sig & (dt < 0)))
        counts["mw_up"] += int(np.sum(m_sig & (dm > 0)))
        counts["mw_down"] += int(np.sum(m_sig & (dm < 0)))
        counts["both_sig"] += int(np.sum(t_sig & m_sig))
        counts["opposite"] += int(np.sum(t_sig & m_sig & (dt * dm < 0)))
        counts["t_only"] += int(np.sum(t_sig & ~m_sig))
        counts["mw_only"] += int(np.sum(~t_sig & m_sig))
        counts["neither"] += int(np.sum(~t_sig & ~m_sig))
    out = {k: v / reps for k, v in counts.items()}
    out["reps"] = float(reps)
    out["n"] = float(n)
    return out


# ---------------------------------------------------------------------------
# Section 3: power under a pure location shift, with the comparability gate
# ---------------------------------------------------------------------------


def run_location(shape: str, d: float, n: int, reps: int, seed: int) -> Dict[str, float]:
    """Rejection rates of both tests on `shape` with the treated group shifted by exactly `d`.

    At d = 0 these are Type I rates and the gate below reads them. At d > 0 they are power, and
    the two hypotheses coincide under a pure location shift, so the comparison is legitimate -
    provided the d = 0 cell said both tests control their error.
    """
    rng = np.random.default_rng(seed)
    t_rej = m_rej = 0
    for take in _chunks(reps):
        a = _standard(shape, (take, n), rng)
        b = _standard(shape, (take, n), rng) + d
        pt, _ = welch(a, b)
        pm, _ = mannwhitney(a, b)
        t_rej += int(np.sum(pt < ALPHA))
        m_rej += int(np.sum(pm < ALPHA))
    return {"shape_d": d, "n": float(n), "reps": float(reps),
            "t": t_rej / reps, "mw": m_rej / reps}


def comparable(t_null: float, mw_null: float, reps: int) -> bool:
    """Power is only a comparison where BOTH tests control Type I at this n.

    The gate that Day 176 had to add after publishing a power table in which an inflated test
    "detected more" - which is the inflation restated, not a finding.
    """
    return verdict(t_null, reps) == "ok" and verdict(mw_null, reps) == "ok"


def _monotone_search(pred: Callable[[int], bool], lo: int, hi: int) -> Optional[int]:
    """Smallest integer in [lo, hi] with `pred` true, asserting the predicate at BOTH ends.

    Carried from Day 178, where a search whose predicate was silently False everywhere (SciPy's
    noncentral t returning nan) walked past the answer and returned `hi` as if it were a result.
    Returns None rather than a bracket end when the bracket does not contain the crossing.
    """
    if pred(lo):
        return lo
    if not pred(hi):
        return None
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def efficiency_from_curve(n_grid: Sequence[int], t_power: Sequence[float],
                          n_ref: int, mw_power_ref: float) -> Optional[float]:
    """n_t / n_ref, where n_t is the sample size the t-test needs to match Mann-Whitney at n_ref.

    Read off the measured t-test power curve by linear interpolation in log n. Returns None when
    the target power falls outside the measured curve - an extrapolated efficiency is a guess,
    and this build does not report guesses. Above 1 means Mann-Whitney was the more efficient
    test here, which is the same direction as `asymptotic_are`.
    """
    xs = [math.log(n) for n in n_grid]
    ys = list(t_power)
    if mw_power_ref < min(ys) or mw_power_ref > max(ys):
        return None
    for i in range(len(xs) - 1):
        lo_y, hi_y = ys[i], ys[i + 1]
        if min(lo_y, hi_y) <= mw_power_ref <= max(lo_y, hi_y):
            if hi_y == lo_y:
                return math.exp(xs[i]) / n_ref
            frac = (mw_power_ref - lo_y) / (hi_y - lo_y)
            return math.exp(xs[i] + frac * (xs[i + 1] - xs[i])) / n_ref
    return None


# ---------------------------------------------------------------------------
# Section 4: Type I error under unequal spread, where P(B > A) = 0.5 exactly
# ---------------------------------------------------------------------------


def run_unequal_spread(shape: str, sd_ratio: float, n1: int, n2: int,
                       reps: int, seed: int) -> Dict[str, float]:
    """Both groups centred at 0, treated group's SD multiplied by `sd_ratio`.

    For a SYMMETRIC population this makes P(B > A) exactly 0.5, so the Mann-Whitney null in its
    stochastic-dominance form is exactly true - and the mean null is exactly true too. Any
    departure from 5% is the test's variance formula assuming the two distributions are
    identical rather than merely balanced.
    """
    rng = np.random.default_rng(seed)
    t_rej = m_rej = 0
    for take in _chunks(reps):
        a = _standard(shape, (take, n1), rng)
        b = _standard(shape, (take, n2), rng) * sd_ratio
        pt, _ = welch(a, b)
        pm, _ = mannwhitney(a, b)
        t_rej += int(np.sum(pt < ALPHA))
        m_rej += int(np.sum(pm < ALPHA))
    return {"sd_ratio": sd_ratio, "n1": float(n1), "n2": float(n2), "reps": float(reps),
            "welch": t_rej / reps, "mw": m_rej / reps}


# ---------------------------------------------------------------------------
# Section 5: ties
# ---------------------------------------------------------------------------


def to_levels(x: np.ndarray, levels: int) -> np.ndarray:
    """Bin standardised values onto a `levels`-point scale - a Likert answer, not a measurement.

    Equal-probability cut points from the normal quantiles, so a normal population lands roughly
    uniformly across the scale and the tie groups are large.
    """
    edges = stats.norm.ppf(np.linspace(0.0, 1.0, levels + 1)[1:-1])
    return np.searchsorted(edges, x).astype(float)


def tie_variance_ratio(rows: np.ndarray, levels: int) -> np.ndarray:
    """SD_corrected / SD_uncorrected for Mann-Whitney's normal approximation, per row.

    Var(U) with ties is `n1 n2 / 12 * ((N + 1) - sum(t^3 - t) / (N (N - 1)))`, against the
    no-ties `n1 n2 (N + 1) / 12`. Ties only ever SHRINK the variance, so this ratio is at most 1
    and the uncorrected test divides by a standard error that is too large. That is the whole
    mechanism of section 5, and reporting it makes the Type I numbers below predictable rather
    than surprising.
    """
    n = rows.shape[1]
    counts = np.stack([(rows == lv).sum(axis=1) for lv in range(levels)], axis=1).astype(float)
    tie_term = (counts ** 3 - counts).sum(axis=1)
    corrected = (n + 1) - tie_term / (n * (n - 1))
    return np.sqrt(corrected / (n + 1))


def run_ties(levels: int, n: int, d: float, reps: int, seed: int) -> Dict[str, float]:
    """Mann-Whitney on a `levels`-point scale, with and without the tie correction.

    At d = 0 these are Type I rates; at d > 0 they are power. `mannwhitney_uncorrected` is
    verified against SciPy on TIE-FREE data first (agreement to 3.5e-17), so the only thing
    separating the two columns here is the tie term.
    """
    rng = np.random.default_rng(seed)
    corrected = uncorrected = 0
    ratio_total = 0.0
    for take in _chunks(reps):
        a = to_levels(rng.standard_normal((take, n)), levels)
        b = to_levels(rng.standard_normal((take, n)) + d, levels)
        pc, _ = mannwhitney(a, b)
        pu = mannwhitney_uncorrected(a, b)
        corrected += int(np.sum(pc < ALPHA))
        uncorrected += int(np.sum(pu < ALPHA))
        both = np.concatenate([b, a], axis=1)
        ratio_total += float(tie_variance_ratio(both, levels).sum())
    return {"levels": float(levels), "n": float(n), "d": d, "reps": float(reps),
            "corrected": corrected / reps, "uncorrected": uncorrected / reps,
            "sd_ratio": ratio_total / reps}


# ---------------------------------------------------------------------------
# Study design lists. Seeds are INDICES into these lists, so the notebook and the app reproduce
# the evidence file's cells rather than drawing a second sample of them. Day 178 shipped seeds
# derived from `hash((shape, d))` and Python's per-process string-hash salt made every run
# different; a test asserts these lists keep their ORDER.
# ---------------------------------------------------------------------------

STUDY_REPS = 40_000
NULL_REPS = 100_000
DISAGREE_REPS = 40_000
TIE_REPS = 40_000

DISAGREE_NS: Tuple[int, ...] = (10, 20, 50, 100, 200, 500)

LOCATION_D = 0.35
LOCATION_NS: Tuple[int, ...] = (20, 30, 45, 65, 95, 140, 200, 300)
LOCATION_REF_N = 65

UNEQUAL_RATIOS: Tuple[float, ...] = (1.0, 2.0, 4.0)
UNEQUAL_NS: Tuple[Tuple[int, int], ...] = ((30, 30), (10, 50), (50, 10))

TIE_LEVELS: Tuple[int, ...] = (2, 3, 5, 7)
TIE_N = 40
TIE_D = 0.5

DISAGREE_SEED_BASE = 11_000
LOCATION_SEED_BASE = 21_000
NULL_SEED_BASE = 31_000
UNEQUAL_SEED_BASE = 41_000
TIE_NULL_SEED_BASE = 51_000
TIE_POWER_SEED_BASE = 61_000

LOCATION_DESIGNS: Tuple[Tuple[str, int], ...] = tuple(
    (shape, n) for shape in SHAPES for n in LOCATION_NS
)
UNEQUAL_DESIGNS: Tuple[Tuple[str, float, int, int], ...] = tuple(
    (shape, ratio, n1, n2)
    for shape in SYMMETRIC for ratio in UNEQUAL_RATIOS for (n1, n2) in UNEQUAL_NS
)
