"""Cohen's d is not the effect. It is one summary of the effect, and the summaries disagree.

The four sibling builds in this domain (`t-test-variants`, `normality-test-trap`,
`assumption-pretest-cost`, `p-value-dance`) all interrogate the TEST - whether it rejects when it
should not, and what its p-value does. This one leaves the test entirely. The question here is
what happens after somebody accepts there is an effect and asks how big it is.

The design that makes this measurable, and it is the whole trick: every population is standardised
to population mean 0 and population SD 1 using its ANALYTIC moments, never sample ones. Group B is
then shifted by exactly `d`. So the population Cohen's d is exactly `d` for every distribution
shape - normal, skewed, heavy-tailed, contaminated, all of them.

That holds d fixed by construction, which means any disagreement between the other effect-size
metrics across shapes cannot be a difference in "how big the effect is". It is the metrics
answering different questions.

What gets measured:

1. **Truth, per shape.** At a fixed true d, what is the true probability of superiority - the
   chance a randomly drawn treated value beats a randomly drawn control value? For normal data
   this is exactly Phi(d/sqrt(2)). For every other shape it is not, and this build computes it.
2. **What the estimators do.** Bias and spread of d-hat, Hedges' g, Glass's delta and Cliff's
   delta at realistic n.
3. **Robustness.** What one contaminated observation in a hundred does to each metric.
4. **Significance versus magnitude.** The exact n at which a difference nobody would act on
   crosses p < 0.05, and what the smallest significant effect at n = 100,000 actually means.
5. **Dichotomisation.** What a median split costs, in power and in the odds ratio it produces.

Note on duplication: `wilson()` also appears in three sibling builds. Deliberate - each build must
run standalone from a bare Colab link, so it cannot import a neighbour. Duplicated ACROSS builds,
exactly one copy WITHIN this one; the chart, the app and the notebook read this module's numbers
rather than recomputing them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
Z99 = 2.5758293035489004
CHUNK = 2000

# Reference draws for a population truth. 20M pairs puts the Monte Carlo SE on a probability at
# about 1e-4, which is two orders below any difference this build calls a finding.
REFERENCE_DRAWS = 20_000_000


def wilson(rate: float, n: int, z: float = Z99) -> Tuple[float, float]:
    """Wilson score interval for a measured proportion - used for every rate comparison here."""
    if n <= 0:
        return (0.0, 1.0)
    denom = 1.0 + z * z / n
    centre = (rate + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mc_interval(values: np.ndarray, z: float = Z99) -> Tuple[float, float]:
    """99% interval on a Monte Carlo mean - so a 'difference' is never read off two point
    estimates whose error bars overlap."""
    m = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return m - z * se, m + z * se


# ---------------------------------------------------------------------------
# Populations, each standardised by its ANALYTIC moments
# ---------------------------------------------------------------------------

LOGNORMAL_S = 0.75
CONTAM_SCALE = 5.0
CONTAM_SHARE = 0.05


def _standard(name: str, size: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw from `name` already standardised to population mean 0, population SD 1.

    Standardising by the analytic moments rather than the sample's is the load-bearing detail.
    Dividing by a sample SD would make the true d only correct on average, and every claim below
    is about a d that is correct exactly.
    """
    if name == "normal":
        return rng.standard_normal(size)
    if name == "t5":
        # Var(t_v) = v/(v-2) = 5/3.
        return rng.standard_t(5, size) / math.sqrt(5.0 / 3.0)
    if name == "uniform":
        return (rng.random(size) - 0.5) * math.sqrt(12.0)
    if name == "exponential":
        # Exp(1) has mean 1 and SD 1, so centring is all that is needed. Skew 2.
        return rng.exponential(1.0, size) - 1.0
    if name == "lognormal":
        s = LOGNORMAL_S
        raw = np.exp(s * rng.standard_normal(size))
        mean = math.exp(s * s / 2.0)
        sd = math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
        return (raw - mean) / sd
    if name == "contaminated":
        # 95% N(0,1) + 5% N(0,5): symmetric, unimodal, and a variance dominated by the tail.
        pick = rng.random(size) < CONTAM_SHARE
        base = rng.standard_normal(size)
        sd = math.sqrt((1 - CONTAM_SHARE) + CONTAM_SHARE * CONTAM_SCALE ** 2)
        return np.where(pick, base * CONTAM_SCALE, base) / sd
    raise ValueError(f"unknown population: {name}")


SHAPES: Tuple[str, ...] = ("normal", "uniform", "t5", "contaminated", "lognormal", "exponential")
SKEWED: Tuple[str, ...] = ("lognormal", "exponential")


def draw_pair(
    shape: str, d: float, n1: int, n2: int, reps: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Control and treated samples. Population Cohen's d is EXACTLY `d`, whatever the shape."""
    a = _standard(shape, (reps, n1), rng)
    b = _standard(shape, (reps, n2), rng) + d
    return a, b


# ---------------------------------------------------------------------------
# Population truth
# ---------------------------------------------------------------------------


def true_prob_superiority(shape: str, d: float, draws: int = REFERENCE_DRAWS,
                          seed: int = 0) -> Tuple[float, float]:
    """P(treated > control) for the POPULATIONS, by reference sampling. Returns (value, 99% CI).

    For normal data this has a closed form, Phi(d/sqrt(2)), which `test_effectsize.py` checks this
    routine against - the calibration cell. For every other shape there is no closed form, which
    is exactly why the number is worth computing rather than assuming.
    """
    rng = np.random.default_rng(1000 + seed)
    hits = 0
    total = 0
    block = 2_000_000
    wins: List[float] = []
    while total < draws:
        take = min(block, draws - total)
        a = _standard(shape, (take,), rng)
        b = _standard(shape, (take,), rng) + d
        w = float((b > a).mean())
        wins.append(w)
        hits += int((b > a).sum())
        total += take
    p = hits / total
    lo, hi = wilson(p, total)
    return p, (hi - lo) / 2.0


def analytic_prob_superiority(d: float) -> float:
    """P(X > Y) for two normals separated by d standard deviations. Normal data ONLY."""
    return float(stats.norm.cdf(d / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# The metrics, all computed on the same samples
# ---------------------------------------------------------------------------

METRICS: Tuple[str, ...] = (
    "cohens_d", "hedges_g", "glass_delta", "prob_superiority", "cliffs_delta",
    "point_biserial_r", "log_odds_median_split",
)


def metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:
    """Every effect-size summary in this build, on one pair of samples (vectorised over rows)."""
    n1, n2 = a.shape[-1], b.shape[-1]
    ma, mb = a.mean(axis=-1), b.mean(axis=-1)
    va, vb = a.var(axis=-1, ddof=1), b.var(axis=-1, ddof=1)
    pooled = np.sqrt(((n1 - 1) * va + (n2 - 1) * vb) / (n1 + n2 - 2))
    d = (mb - ma) / pooled
    # Hedges' small-sample correction. Its whole claim is that d is biased upward at small n.
    j = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    # Glass's delta divides by the CONTROL SD only - the standard advice when the treatment is
    # expected to change the variance too.
    glass = (mb - ma) / np.sqrt(va)

    # P(treated > control), via Mann-Whitney U. Midrank ties are counted as half, which is what
    # makes this the "common language" effect size rather than a strict inequality.
    u = stats.mannwhitneyu(b, a, axis=-1, method="asymptotic").statistic
    ps = np.asarray(u, dtype=float) / (n1 * n2)

    # Point-biserial r, from the t statistic, so it is exactly the correlation between the value
    # and the group indicator.
    t = stats.ttest_ind(b, a, axis=-1, equal_var=True).statistic
    t = np.asarray(t, dtype=float)
    df = n1 + n2 - 2
    r = t / np.sqrt(t * t + df)

    # The median split: dichotomise at the POOLED median, then a 2x2 odds ratio with the Haldane
    # correction so an empty cell does not produce an infinity.
    both = np.concatenate([a, b], axis=-1)
    cut = np.median(both, axis=-1, keepdims=True)
    a_hi = (a > cut).sum(axis=-1)
    b_hi = (b > cut).sum(axis=-1)
    odds = (((b_hi + 0.5) / (n2 - b_hi + 0.5)) / ((a_hi + 0.5) / (n1 - a_hi + 0.5)))

    return {
        "cohens_d": d, "hedges_g": d * j, "glass_delta": glass,
        "prob_superiority": ps, "cliffs_delta": 2.0 * ps - 1.0,
        "point_biserial_r": r, "log_odds_median_split": np.log(odds),
    }


# ---------------------------------------------------------------------------
# One cell of the study
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    shape: str
    d_true: float
    n: int
    reps: int
    truth_ps: float
    truth_ps_err: float
    means: Dict[str, float]
    cis: Dict[str, Tuple[float, float]]
    sds: Dict[str, float]
    d_bias: float
    g_bias: float
    ps_bias: float
    power: float
    seed: int

    @property
    def label(self) -> str:
        return f"{self.shape}, d={self.d_true:g}, n={self.n}"


def analyse(shape: str, d: float, n: int, reps: int, seed: int,
            truth: Optional[Tuple[float, float]] = None) -> Cell:
    rng = np.random.default_rng(seed)
    acc: Dict[str, List[np.ndarray]] = {m: [] for m in METRICS}
    sig: List[np.ndarray] = []
    done = 0
    while done < reps:
        take = min(CHUNK, reps - done)
        a, b = draw_pair(shape, d, n, n, take, rng)
        for k, v in metrics(a, b).items():
            acc[k].append(v)
        sig.append(np.asarray(stats.ttest_ind(b, a, axis=-1, equal_var=True).pvalue) < ALPHA)
        done += take
    vals = {k: np.concatenate(v) for k, v in acc.items()}
    ps_true, ps_err = truth if truth is not None else true_prob_superiority(shape, d)
    return Cell(
        shape=shape, d_true=d, n=n, reps=reps, truth_ps=ps_true, truth_ps_err=ps_err,
        means={k: float(v.mean()) for k, v in vals.items()},
        cis={k: mc_interval(v) for k, v in vals.items()},
        sds={k: float(v.std(ddof=1)) for k, v in vals.items()},
        d_bias=float(vals["cohens_d"].mean()) - d,
        g_bias=float(vals["hedges_g"].mean()) - d,
        ps_bias=float(vals["prob_superiority"].mean()) - ps_true,
        power=float(np.concatenate(sig).mean()),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Robustness: what one contaminated observation in a hundred does
# ---------------------------------------------------------------------------


@dataclass
class Robustness:
    metric: str
    clean: float
    dirty: float
    pct_change: float
    direction: str


def contaminate(shape: str, d: float, n: int, reps: int, seed: int,
                share: float = 0.01, magnitude: float = 10.0) -> List[Robustness]:
    """Replace `share` of the TREATED values with an outlier `magnitude` SDs out, then remeasure.

    The outlier is placed in the direction of the effect, which is the charitable case: it is a
    real-looking big response, not a data-entry error. Anything that moves a lot here moves more
    on a genuinely bad value.
    """
    rng = np.random.default_rng(seed)
    a, b = draw_pair(shape, d, n, n, reps, rng)
    clean = metrics(a, b)
    dirty_b = b.copy()
    k = max(1, int(round(share * n)))
    dirty_b[:, :k] = magnitude
    dirty = metrics(a, dirty_b)
    out: List[Robustness] = []
    for m in METRICS:
        c, dd = float(clean[m].mean()), float(dirty[m].mean())
        pct = (dd - c) / abs(c) * 100.0 if c != 0 else float("nan")
        out.append(Robustness(m, c, dd, pct, "up" if dd > c else "down"))
    return out


# ---------------------------------------------------------------------------
# Significance versus magnitude
# ---------------------------------------------------------------------------


def nct_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Exact two-sided power from the noncentral t. Returns nan where SciPy cannot evaluate it."""
    df = 2 * n - 2
    nc = d * math.sqrt(n / 2.0)
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc))


def normal_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Large-sample approximation: the noncentral t tends to a shifted normal as df grows."""
    nc = d * math.sqrt(n / 2.0)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(stats.norm.sf(z - nc) + stats.norm.cdf(-z - nc))


def analytic_power(d: float, n: int, alpha: float = ALPHA) -> float:
    """Two-sided power of Student's two-sample t at effect d, n per group.

    Exact via the noncentral t wherever SciPy can evaluate it, and the normal approximation where
    it cannot. That fallback exists because of a defect this build hit and nearly shipped:

        scipy.stats.nct (1.17.1 here) returns **nan** - not an error, not a warning - over a band
        of large degrees of freedom. n = 1,000 and n = 5,000 both produce nan, while n = 500 and
        n = 20,000 are fine.

    A nan is not obviously wrong. It is silently False in every comparison, so `power >= target`
    reads as "not powerful enough" and a monotone binary search walks straight past the answer and
    returns a larger, entirely plausible number. The first version of this module reported that
    d = 0.5 needs n = 11,417 per group for 80% power. The real answer is 64.

    The fallback is only trusted because it is measured: `power_disagreement()` reports the worst
    gap between the two formulas everywhere both are finite, and the tests check the hybrid
    against simulation.
    """
    exact = nct_power(d, n, alpha)
    if math.isfinite(exact):
        return exact
    return normal_power(d, n, alpha)


def power_disagreement(d: float, ns: Sequence[int], alpha: float = ALPHA) -> Dict[str, float]:
    """Largest gap between the exact and approximate formulas where BOTH are finite.

    The fallback above is a claim that the approximation is good in the region where the exact
    form fails. This is how that claim gets a number instead of a shrug.
    """
    gaps: List[Tuple[int, float]] = []
    nan_ns: List[int] = []
    for n in ns:
        e = nct_power(d, n, alpha)
        if not math.isfinite(e):
            nan_ns.append(n)
            continue
        gaps.append((n, abs(e - normal_power(d, n, alpha))))
    worst_n, worst = max(gaps, key=lambda kv: kv[1]) if gaps else (0, float("nan"))
    big = [(n, g) for n, g in gaps if 2 * n - 2 >= 400]
    worst_big_n, worst_big = max(big, key=lambda kv: kv[1]) if big else (0, float("nan"))
    return {
        "worst_gap": worst, "worst_gap_n": worst_n,
        "worst_gap_large_df": worst_big, "worst_gap_large_df_n": worst_big_n,
        "n_nan": len(nan_ns), "first_nan_n": nan_ns[0] if nan_ns else -1,
        "last_nan_n": nan_ns[-1] if nan_ns else -1,
    }


def _monotone_search(ok: Callable[[int], bool], lo: int, hi: int) -> int:
    """Binary search with the monotonicity it assumes CHECKED at both ends.

    An unchecked binary search over a predicate that is silently False everywhere (the nan case)
    returns `hi` - a number, in range, and wrong. Both ends are asserted so it raises instead.
    """
    if ok(lo):
        return lo
    if not ok(hi):
        raise ValueError(f"predicate is False at the top of the range ({hi}) - no answer here")
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def n_for_significance(d: float, target: float = 0.80, alpha: float = ALPHA) -> int:
    """Smallest n per group at which an effect of size d is detected `target` of the time."""
    if d <= 0:
        return -1
    return _monotone_search(lambda n: analytic_power(d, n, alpha) >= target, 4, 200_000_000)


def smallest_significant_d(n: int, alpha: float = ALPHA) -> float:
    """The smallest true d that a study of this size detects half the time.

    'Half the time' rather than 'at all' on purpose: any d > 0 is detected sometimes, so there is
    no smallest detectable effect - only a smallest effect the study is not guessing about.
    """
    lo, hi = 0.0, 5.0
    if analytic_power(hi, n, alpha) < 0.5:
        raise ValueError(f"n={n} cannot reach 50% power even at d=5")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if analytic_power(mid, n, alpha) >= 0.5:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Predicates the findings are stated as - each must be able to return either way
# ---------------------------------------------------------------------------


def truths_disagree(ps_by_shape: Dict[str, float], tol: float = 0.01) -> bool:
    """True when the population P(X>Y) varies by more than `tol` across shapes.

    This is the real subject of the headline claim, and it is stated on the POPULATION truths -
    the estimator's noise has nothing to do with it. `metrics_disagree` below is the same test
    reached from a list of Cells; there is one implementation, not two.
    """
    vals = list(ps_by_shape.values())
    return bool(vals and (max(vals) - min(vals)) > tol)


def metrics_disagree(cells: Sequence[Cell], metric: str = "prob_superiority",
                     tol: float = 0.01) -> bool:
    """True when, at a FIXED true Cohen's d, `metric`'s population truth varies by more than
    `tol` across distribution shapes.

    Since d is identical by construction, any spread here is the metrics answering different
    questions - not a difference in how big the effect is.
    """
    ds = {c.d_true for c in cells}
    if len(ds) != 1:
        raise ValueError("this predicate compares shapes at ONE fixed d")
    return truths_disagree({c.shape: c.truth_ps for c in cells}, tol)


def correction_matters(cell: Cell, tol: float = 0.01) -> bool:
    """True when Hedges' correction changes the answer by more than `tol` in absolute effect."""
    return abs(cell.means["cohens_d"] - cell.means["hedges_g"]) > tol


# ---------------------------------------------------------------------------
# What a median split costs
# ---------------------------------------------------------------------------


@dataclass
class Dichotomisation:
    shape: str
    d_true: float
    n: int
    reps: int
    power_t: float
    power_chi2: float
    power_t_ci: Tuple[float, float]
    power_chi2_ci: Tuple[float, float]
    equivalent_n: int
    n_wasted_share: float
    mean_log_odds: float
    null_t: float
    null_chi2: float
    null_t_ok: bool
    null_chi2_ok: bool
    comparable: bool


BAND_LO, BAND_HI = 0.045, 0.055


def _two_test_rates(shape: str, d: float, n: int, reps: int,
                    rng: np.random.Generator) -> Tuple[float, float, np.ndarray]:
    """Rejection rates of the t-test and the median-split test on the SAME draws."""
    sig_t: List[np.ndarray] = []
    sig_c: List[np.ndarray] = []
    odds: List[np.ndarray] = []
    done = 0
    while done < reps:
        take = min(CHUNK, reps - done)
        a, b = draw_pair(shape, d, n, n, take, rng)
        sig_t.append(np.asarray(stats.ttest_ind(b, a, axis=-1, equal_var=True).pvalue) < ALPHA)
        both = np.concatenate([a, b], axis=-1)
        cut = np.median(both, axis=-1, keepdims=True)
        a_hi = (a > cut).sum(axis=-1)
        b_hi = (b > cut).sum(axis=-1)
        # Two-proportion z on the median split, which is the chi-square without the machinery.
        p1, p2 = b_hi / n, a_hi / n
        pool = (b_hi + a_hi) / (2 * n)
        se = np.sqrt(np.maximum(pool * (1 - pool) * 2.0 / n, 1e-300))
        sig_c.append(np.abs((p1 - p2) / se) > stats.norm.ppf(1 - ALPHA / 2))
        odds.append(metrics(a, b)["log_odds_median_split"])
        done += take
    return (float(np.concatenate(sig_t).mean()), float(np.concatenate(sig_c).mean()),
            np.concatenate(odds))


def dichotomisation_cost(shape: str, d: float, n: int, reps: int, seed: int) -> Dichotomisation:
    """Run the t-test on the numbers, and a chi-square on the same data cut at the pooled median.

    The cost is reported as an EQUIVALENT SAMPLE SIZE - the n at which the t-test would have had
    the chi-square's power - because "you lose 8 points of power" means nothing to anyone and
    "you threw away a third of your participants" means something to everybody.

    A NULL cell is run first, at d = 0 on the same shape and n, and the power comparison is only
    marked `comparable` when BOTH tests hold a nominal 5%. A test that over-rejects detects more
    of everything, so its power lead is the inflation restated - the error that
    `assumption-pretest-cost` shipped and had to fix. On skewed data the t-test does not hold
    nominal at small n (see the sibling `normality-test-trap`), so this guard is not decoration:
    it disqualifies real rows in this very study.
    """
    rng = np.random.default_rng(seed)
    pt, pc, odds = _two_test_rates(shape, d, n, reps, rng)
    null_t, null_c, _ = _two_test_rates(shape, 0.0, n, reps, np.random.default_rng(seed + 50_000))
    t_lo, t_hi = wilson(null_t, reps)
    c_lo, c_hi = wilson(null_c, reps)
    # In band only when the WHOLE 99% interval sits inside it - never the point estimate.
    t_ok = not (t_hi < BAND_LO or t_lo > BAND_HI)
    c_ok = not (c_hi < BAND_LO or c_lo > BAND_HI)
    # The n at which the t-test alone would have scored the median-split test's power.
    try:
        eq = _monotone_search(lambda m: analytic_power(d, m) >= pc, 4, 200_000_000)
    except ValueError:
        eq = -1
    return Dichotomisation(
        shape=shape, d_true=d, n=n, reps=reps, power_t=pt, power_chi2=pc,
        power_t_ci=wilson(pt, reps), power_chi2_ci=wilson(pc, reps),
        equivalent_n=eq, n_wasted_share=(1.0 - eq / n) if eq > 0 else float("nan"),
        mean_log_odds=float(odds.mean()),
        null_t=null_t, null_chi2=null_c, null_t_ok=t_ok, null_chi2_ok=c_ok,
        comparable=bool(t_ok and c_ok),
    )


def dichotomisation_is_costly(dd: Dichotomisation) -> bool:
    """True only when the row is COMPARABLE and the t-test's power interval clears the split's.

    Two guards, both learned the hard way by sibling builds. The materiality rule (99% Wilson
    intervals that do not overlap) stops coin-flips reading as findings. The `comparable` gate
    stops a power comparison between tests that do not BOTH control Type I - where the "winner"
    may simply be the test that rejects more of everything. On skewed data the t-test does not
    hold nominal at small n, so this gate disqualifies real rows in this very study rather than
    decorating it.
    """
    if not dd.comparable:
        return False
    return dd.power_t_ci[0] > dd.power_chi2_ci[1]


def split_beats_t(dd: Dichotomisation) -> bool:
    """The opposite verdict, which must also be reachable or the study only has one answer."""
    if not dd.comparable:
        return False
    return dd.power_chi2_ci[0] > dd.power_t_ci[1]


def correction_fixes_bias(cell: Cell, tol: float = 0.01) -> bool:
    """True when Hedges' g lands within `tol` of the truth - i.e. the correction actually worked.

    Separate from `correction_matters`, and the distinction is the point: a correction can change
    the number (matters) without removing the bias (fixes). Hedges' J is derived under normality,
    so this predicate is expected to split by SHAPE rather than by n, and it does.
    """
    return abs(cell.g_bias) <= tol


# ---------------------------------------------------------------------------
# The study's designs. The SEED IS THE INDEX into these lists, so their order is load-bearing -
# it is asserted in the tests. They live here rather than in evidence.py so that the notebook and
# the evidence file cannot be running two different studies: one definition, two readers.
# ---------------------------------------------------------------------------

STUDY_REPS = 20000
STUDY_REFERENCE = 20_000_000
STUDY_DS: Tuple[float, ...] = (0.2, 0.5, 0.8)
STUDY_NS: Tuple[int, ...] = (10, 20, 50, 200)
DICH_D = 0.4
DICH_NS: Tuple[int, ...] = (50, 200)

TRUTH_DESIGNS: Tuple[Tuple[str, float], ...] = tuple(
    (shape, d) for d in STUDY_DS for shape in SHAPES)
CELL_DESIGNS: Tuple[Tuple[str, int], ...] = tuple(
    (shape, n) for shape in SHAPES for n in STUDY_NS)
DICH_DESIGNS: Tuple[Tuple[str, int], ...] = tuple(
    (shape, n) for shape in SHAPES for n in DICH_NS)
DICH_SEED_BASE = 200
