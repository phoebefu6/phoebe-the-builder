"""Sequential testing: what a stopping rule does to a p-value.

Nothing here is a wrapper around a library that already knows the answer. The
group-sequential boundaries are solved from the Armitage-McPherson recursion on
the joint density of the accumulating test statistic, so they can be checked
against the published 1977/1979 tables; the false-positive rates are measured on
simulated Bernoulli traffic rather than read off the same theory that produced
the boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm

# --------------------------------------------------------------------------
# 0. The one place the experiment is described
# --------------------------------------------------------------------------
# evidence.py, make_chart.py, app.py and the notebook all read these, so there
# is one description of the world rather than four that can drift apart.

ALPHA = 0.05
P0 = 0.10  # control conversion rate
LIFT_REL = 0.10  # the relative lift the experiment is powered for
P1 = P0 * (1 + LIFT_REL)
N_MAX = 20_000  # visitors per arm at the planned end of the experiment
K_DAILY = 20  # a look per day for 20 days


def equal_looks(k: int, n_max: int = N_MAX) -> np.ndarray:
    """Cumulative per-arm sample sizes for k equally spaced analyses."""
    return np.linspace(n_max / k, n_max, k).astype(np.int64)


# --------------------------------------------------------------------------
# 1. Group-sequential boundaries, solved rather than looked up
# --------------------------------------------------------------------------
#
# With K equally spaced analyses and independent increments, the partial sum
# S_k = sum_{j<=k} X_j has X_j ~ N(0,1) under the null, and the standardised
# statistic is Z_k = S_k / sqrt(k). A boundary b_k on the Z scale is a boundary
# b_k*sqrt(k) on the S scale. The probability of ever crossing is accumulated by
# carrying the sub-density of "still continuing" forward one convolution at a
# time -- the recursion in Armitage, McPherson & Rowe (1969).


def _tail_mass(grid: np.ndarray, dens: np.ndarray, thresh: float) -> float:
    """Integrate a sub-density over |s| >= thresh on a uniform grid."""
    step = grid[1] - grid[0]
    right = grid >= thresh
    left = grid <= -thresh
    total = 0.0
    if right.sum() > 1:
        total += float(np.trapz(dens[right], dx=step))
    if left.sum() > 1:
        total += float(np.trapz(dens[left], dx=step))
    return total


def crossing_probability(
    bounds: Sequence[float], step: float = 0.01, tail: float = 8.0
) -> Tuple[float, List[float]]:
    """P(|Z_k| >= bounds[k] for some k) under the null, plus the per-look exits.

    `bounds` is on the Z scale, one entry per equally spaced analysis.
    """
    K = len(bounds)
    lim = tail * math.sqrt(K) + 1.0
    n_half = int(math.ceil(lim / step))
    grid = np.arange(-n_half, n_half + 1) * step

    # Convolution kernel: the density of one unit-variance increment.
    k_half = int(math.ceil(tail / step))
    kern_grid = np.arange(-k_half, k_half + 1) * step
    kern = norm.pdf(kern_grid)

    dens = norm.pdf(grid)  # sub-density of S_1, nothing excluded yet
    exits: List[float] = []
    for k, b in enumerate(bounds, start=1):
        thresh = b * math.sqrt(k)
        exits.append(_tail_mass(grid, dens, thresh))
        if k == K:
            break
        dens = np.where(np.abs(grid) < thresh, dens, 0.0)
        dens = fftconvolve(dens, kern, mode="same") * step
    return float(sum(exits)), exits


def _solve_constant(shape: np.ndarray, alpha: float, step: float) -> float:
    """Find c such that bounds = c * shape spends exactly `alpha`."""
    lo, hi = 1.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        spent, _ = crossing_probability(mid * shape, step=step)
        if spent > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7:
            break
    return 0.5 * (lo + hi)


def pocock_bounds(K: int, alpha: float = 0.05, step: float = 0.01) -> np.ndarray:
    """Constant boundary on the Z scale (Pocock 1977)."""
    shape = np.ones(K)
    return _solve_constant(shape, alpha, step) * shape


def obf_bounds(K: int, alpha: float = 0.05, step: float = 0.01) -> np.ndarray:
    """b_k proportional to 1/sqrt(k) -- O'Brien & Fleming (1979)."""
    k = np.arange(1, K + 1)
    shape = np.sqrt(K / k)
    return _solve_constant(shape, alpha, step) * shape


def bonferroni_bounds(K: int, alpha: float = 0.05) -> np.ndarray:
    """The fix people reach for first: split alpha evenly and ignore the
    correlation between looks."""
    return np.full(K, float(norm.ppf(1.0 - alpha / (2 * K))))


def naive_bounds(K: int, alpha: float = 0.05) -> np.ndarray:
    """No correction at all: the fixed-horizon critical value at every look."""
    return np.full(K, float(norm.ppf(1.0 - alpha / 2)))


# --------------------------------------------------------------------------
# 2. A two-arm world, simulated at the look boundaries
# --------------------------------------------------------------------------


@dataclass
class Trial:
    """One simulated experiment set, evaluated at K analyses.

    Arrays are (n_sims, K). `z` is the pooled two-proportion z statistic,
    `diff` the observed absolute lift, `se` its unpooled standard error.
    """

    looks: np.ndarray  # cumulative per-arm sample size at each analysis
    z: np.ndarray
    diff: np.ndarray
    se: np.ndarray
    p0: float
    p1: float

    @property
    def true_diff(self) -> float:
        return self.p1 - self.p0


def simulate(
    looks: Sequence[int],
    p0: float,
    p1: float,
    n_sims: int,
    seed: int,
) -> Trial:
    """Balanced Bernoulli traffic, accumulated to each look.

    Increments between looks are independent binomials, which is exactly what
    arriving visitors are -- so a 400-look monitoring scheme costs no more to
    simulate than a 5-look one.
    """
    looks = np.asarray(looks, dtype=np.int64)
    rng = np.random.default_rng(seed)
    inc = np.diff(np.concatenate([[0], looks]))
    c0 = np.cumsum(rng.binomial(inc, p0, size=(n_sims, len(inc))), axis=1)
    c1 = np.cumsum(rng.binomial(inc, p1, size=(n_sims, len(inc))), axis=1)
    n = looks.astype(float)[None, :]
    ph0, ph1 = c0 / n, c1 / n
    pooled = (c0 + c1) / (2 * n)
    var_pool = pooled * (1 - pooled) * (2.0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var_pool > 0, (ph1 - ph0) / np.sqrt(var_pool), 0.0)
    se = np.sqrt(np.maximum(ph0 * (1 - ph0) + ph1 * (1 - ph1), 1e-12) / n)
    return Trial(looks=looks, z=np.nan_to_num(z), diff=ph1 - ph0, se=se, p0=p0, p1=p1)


# --------------------------------------------------------------------------
# 3. Stopping rules
# --------------------------------------------------------------------------


def first_crossing(stat: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Index of the first look whose statistic crosses, or -1 for never."""
    hit = np.abs(stat) >= bounds[None, :]
    any_hit = hit.any(axis=1)
    idx = np.where(any_hit, hit.argmax(axis=1), -1)
    return idx


def msprt_statistic(trial: Trial, tau: float) -> np.ndarray:
    """Mixture sequential probability ratio test (Robbins 1970; Johari et al.).

    Mixing the alternative over N(0, tau^2) gives a likelihood ratio that is a
    martingale under the null, so P(sup_n Lambda_n >= 1/alpha) <= alpha at ANY
    stopping time -- including one chosen by looking at the data.
    """
    v = trial.se**2
    return np.sqrt(v / (v + tau**2)) * np.exp(tau**2 * trial.diff**2 / (2 * v * (v + tau**2)))


def msprt_crossing(trial: Trial, tau: float, alpha: float = 0.05) -> np.ndarray:
    lam = msprt_statistic(trial, tau)
    hit = lam >= 1.0 / alpha
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1), -1)


# --------------------------------------------------------------------------
# 4. Scoring a rule: rejection rate, sample size, and what it reports
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    name: str
    reject_rate: float
    expected_n: float
    median_n: float
    est_at_stop: float  # mean observed lift among rejections
    est_bias: float  # relative to the true lift, in relative terms
    ci_coverage: float  # naive 95% interval at the stopping look
    ci_coverage_rejected: float


def score(trial: Trial, idx: np.ndarray, name: str) -> Outcome:
    """Everything a stopping rule should have to report about itself."""
    K = trial.z.shape[1]
    stop = np.where(idx >= 0, idx, K - 1)  # never crossed -> ran to the end
    rows = np.arange(trial.z.shape[0])
    n_at_stop = trial.looks[stop].astype(float)
    diff_at_stop = trial.diff[rows, stop]
    se_at_stop = trial.se[rows, stop]
    rejected = idx >= 0

    lo = diff_at_stop - 1.96 * se_at_stop
    hi = diff_at_stop + 1.96 * se_at_stop
    covered = (lo <= trial.true_diff) & (trial.true_diff <= hi)

    if rejected.any():
        est = float(diff_at_stop[rejected].mean())
        cov_r = float(covered[rejected].mean())
    else:
        est, cov_r = float("nan"), float("nan")
    bias = (est / trial.true_diff - 1.0) if trial.true_diff != 0 else float("nan")

    return Outcome(
        name=name,
        reject_rate=float(rejected.mean()),
        expected_n=float(n_at_stop.mean()),
        median_n=float(np.median(n_at_stop)),
        est_at_stop=est,
        est_bias=bias,
        ci_coverage=float(covered.mean()),
        ci_coverage_rejected=cov_r,
    )


def with_futility(
    stat: np.ndarray,
    reject_bounds: np.ndarray,
    futility_bounds: Optional[np.ndarray],
    signed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stop for success OR for hopelessness. Returns (reject_idx, stop_idx).

    A futility boundary is the half of sequential design that costs almost
    nothing and gets left out: it never adds a false positive, it only ends
    experiments that are not going anywhere. `futility_bounds` entries of -inf
    mean "no futility check at this look"; with `signed=True` the check is on
    the signed statistic, which is what "it is flat or negative, kill it"
    actually means.
    """
    n_sims, K = stat.shape
    reject_idx = np.full(n_sims, -1)
    stop_idx = np.full(n_sims, K - 1)
    live = np.ones(n_sims, dtype=bool)
    for k in range(K):
        win = live & (np.abs(stat[:, k]) >= reject_bounds[k])
        reject_idx[win] = k
        stop_idx[win] = k
        live &= ~win
        if futility_bounds is not None and k < K - 1:
            s = stat[:, k] if signed else np.abs(stat[:, k])
            quit_ = live & (s < futility_bounds[k])
            stop_idx[quit_] = k
            live &= ~quit_
    return reject_idx, stop_idx


def score_with_stop(trial: Trial, reject_idx: np.ndarray, stop_idx: np.ndarray, name: str) -> Outcome:
    """Like `score`, but the sample size comes from a stopping index that can
    differ from the rejection index (an experiment stopped for futility ran
    fewer visitors and did not reject)."""
    rows = np.arange(trial.z.shape[0])
    n_at_stop = trial.looks[stop_idx].astype(float)
    diff_at_stop = trial.diff[rows, stop_idx]
    se_at_stop = trial.se[rows, stop_idx]
    rejected = reject_idx >= 0
    lo = diff_at_stop - 1.96 * se_at_stop
    hi = diff_at_stop + 1.96 * se_at_stop
    covered = (lo <= trial.true_diff) & (trial.true_diff <= hi)
    if rejected.any():
        est = float(diff_at_stop[rejected].mean())
        cov_r = float(covered[rejected].mean())
    else:
        est, cov_r = float("nan"), float("nan")
    bias = (est / trial.true_diff - 1.0) if trial.true_diff != 0 else float("nan")
    return Outcome(
        name=name,
        reject_rate=float(rejected.mean()),
        expected_n=float(n_at_stop.mean()),
        median_n=float(np.median(n_at_stop)),
        est_at_stop=est,
        est_bias=bias,
        ci_coverage=float(covered.mean()),
        ci_coverage_rejected=cov_r,
    )


BOUNDARY_BUILDERS: Dict[str, object] = {
    "naive": naive_bounds,
    "bonferroni": bonferroni_bounds,
    "pocock": pocock_bounds,
    "obf": obf_bounds,
}
