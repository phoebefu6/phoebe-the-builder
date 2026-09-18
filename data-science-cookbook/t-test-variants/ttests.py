"""Four procedures called "the t-test", and a Monte-Carlo harness that measures what each one does.

The named tests are implemented from the formulas rather than called out of scipy, so the
notebook can carry a self-contained copy. Every implementation is checked against scipy to
1e-12 in test_ttests.py - the point of the build is the measured error rate, and a measured
error rate is worthless if the statistic underneath it is wrong.

Deliberately NOT here: the conditional "run Levene, then pick a test" procedure. That is its
own backlog item (`assumption-pretest-cost`) and it has its own finding to report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

ALPHA = 0.05
DISTRIBUTIONS = ("normal", "lognormal", "t3", "uniform")


@dataclass(frozen=True)
class TTestResult:
    """One test on one pair of samples."""

    name: str
    t: float
    df: float
    p: float
    note: str = ""


def _degenerate(name: str, note: str) -> TTestResult:
    """Both groups constant, or too few observations: the statistic does not exist.

    Returning p=1.0 is the conservative reading - no evidence - and the note says why, so a
    caller never mistakes an undefined statistic for a null result that was actually computed.
    """
    return TTestResult(name=name, t=float("nan"), df=float("nan"), p=1.0, note=note)


def student_t(x: np.ndarray, y: np.ndarray) -> TTestResult:
    """Two-sample t with a POOLED variance. Assumes the two populations share one sigma."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return _degenerate("student", "needs at least 2 observations per group")
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    df = n1 + n2 - 2
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    se = math.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return _degenerate("student", "zero variance in both groups")
    t = (x.mean() - y.mean()) / se
    return TTestResult("student", t, float(df), float(2 * stats.t.sf(abs(t), df)))


def welch_t(x: np.ndarray, y: np.ndarray) -> TTestResult:
    """Two-sample t with SEPARATE variances and Satterthwaite degrees of freedom."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        return _degenerate("welch", "needs at least 2 observations per group")
    a, b = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    se2 = a + b
    if se2 == 0.0:
        return _degenerate("welch", "zero variance in both groups")
    df = se2**2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    t = (x.mean() - y.mean()) / math.sqrt(se2)
    return TTestResult("welch", t, float(df), float(2 * stats.t.sf(abs(t), df)))


def paired_t(x: np.ndarray, y: np.ndarray) -> TTestResult:
    """Paired t. Identical, to machine precision, to one_sample_t on the differences."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError(f"paired_t needs equal-length samples, got {x.size} and {y.size}")
    out = one_sample_t(x - y)
    return TTestResult("paired", out.t, out.df, out.p, out.note)


def one_sample_t(x: np.ndarray, mu: float = 0.0) -> TTestResult:
    """One-sample t against a fixed mu."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return _degenerate("one_sample", "needs at least 2 observations")
    se = x.std(ddof=1) / math.sqrt(n)
    if se == 0.0:
        return _degenerate("one_sample", "zero variance")
    t = (x.mean() - mu) / se
    return TTestResult("one_sample", t, float(n - 1), float(2 * stats.t.sf(abs(t), n - 1)))


# ----------------------------------------------------------------------------------
# Vectorised p-values. The Monte-Carlo below runs tens of thousands of replicates per
# cell; looping the scalar functions above would make the Streamlit app unusable, so the
# same two formulas are written once more over a (reps, n) array. test_ttests.py checks
# the vectorised and scalar paths agree, because two implementations of one formula is
# exactly the kind of thing that drifts.
# ----------------------------------------------------------------------------------


def student_p_vec(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n1, n2 = X.shape[1], Y.shape[1]
    v1, v2 = X.var(axis=1, ddof=1), Y.var(axis=1, ddof=1)
    df = n1 + n2 - 2
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    se = np.sqrt(pooled * (1.0 / n1 + 1.0 / n2))
    t = (X.mean(axis=1) - Y.mean(axis=1)) / se
    return 2 * stats.t.sf(np.abs(t), df)


def welch_p_vec(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n1, n2 = X.shape[1], Y.shape[1]
    a, b = X.var(axis=1, ddof=1) / n1, Y.var(axis=1, ddof=1) / n2
    se2 = a + b
    df = se2**2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    t = (X.mean(axis=1) - Y.mean(axis=1)) / np.sqrt(se2)
    return 2 * stats.t.sf(np.abs(t), df)


def zscore_p_vec(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Welch's statistic read off the NORMAL curve instead of a t curve.

    Included because it is what every hand-rolled "significance" helper does, and the whole
    cost of the shortcut is the tail of a t with small df.
    """
    n1, n2 = X.shape[1], Y.shape[1]
    se2 = X.var(axis=1, ddof=1) / n1 + Y.var(axis=1, ddof=1) / n2
    z = (X.mean(axis=1) - Y.mean(axis=1)) / np.sqrt(se2)
    return 2 * stats.norm.sf(np.abs(z))


# ----------------------------------------------------------------------------------
# Draws
# ----------------------------------------------------------------------------------


def draw(rng: np.random.Generator, reps: int, n: int, sd: float, dist: str, shift: float = 0.0) -> np.ndarray:
    """(reps, n) draws with population mean `shift` and population sd `sd`.

    Every family is standardised by its THEORETICAL mean and sd, not the sample's, so the null
    being tested is exactly true and the measured rejection rate is a Type I error rather than
    a bias the standardisation introduced.
    """
    if dist == "normal":
        z = rng.standard_normal((reps, n))
    elif dist == "lognormal":
        sigma = 1.0
        raw = rng.lognormal(mean=0.0, sigma=sigma, size=(reps, n))
        m = math.exp(sigma**2 / 2)
        s = math.sqrt((math.exp(sigma**2) - 1) * math.exp(sigma**2))
        z = (raw - m) / s
    elif dist == "t3":
        nu = 3.0
        z = rng.standard_t(nu, size=(reps, n)) / math.sqrt(nu / (nu - 2))
    elif dist == "uniform":
        z = (rng.random((reps, n)) - 0.5) * math.sqrt(12.0)
    else:
        raise ValueError(f"unknown distribution {dist!r}; expected one of {DISTRIBUTIONS}")
    return z * sd + shift


@dataclass(frozen=True)
class Cell:
    """One condition of the study."""

    n1: int
    n2: int
    sd1: float = 1.0
    sd2: float = 1.0
    dist: str = "normal"
    shift: float = 0.0

    @property
    def sd_ratio(self) -> float:
        return self.sd2 / self.sd1

    @property
    def balance(self) -> str:
        if self.n1 == self.n2:
            return "balanced"
        bigger_has_bigger_sd = (self.n1 > self.n2) == (self.sd1 > self.sd2)
        return "large-n has large-sd" if bigger_has_bigger_sd else "small-n has large-sd"

    def label(self) -> str:
        return f"n={self.n1}/{self.n2}, sd={self.sd1:g}/{self.sd2:g}, {self.dist}"


TESTS = ("student", "welch", "z-shortcut")


def simulate(cell: Cell, reps: int = 40_000, alpha: float = ALPHA, seed: int = 0) -> Dict[str, float]:
    """Rejection rate of each test under `cell`. Under shift=0 that rate IS the Type I error."""
    rng = np.random.default_rng(seed)
    X = draw(rng, reps, cell.n1, cell.sd1, cell.dist, 0.0)
    Y = draw(rng, reps, cell.n2, cell.sd2, cell.dist, cell.shift)
    return {
        "student": float((student_p_vec(X, Y) < alpha).mean()),
        "welch": float((welch_p_vec(X, Y) < alpha).mean()),
        "z-shortcut": float((zscore_p_vec(X, Y) < alpha).mean()),
    }


def wilson(k: float, n: int, conf: float = 0.95) -> Tuple[float, float]:
    """Wilson interval on a rejection RATE.

    A simulated 0.083 is itself an estimate; reporting it bare invites reading noise as an
    effect. At 40k reps the half-width near 0.05 is about 0.002, which is what makes the
    Student-vs-Welch gaps below readable at all.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    k = float(k)
    z = float(stats.norm.isf((1 - conf) / 2))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def verdict(rate: float, reps: int, alpha: float = ALPHA) -> str:
    """Is a measured rate distinguishable from nominal, and in which direction."""
    lo, hi = wilson(rate * reps, reps)
    if lo > alpha:
        return f"INFLATED ({rate / alpha:.2f}x nominal)"
    if hi < alpha:
        return f"conservative ({rate / alpha:.2f}x nominal)"
    return "nominal"


def type_i_grid(
    sd_ratios: List[float],
    configs: List[Tuple[int, int]],
    dist: str = "normal",
    reps: int = 40_000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i, (n1, n2) in enumerate(configs):
        for j, r in enumerate(sd_ratios):
            cell = Cell(n1=n1, n2=n2, sd1=1.0, sd2=r, dist=dist)
            rates = simulate(cell, reps=reps, seed=seed + 1000 * i + j)
            rows.append({"cell": cell, "rates": rates})
    return rows


def power_curve(
    shifts: List[float],
    n1: int,
    n2: int,
    sd1: float = 1.0,
    sd2: float = 1.0,
    dist: str = "normal",
    reps: int = 20_000,
    seed: int = 7,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for k, s in enumerate(shifts):
        cell = Cell(n1=n1, n2=n2, sd1=sd1, sd2=sd2, dist=dist, shift=s)
        out.append({"shift": s, "rates": simulate(cell, reps=reps, seed=seed + k)})
    return out


def scipy_agreement(seed: int = 3, n1: int = 17, n2: int = 9) -> Dict[str, float]:
    """Largest absolute p-value disagreement with scipy over 200 random samples."""
    rng = np.random.default_rng(seed)
    worst = {"student": 0.0, "welch": 0.0, "paired": 0.0, "one_sample": 0.0}
    for _ in range(200):
        x = rng.standard_normal(n1) * rng.uniform(0.5, 3)
        y = rng.standard_normal(n2) * rng.uniform(0.5, 3)
        worst["student"] = max(worst["student"], abs(student_t(x, y).p - stats.ttest_ind(x, y, equal_var=True).pvalue))
        worst["welch"] = max(worst["welch"], abs(welch_t(x, y).p - stats.ttest_ind(x, y, equal_var=False).pvalue))
        xp, yp = x[:n2], y
        worst["paired"] = max(worst["paired"], abs(paired_t(xp, yp).p - stats.ttest_rel(xp, yp).pvalue))
        worst["one_sample"] = max(worst["one_sample"], abs(one_sample_t(x).p - stats.ttest_1samp(x, 0.0).pvalue))
    return worst


def welch_df_of(x: np.ndarray, y: np.ndarray) -> float:
    return welch_t(x, y).df


def describe(cell: Cell, rates: Dict[str, float], reps: int) -> str:
    parts = [f"{cell.label():38s}"]
    for name in TESTS:
        parts.append(f"{name}={rates[name]:.4f}")
    return "  ".join(parts)


__all__ = [
    "ALPHA",
    "Cell",
    "DISTRIBUTIONS",
    "TESTS",
    "TTestResult",
    "describe",
    "draw",
    "one_sample_t",
    "paired_t",
    "power_curve",
    "scipy_agreement",
    "simulate",
    "student_p_vec",
    "student_t",
    "type_i_grid",
    "verdict",
    "welch_df_of",
    "welch_p_vec",
    "welch_t",
    "wilson",
    "zscore_p_vec",
]
