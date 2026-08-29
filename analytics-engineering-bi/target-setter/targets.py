"""A target is not a number. It is a method plus a claim about the future.

This module holds three things and nothing else:

1.  A metric history with a *known* data-generating process, so that every
    claim about a target can be checked against the truth that produced it.
2.  Eleven target-setting methods, each of which a finance or product team
    would defend in a planning meeting without embarrassment.
3.  A rolling-origin backtest that sets a target at every historical origin
    and then looks at what actually happened.

Nothing here decides what a good target is. The point of the exercise is
that the eleven methods disagree, that the disagreement is larger than the
gap the target is meant to close, and that the statistic everyone reaches
for -- the hit rate -- measures the method rather than the team.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 1. The world
# --------------------------------------------------------------------------

# Monthly signups for a mid-market SaaS product. Multiplicative, because
# metrics like this are: growth compounds, seasonality scales, and the noise
# is proportional rather than additive.
#
#     y_t = BASE * (1 + G)**t * S[t % 12] * exp(eps_t),  eps_t ~ N(0, SIGMA**2)
#
# Everything downstream is derived from these five constants, so any claim
# in evidence.py can be traced back to a number written here.

BASE = 1000.0        # month-zero level before seasonality
G = 0.0125           # 1.25% per month == 16.1% per year of real trend growth
SIGMA = 0.12         # lognormal noise; 12% is ordinary for a monthly metric
N_MONTHS = 132       # eleven years, so the backtest has 93 origins
SEED = 159           # the build day, so the series is reproducible by name

# Seasonal index, mean 1.0 by construction: a December peak, a summer trough.
SEASONAL = np.array(
    [0.92, 0.95, 1.04, 1.02, 0.99, 0.88, 0.83, 0.87, 1.06, 1.12, 1.14, 1.18]
)

# The plan of record for headcount, which is what a bottom-up target is built
# from. It grows in steps, and deliberately slower than the metric does --
# that gap is the whole reconciliation problem in section 5 of the evidence.
HEADCOUNT_START = 8.0
HEADCOUNT_STEP_MONTHS = 6
HEADCOUNT_STEP = 1.0

HORIZON = 3          # a target covers the next quarter, summed
MIN_HISTORY = 36     # 24 months are needed before a YoY method has an answer

# External claims that enter a target from outside the history.
MARKET_GROWTH_Q = 0.030   # "the category grows 3% a quarter", from a report
BOARD_MULTIPLE = 1.40     # "we want to be 40% up on last year"


def seasonal_index(t: int) -> float:
    """The seasonal multiplier for calendar month index ``t``."""
    return float(SEASONAL[t % 12])


def headcount(t: int) -> float:
    """Planned headcount in month ``t``, from the plan of record."""
    return HEADCOUNT_START + HEADCOUNT_STEP * (t // HEADCOUNT_STEP_MONTHS)


def expected_level(t: int) -> float:
    """E[y_t] under the DGP.

    Note the ``exp(SIGMA**2 / 2)``: the lognormal mean sits *above* the
    lognormal median. A target set at the expectation is therefore missed
    more often than it is hit, before anybody has done any work. That is
    section 3 of the evidence, and it is a property of the arithmetic
    rather than of the team.
    """
    return BASE * (1.0 + G) ** t * seasonal_index(t) * np.exp(SIGMA**2 / 2.0)


def median_level(t: int) -> float:
    """The median of y_t, which is the expectation without the skew term."""
    return BASE * (1.0 + G) ** t * seasonal_index(t)


def make_history(n_months: int = N_MONTHS, seed: int = SEED) -> np.ndarray:
    """One realised monthly series from the DGP above."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    trend = BASE * (1.0 + G) ** t
    seas = np.array([seasonal_index(int(i)) for i in t])
    noise = np.exp(rng.normal(0.0, SIGMA, size=n_months))
    return trend * seas * noise


def truth_quarter(origin: int, horizon: int = HORIZON) -> Tuple[float, float]:
    """(mean, median) of the sum of the ``horizon`` months after ``origin``.

    The sum of lognormals has no closed form, so the mean is exact (it is
    additive) and the median is approximated by the sum of medians. Both are
    only ever used as reference lines, never as a target.
    """
    months = range(origin, origin + horizon)
    return (
        float(sum(expected_level(m) for m in months)),
        float(sum(median_level(m) for m in months)),
    )


# --------------------------------------------------------------------------
# 2. The methods
# --------------------------------------------------------------------------
#
# Each method has the signature (history, origin) -> target, where ``history``
# is y[:origin] -- everything known when the target is set -- and the target
# covers months origin .. origin+HORIZON-1. No method may look forward.

Method = Callable[[np.ndarray, int], float]


def m_last_quarter(hist: np.ndarray, origin: int) -> float:
    """Do again what we just did. The lowest-effort defensible target."""
    return float(hist[-HORIZON:].sum())


def m_run_rate(hist: np.ndarray, origin: int) -> float:
    """Annualise the latest month. The most common target in the wild."""
    return float(hist[-1] * HORIZON)


def m_seasonal_naive(hist: np.ndarray, origin: int) -> float:
    """The same quarter last year, flat. Concedes there is no growth."""
    return float(hist[-12 : -12 + HORIZON].sum())


def m_yoy_growth(hist: np.ndarray, origin: int) -> float:
    """Same quarter last year, grown by the trailing twelve-month YoY rate."""
    last_12 = hist[-12:].sum()
    prior_12 = hist[-24:-12].sum()
    rate = last_12 / prior_12 if prior_12 > 0 else 1.0
    return float(hist[-12 : -12 + HORIZON].sum() * rate)


def _fit_log_linear_seasonal(
    hist: np.ndarray,
) -> Tuple[float, float, np.ndarray, float]:
    """Fit log(y) = intercept + slope*t + seasonal[t%12] and return the pieces.

    One fitter, used by both trend-with-seasonality methods and by the
    prediction interval, so that the three cannot drift apart. The residual
    sd is taken *after* the seasonal index is removed -- taking it before
    inflates sigma with seasonal variance, and section 3 measures what that
    inflation does to a target.
    """
    y = np.log(hist)
    t = np.arange(len(hist), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    resid = y - (intercept + slope * t)
    idx = np.array([resid[m::12].mean() for m in range(12)])
    idx = idx - idx.mean()
    deseasonalised = resid - idx[np.arange(len(hist)) % 12]
    sd = float(deseasonalised.std(ddof=13))
    return float(intercept), float(slope), idx, sd


def m_trend_ols(hist: np.ndarray, origin: int) -> float:
    """OLS on log(y) over the whole history, extrapolated. No seasonality.

    This is the method that looks the most rigorous in a slide and carries a
    known defect: fitting a trend through a seasonal series and projecting it
    puts the seasonal error of the target quarter straight into the number.
    """
    y = np.log(hist)
    t = np.arange(len(hist), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    fut = np.arange(origin, origin + HORIZON, dtype=float)
    return float(np.exp(intercept + slope * fut).sum())


def m_trend_seasonal(hist: np.ndarray, origin: int) -> float:
    """Log-linear trend with an estimated seasonal index, corrected for skew.

    This is the reference forecast: it has the shape of the DGP, so it is
    close to unbiased for the *mean*. It is included to show that being the
    best available forecast is not the same as being a target anybody wants.
    """
    intercept, slope, idx, sd = _fit_log_linear_seasonal(hist)
    out = 0.0
    for m in range(origin, origin + HORIZON):
        out += float(np.exp(intercept + slope * m + idx[m % 12] + sd**2 / 2.0))
    return out


def m_trend_seasonal_median(hist: np.ndarray, origin: int) -> float:
    """The same forecast without the skew correction: a median target."""
    intercept, slope, idx, _sd = _fit_log_linear_seasonal(hist)
    out = 0.0
    for m in range(origin, origin + HORIZON):
        out += float(np.exp(intercept + slope * m + idx[m % 12]))
    return out


def m_benchmark(hist: np.ndarray, origin: int) -> float:
    """Last quarter grown at the published category rate. An outside anchor."""
    return float(hist[-HORIZON:].sum() * (1.0 + MARKET_GROWTH_Q))


def m_capacity(hist: np.ndarray, origin: int) -> float:
    """Bottom-up: planned heads times what a head has recently delivered."""
    recent = hist[-12:]
    heads_recent = np.array([headcount(origin - 12 + i) for i in range(12)])
    productivity = float((recent / heads_recent).mean())
    planned = sum(headcount(m) for m in range(origin, origin + HORIZON))
    return float(productivity * planned)


def m_top_down(hist: np.ndarray, origin: int) -> float:
    """Last year's same quarter times the multiple the board asked for."""
    return float(hist[-12 : -12 + HORIZON].sum() * BOARD_MULTIPLE)


def m_stretch_best_ever(hist: np.ndarray, origin: int) -> float:
    """The best run of HORIZON consecutive months the company has ever had."""
    windows = np.convolve(hist, np.ones(HORIZON), mode="valid")
    return float(windows.max())


def m_split_difference(hist: np.ndarray, origin: int) -> float:
    """Meet in the middle between bottom-up and top-down.

    The compromise that ends the planning meeting. It is above what the
    resourcing supports and below what the board asked for, so it is
    unachievable by the first argument and unacceptable to the second.
    """
    return 0.5 * (m_capacity(hist, origin) + m_top_down(hist, origin))


METHODS: Dict[str, Method] = {
    "seasonal_naive": m_seasonal_naive,
    "last_quarter": m_last_quarter,
    "run_rate": m_run_rate,
    "capacity": m_capacity,
    "trend_seasonal_median": m_trend_seasonal_median,
    "trend_seasonal": m_trend_seasonal,
    "trend_ols": m_trend_ols,
    "benchmark": m_benchmark,
    "yoy_growth": m_yoy_growth,
    "split_difference": m_split_difference,
    "top_down": m_top_down,
    "stretch_best_ever": m_stretch_best_ever,
}

# How each target would be described in the room it is set in.
PROVENANCE: Dict[str, str] = {
    "seasonal_naive": "same quarter last year, flat",
    "last_quarter": "repeat the quarter we just had",
    "run_rate": "latest month annualised",
    "capacity": "bottom-up: planned heads x recent productivity",
    "trend_seasonal_median": "trend + seasonality, median forecast",
    "trend_seasonal": "trend + seasonality, mean forecast",
    "trend_ols": "log-linear trend extrapolated",
    "benchmark": "last quarter + published category growth",
    "yoy_growth": "last year's quarter + trailing YoY rate",
    "split_difference": "midpoint of bottom-up and top-down",
    "top_down": "last year's quarter x the board multiple",
    "stretch_best_ever": "the best quarter we have ever had",
}


# --------------------------------------------------------------------------
# 3. The backtest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodResult:
    """What one method did across every origin in the backtest."""

    name: str
    targets: np.ndarray      # the target set at each origin
    actuals: np.ndarray      # what the next HORIZON months actually summed to
    truth_mean: np.ndarray   # E[sum] under the DGP, for reference only

    @property
    def hits(self) -> np.ndarray:
        return self.actuals >= self.targets

    @property
    def hit_rate(self) -> float:
        return float(self.hits.mean())

    @property
    def ambition(self) -> float:
        """Mean target as a multiple of the truth it is aiming at.

        This is the only property of a target that is fixed at the moment it
        is set. Section 2 of the evidence shows the hit rate is a monotone
        function of it, which is why a hit rate cannot grade a team.
        """
        return float((self.targets / self.truth_mean).mean())

    @property
    def mape(self) -> float:
        """Mean absolute percentage error against the realised actual."""
        return float((np.abs(self.targets - self.actuals) / self.actuals).mean())

    @property
    def bias(self) -> float:
        """Mean signed error as a fraction of the actual."""
        return float(((self.targets - self.actuals) / self.actuals).mean())


def origins(
    series: np.ndarray, min_history: int = MIN_HISTORY, horizon: int = HORIZON
) -> List[int]:
    """Every month at which a target can be set and later scored."""
    return list(range(min_history, len(series) - horizon + 1))


def backtest(
    series: np.ndarray,
    methods: Optional[Dict[str, Method]] = None,
    min_history: int = MIN_HISTORY,
    horizon: int = HORIZON,
) -> Dict[str, MethodResult]:
    """Set a target with every method at every origin, then score it."""
    methods = METHODS if methods is None else methods
    ors = origins(series, min_history, horizon)
    actuals = np.array([series[o : o + horizon].sum() for o in ors])
    truth = np.array([truth_quarter(o, horizon)[0] for o in ors])
    out: Dict[str, MethodResult] = {}
    for name, fn in methods.items():
        targets = np.array([fn(series[:o], o) for o in ors])
        out[name] = MethodResult(name, targets, actuals, truth)
    return out


def targets_at(series: np.ndarray, origin: int) -> Dict[str, float]:
    """Every method's target for one specific origin."""
    return {name: fn(series[:origin], origin) for name, fn in METHODS.items()}


def prediction_interval(
    series: np.ndarray, origin: int, level: float = 0.80, horizon: int = HORIZON
) -> Tuple[float, float]:
    """A ``level`` interval for the next-quarter sum, from in-sample residuals.

    Deliberately simple: fit the reference model, take the residual sd, and
    propagate it through the horizon assuming independence. It is only used
    to answer one question -- how many of the pairwise gaps between the
    twelve methods are smaller than the width of this interval.
    """
    from scipy import stats

    hist = series[:origin]
    intercept, slope, idx, sd = _fit_log_linear_seasonal(hist)

    med = np.array(
        [np.exp(intercept + slope * m + idx[m % 12]) for m in range(origin, origin + horizon)]
    )
    point = float((med * np.exp(sd**2 / 2.0)).sum())
    # Variance of a sum of independent lognormals.
    var = float((med**2 * np.exp(sd**2) * (np.exp(sd**2) - 1.0)).sum())
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    half = z * np.sqrt(var)
    return (point - half, point + half)


# --------------------------------------------------------------------------
# 4. Repeating the eleven years
# --------------------------------------------------------------------------
#
# One realised history gives one hit rate, and a hit rate computed on 94
# overlapping quarters looks like a precise measurement. It is not. These
# helpers re-run the same eleven years from a different draw of the same
# process, so that every statistic can be reported with the spread it has
# across paths rather than the single value one path happened to give.

N_PATHS = 500
PATH_SEED0 = 20_000


def multipath(
    n_paths: int = N_PATHS, seed0: int = PATH_SEED0
) -> Dict[str, Dict[str, np.ndarray]]:
    """Backtest every method on ``n_paths`` independent draws of the DGP.

    Returns ``{method: {"hit_rate": ..., "ambition": ..., "mape": ...,
    "bias": ...}}`` with one entry per path in each array.
    """
    keys = ("hit_rate", "ambition", "mape", "bias")
    acc: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in keys} for m in METHODS
    }
    for i in range(n_paths):
        res = backtest(make_history(seed=seed0 + i))
        for name, r in res.items():
            acc[name]["hit_rate"].append(r.hit_rate)
            acc[name]["ambition"].append(r.ambition)
            acc[name]["mape"].append(r.mape)
            acc[name]["bias"].append(r.bias)
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in acc.items()}


def oracle_hit_rates(
    n_paths: int = N_PATHS, seed0: int = PATH_SEED0
) -> Dict[str, float]:
    """Hit rates for targets set at the *true* mean and median of the future.

    No estimation is involved: these are the hit rates a team would get if
    somebody handed them the data-generating process itself. They are the
    ceiling on what any forecast-based target can claim.
    """
    mean_hits: List[float] = []
    med_hits: List[float] = []
    for i in range(n_paths):
        s = make_history(seed=seed0 + i)
        ors = origins(s)
        act = np.array([s[o : o + HORIZON].sum() for o in ors])
        tm = np.array([truth_quarter(o)[0] for o in ors])
        td = np.array([truth_quarter(o)[1] for o in ors])
        mean_hits.append(float((act >= tm).mean()))
        med_hits.append(float((act >= td).mean()))
    return {
        "mean_target": float(np.mean(mean_hits)),
        "mean_target_sd": float(np.std(mean_hits)),
        "median_target": float(np.mean(med_hits)),
        "median_target_sd": float(np.std(med_hits)),
    }


def ensemble_result(series: np.ndarray) -> MethodResult:
    """The average of all twelve targets, scored like any other method."""
    res = backtest(series)
    names = list(METHODS)
    stack = np.vstack([res[n].targets for n in names])
    ref = res[names[0]]
    return MethodResult("ensemble", stack.mean(axis=0), ref.actuals, ref.truth_mean)


def quarters_to_distinguish(p0: float, p1: float, alpha: float = 0.05,
                            power: float = 0.80) -> int:
    """Quarters needed to tell a ``p0`` hitter from a ``p1`` hitter.

    Normal approximation to a one-sided two-proportion test. Used once, to
    put a number on how much evidence a hit rate carries.
    """
    from scipy import stats

    za = float(stats.norm.ppf(1 - alpha))
    zb = float(stats.norm.ppf(power))
    pbar = (p0 + p1) / 2.0
    num = (
        za * np.sqrt(2 * pbar * (1 - pbar))
        + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    ) ** 2
    return int(np.ceil(num / (p1 - p0) ** 2))
