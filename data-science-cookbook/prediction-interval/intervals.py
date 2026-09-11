"""Prediction intervals as CLAIMS, and the measurement that decides whether a claim held.

Day 172 (`forecast-backtest`) treated the backtest protocol as an estimator of a point error.
This build takes the next step: an interval is not an estimate, it is a claim about a
frequency - "95% of future points will land inside" - and a claim has to be checked against
the frequency, at the horizon and in the regime where it will be used.

Three things live here and nothing else:

* a process whose future can be redrawn (so true coverage is measurable rather than assumed),
  with an optional volatility regime, because the interesting failures are conditional;
* six interval methods spanning what people actually ship - the textbook Gaussian band with
  sqrt(h) widening, the same band with the regression's own parameter-uncertainty term,
  empirical residual quantiles, rolling-origin h-step quantiles, split conformal, and a
  simulated residual path;
* the measurements that decide between them: pointwise coverage by horizon, coverage
  CONDITIONAL on regime, whole-path coverage, mean width, the interval score, and the power
  of the coverage check itself.

Nothing here reads results.json. Every number in the evidence run is computed live.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

PERIOD = 12
Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054,
     0.99: 2.5758293035489004}


# --------------------------------------------------------------------------- the process


def make_series(
    rng: np.random.Generator,
    n: int,
    level: float = 100.0,
    trend: float = 0.15,
    season_amp: float = 8.0,
    period: int = PERIOD,
    sigma: float = 4.0,
    rho: float = 0.0,
    vol_period: Optional[int] = None,
    vol_ratio: float = 1.0,
    vol_phase: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (series, per-point noise sd).

    `vol_period` switches the noise between a quiet and a volatile block of that length, with
    var_hi / var_lo = vol_ratio and the AVERAGE VARIANCE pinned at sigma^2:
    var_hi = sigma^2 * 2r/(1+r), var_lo = sigma^2 * 2/(1+r). The naive version of this -
    multiplying and dividing sigma by sqrt(r) - leaves the mean variance at
    sigma^2 (r + 1/r) / 2, which is 4.6x too loud at r=9 and turns the comparison into a
    test of "more noise" rather than of "the same noise, unevenly spread". That constancy is the
    point: a method can hit its marginal coverage exactly while being wrong in both regimes,
    and no marginal check can see it.

    The block PHASE is drawn per series unless pinned. Without that, the regime is a
    deterministic function of t, a fixed training length always forecasts into the same
    regime, and the conditional split has nothing in one of its cells - which is how the
    first version of this build produced a NaN instead of a finding.
    """
    t = np.arange(n, dtype=float)
    signal = level + trend * t + season_amp * np.sin(2 * np.pi * t / period)
    if vol_period is None or vol_ratio == 1.0:
        sd = np.full(n, float(sigma))
    else:
        phase = int(rng.integers(0, 2 * vol_period)) if vol_phase is None else int(vol_phase)
        hi = ((t + phase) // vol_period).astype(int) % 2 == 1
        r = float(vol_ratio)
        sd = np.where(hi, sigma * np.sqrt(2 * r / (1 + r)), sigma * np.sqrt(2 / (1 + r)))
    if rho == 0.0:
        noise = rng.normal(0.0, 1.0, n) * sd
    else:
        e = rng.normal(0.0, np.sqrt(1 - rho**2), n)
        z = np.empty(n)
        z[0] = rng.normal(0.0, 1.0)
        for i in range(1, n):
            z[i] = rho * z[i - 1] + e[i]
        noise = z * sd
    return signal + noise, sd


# --------------------------------------------------------------------------- forecasters


def _cols(t: np.ndarray, mu: float, sd: float, period: int) -> np.ndarray:
    ts = (t - mu) / sd
    return np.column_stack([np.ones_like(ts), ts,
                            np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)])


def fit_trend_season(y: np.ndarray, period: int = PERIOD):
    """Least squares on [1, t, sin, cos]. Returns everything the interval methods need."""
    n = len(y)
    t = np.arange(n, dtype=float)
    mu, sd = t.mean(), max(t.std(), 1e-9)
    X = _cols(t, mu, sd, period)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - X.shape[1], 1)
    return {"beta": beta, "mu": mu, "sd": sd, "period": period, "n": n,
            "resid": resid, "s2": float(resid @ resid / dof), "XtXinv": np.linalg.pinv(X.T @ X)}


def predict(fit: Dict, h: int) -> np.ndarray:
    t_out = np.arange(fit["n"], fit["n"] + h, dtype=float)
    return _cols(t_out, fit["mu"], fit["sd"], fit["period"]) @ fit["beta"]


def leverage(fit: Dict, h: int) -> np.ndarray:
    """x0' (X'X)^-1 x0 at each future point: the parameter-uncertainty term people drop."""
    t_out = np.arange(fit["n"], fit["n"] + h, dtype=float)
    X0 = _cols(t_out, fit["mu"], fit["sd"], fit["period"])
    return np.einsum("ij,jk,ik->i", X0, fit["XtXinv"], X0)


# --------------------------------------------------------------------------- the methods
# Each returns (lo, hi) of length h. Signature is uniform so the evidence run can loop.


def gaussian_iid(y: np.ndarray, h: int, conf: float = 0.95, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """The band almost everything ships: residual sd, widened by sqrt(h), times z."""
    fit = fit_trend_season(y)
    p = predict(fit, h)
    w = Z[conf] * np.sqrt(fit["s2"]) * np.sqrt(np.arange(1, h + 1))
    return p - w, p + w


def gaussian_flat(y: np.ndarray, h: int, conf: float = 0.95, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Residual sd, NOT widened. The right answer for a stationary direct forecast, and the
    wrong-looking one, which is the point: the sqrt(h) fan is an assumption about the model."""
    fit = fit_trend_season(y)
    p = predict(fit, h)
    w = np.full(h, Z[conf] * np.sqrt(fit["s2"]))
    return p - w, p + w


def gaussian_theory(y: np.ndarray, h: int, conf: float = 0.95, **kw) -> Tuple[np.ndarray, np.ndarray]:
    """Textbook regression prediction interval: s * sqrt(1 + x0'(X'X)^-1 x0). Includes the
    parameter uncertainty that grows as you extrapolate away from the training window."""
    fit = fit_trend_season(y)
    p = predict(fit, h)
    w = Z[conf] * np.sqrt(fit["s2"] * (1.0 + leverage(fit, h)))
    return p - w, p + w


def empirical_residual(y: np.ndarray, h: int, conf: float = 0.95, **kw):
    """Quantiles of the in-sample residuals - no normality assumption, same blind spot."""
    fit = fit_trend_season(y)
    p = predict(fit, h)
    a = (1 - conf) / 2
    lo_q, hi_q = np.quantile(fit["resid"], [a, 1 - a])
    return p + lo_q, p + hi_q


def rolling_empirical(y: np.ndarray, h: int, conf: float = 0.95, k: int = 30,
                      min_train: int = 96, **kw):
    """Quantiles of ACTUAL h-step errors from rolling origins - one quantile pair per horizon.

    This is the honest construction, and it is exactly Day 172's protocol reused: the errors it
    quantiles are the same errors that build's scoreboard averaged.
    """
    fit = fit_trend_season(y)
    p = predict(fit, h)
    errs = _rolling_errors(y, h, k, min_train)
    a = (1 - conf) / 2
    if len(errs) < 4:
        return gaussian_iid(y, h, conf)
    lo = np.array([np.quantile(errs[:, j][~np.isnan(errs[:, j])], a) for j in range(h)])
    hi = np.array([np.quantile(errs[:, j][~np.isnan(errs[:, j])], 1 - a) for j in range(h)])
    return p + lo, p + hi


def split_conformal(y: np.ndarray, h: int, conf: float = 0.95, k: int = 30,
                    min_train: int = 96, **kw):
    """Absolute-residual conformal on rolling h-step errors, with the finite-sample index.

    The quantile index ceil((m+1)(1-alpha))/m is what buys the >= 1-alpha marginal guarantee;
    using m instead of m+1 is the off-by-one that quietly removes it at small m.
    """
    fit = fit_trend_season(y)
    p = predict(fit, h)
    errs = _rolling_errors(y, h, k, min_train)
    if len(errs) < 4:
        return gaussian_iid(y, h, conf)
    w = np.empty(h)
    for j in range(h):
        e = np.abs(errs[:, j][~np.isnan(errs[:, j])])
        m = len(e)
        q = min(1.0, np.ceil((m + 1) * conf) / m)
        w[j] = np.quantile(e, q)
    return p - w, p + w


def bootstrap_path(y: np.ndarray, h: int, conf: float = 0.95, n_sim: int = 400,
                   seed: int = 0, **kw):
    """Resample in-sample residuals into simulated futures and take pointwise quantiles."""
    fit = fit_trend_season(y)
    p = predict(fit, h)
    rng = np.random.default_rng(seed)
    draws = rng.choice(fit["resid"], size=(n_sim, h), replace=True)
    a = (1 - conf) / 2
    return p + np.quantile(draws, a, axis=0), p + np.quantile(draws, 1 - a, axis=0)


METHODS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "gaussian sqrt(h)": gaussian_iid,
    "gaussian flat": gaussian_flat,
    "gaussian + parameter term": gaussian_theory,
    "empirical residual": empirical_residual,
    "rolling h-step quantile": rolling_empirical,
    "split conformal": split_conformal,
    "bootstrap path": bootstrap_path,
}


# --------------------------------------------------------------------------- machinery


def _rolling_errors(y: np.ndarray, h: int, k: int, min_train: int) -> np.ndarray:
    """(origins x h) matrix of actual h-step errors, newest origins first."""
    last = len(y) - h
    ors = sorted(list(range(last, min_train - 1, -1))[:k])
    out = np.full((len(ors), h), np.nan)
    for i, o in enumerate(ors):
        fit = fit_trend_season(y[:o])
        pred = predict(fit, h)
        act = y[o:o + h]
        out[i, :len(act)] = act - pred[:len(act)]
    return out


def coverage_run(
    method: Callable,
    h: int,
    n_train: int,
    reps: int,
    rng: np.random.Generator,
    conf: float = 0.95,
    **series_kw,
) -> Dict[str, np.ndarray]:
    """Draw `reps` independent futures, build the interval, record hits, width and regime.

    Returns per-horizon arrays so pointwise, conditional and whole-path coverage all come from
    the same run - three different questions about one set of intervals.
    """
    hits = np.zeros((reps, h), dtype=bool)
    width = np.zeros((reps, h))
    sd_at = np.zeros((reps, h))
    for i in range(reps):
        y, sd = make_series(rng, n_train + h, **series_kw)
        lo, hi = method(y[:n_train], h, conf)
        act = y[n_train:]
        hits[i] = (act >= lo) & (act <= hi)
        width[i] = hi - lo
        sd_at[i] = sd[n_train:]
    return {"hits": hits, "width": width, "sd": sd_at}


def interval_score(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray, conf: float) -> np.ndarray:
    """Winkler/interval score: width plus a penalty for each miss, scaled by 2/alpha.

    Proper, so it cannot be gamed by widening - unlike coverage, which always can.
    """
    alpha = 1 - conf
    below = np.clip(lo - actual, 0, None)
    above = np.clip(actual - hi, 0, None)
    return (hi - lo) + (2 / alpha) * (below + above)


def coverage_check_power(true_cov: float, nominal: float, m: int, k_eff_ratio: float = 1.0,
                         alpha: float = 0.05) -> float:
    """Power of a one-sided binomial check that coverage is below nominal, at m test points.

    `k_eff_ratio` is the fraction of m that is INDEPENDENT - Day 172 measured 1/20 for rolling
    windows stepped by 1. The check people run uses m; the honest one uses m * k_eff_ratio.
    """
    from math import erf, sqrt

    m_eff = max(1.0, m * k_eff_ratio)
    se0 = sqrt(nominal * (1 - nominal) / m_eff)
    se1 = sqrt(true_cov * (1 - true_cov) / m_eff)
    z = 1.6448536269514722 if alpha == 0.05 else 1.959963984540054
    crit = nominal - z * se0
    return float(0.5 * (1 + erf((crit - true_cov) / (se1 * sqrt(2)))))
