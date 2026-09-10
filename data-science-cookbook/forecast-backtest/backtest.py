"""Backtest protocols as ESTIMATORS of a forecaster's future error, and what each one gets wrong.

The object under test here is not a forecasting model - `ts-forecaster` (Day 118) already
built one. The object under test is the EVALUATION PROTOCOL. A single train/test split, a
rolling origin, a random k-fold on a lag table and a blocked CV all return a number that
gets published as "the accuracy"; every one of them is an estimator of a quantity, and the
quantity is not the same for all of them.

Three things live here and nothing else:

* a generator whose future can be drawn as many times as you like, so the ESTIMAND - the
  expected h-step error of a model trained on n points from this process - is computable to
  Monte Carlo precision instead of being approximated by a holdout;
* six forecasters, deliberately spanning a well-specified one, two under-specified ones and
  one flexible enough to fit the noise, so protocol RANKINGS have something to get wrong;
* the protocols themselves - single split, rolling origin (expanding and sliding), random
  k-fold on a supervised lag table, and blocked k-fold with an embargo - all returning the
  per-origin, per-horizon error matrix rather than a scalar, because the scalar is the part
  that hides the variance.

The frame this build inherits (Days 169-171): when a reported number is the output of a
SELECTION, it is the selection's number, not the object's. A single split selects an origin.
Best-of-M on that split selects a model. Both publish what the selection produced.

Nothing here reads results.json. Every number in the evidence run is computed live.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- the process

PERIOD = 12


def make_series(
    rng: np.random.Generator,
    n: int,
    level: float = 100.0,
    trend: float = 0.15,
    season_amp: float = 8.0,
    period: int = PERIOD,
    sigma: float = 4.0,
    rho: float = 0.0,
    break_at: Optional[int] = None,
    break_shift: float = 0.0,
    break_trend: Optional[float] = None,
) -> np.ndarray:
    """A series with a known signal and AR(1) noise, optionally with a regime break.

    rho is the noise autocorrelation. It is the parameter that decides how much a random
    k-fold leaks: with rho = 0 a neighbouring row carries no information about a held-out
    one, and the leaky protocol is merely inefficient rather than optimistic.
    """
    t = np.arange(n, dtype=float)
    signal = level + trend * t + season_amp * np.sin(2 * np.pi * t / period)
    if break_at is not None:
        after = t >= break_at
        signal = signal + break_shift * after
        if break_trend is not None:
            signal = signal + (break_trend - trend) * np.where(after, t - break_at, 0.0)
    if rho == 0.0:
        noise = rng.normal(0.0, sigma, n)
    else:
        e = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n)
        noise = np.empty(n)
        noise[0] = rng.normal(0.0, sigma)
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + e[i]
    return signal + noise


# --------------------------------------------------------------------------- forecasters
# Every forecaster is fit(y_train) -> h-step path. Pure numpy, no state, no libraries.


def f_naive(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    return np.repeat(y[-1], h)


def f_snaive(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    tail = y[-period:]
    return np.array([tail[i % period] for i in range(h)])


def f_drift(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    slope = (y[-1] - y[0]) / max(len(y) - 1, 1)
    return y[-1] + slope * np.arange(1, h + 1)


def f_mean(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    return np.repeat(y.mean(), h)


def _design(t: np.ndarray, period: int, poly: int) -> np.ndarray:
    """Intercept + poly powers of a centred, scaled t + one sin/cos pair for the season."""
    ts = (t - t.mean()) / max(t.std(), 1e-9)
    cols = [np.ones_like(ts)] + [ts**p for p in range(1, poly + 1)]
    cols += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
    return np.column_stack(cols)


def _ols_forecast(y: np.ndarray, h: int, period: int, poly: int) -> np.ndarray:
    n = len(y)
    t_in = np.arange(n, dtype=float)
    t_out = np.arange(n, n + h, dtype=float)
    mu, sd = t_in.mean(), max(t_in.std(), 1e-9)
    X = _design(t_in, period, poly)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ts_out = (t_out - mu) / sd
    cols = [np.ones_like(ts_out)] + [ts_out**p for p in range(1, poly + 1)]
    cols += [np.sin(2 * np.pi * t_out / period), np.cos(2 * np.pi * t_out / period)]
    return np.column_stack(cols) @ beta


def f_trend_season(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    """The well-specified model for `make_series`: linear trend + one seasonal harmonic."""
    return _ols_forecast(y, h, period, poly=1)


def f_flexible(y: np.ndarray, h: int, period: int = PERIOD) -> np.ndarray:
    """Same family, degree 6. Fits the training window better and extrapolates worse."""
    return _ols_forecast(y, h, period, poly=6)


MODELS: Dict[str, Callable[..., np.ndarray]] = {
    "naive": f_naive,
    "seasonal naive": f_snaive,
    "drift": f_drift,
    "mean": f_mean,
    "trend+season": f_trend_season,
    "flexible (deg 6)": f_flexible,
}

# --------------------------------------------------------------------------- error metrics


def mae(err: np.ndarray) -> float:
    return float(np.mean(np.abs(err)))


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)) * 100.0)


def mase_scale(y_train: np.ndarray, period: int = 1) -> float:
    """In-sample one-step naive MAE - the denominator that makes MASE scale-free."""
    d = np.abs(np.diff(y_train, n=1) if period == 1 else y_train[period:] - y_train[:-period])
    return float(np.mean(d))


# --------------------------------------------------------------------------- protocols


def forecast_at(
    y: np.ndarray,
    model: Callable[..., np.ndarray],
    origin: int,
    h: int,
    period: int = PERIOD,
    train_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train on y[:origin] (or its last train_len points), predict y[origin:origin+h]."""
    train = y[:origin] if train_len is None else y[max(0, origin - train_len):origin]
    pred = model(train, h, period)
    return y[origin:origin + h], pred


def origins(n: int, h: int, k: int, min_train: int, step: int = 1) -> List[int]:
    """The k latest origins that leave h points of future, spaced `step` apart."""
    last = n - h
    o = list(range(last, min_train - 1, -step))[:k]
    return sorted(o)


def rolling_origin(
    y: np.ndarray,
    model: Callable[..., np.ndarray],
    h: int,
    k: int,
    min_train: int,
    period: int = PERIOD,
    step: int = 1,
    window: str = "expanding",
    train_len: Optional[int] = None,
) -> np.ndarray:
    """Error matrix (k origins x h horizons). Nothing is averaged here on purpose."""
    ors = origins(len(y), h, k, min_train, step)
    out = np.full((len(ors), h), np.nan)
    tl = None if window == "expanding" else (train_len if train_len is not None else min_train)
    for i, o in enumerate(ors):
        actual, pred = forecast_at(y, model, o, h, period, train_len=tl)
        out[i, :len(actual)] = actual - pred
    return out


def single_split(
    y: np.ndarray,
    model: Callable[..., np.ndarray],
    h: int,
    period: int = PERIOD,
) -> np.ndarray:
    """One origin, at the end. The protocol that ships in most notebooks."""
    actual, pred = forecast_at(y, model, len(y) - h, h, period)
    return actual - pred


# ----------------------------------------------------- supervised-table protocols (leaky)


def lag_table(y: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rows (y[i-p..i-1] -> y[i]). This shape is what makes 'just use KFold' look reasonable."""
    n = len(y)
    X = np.column_stack([y[i:n - p + i] for i in range(p)])
    return X, y[p:]


def _ridge_fit(X: np.ndarray, z: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ z)


def _ridge_pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def kfold_rmse(
    y: np.ndarray,
    p: int,
    k: int,
    rng: np.random.Generator,
    scheme: str = "random",
    embargo: int = 0,
) -> float:
    """One-step RMSE from k-fold on the lag table.

    scheme='random' shuffles rows, so a fold's training set contains rows AFTER its test
    rows and rows that overlap them by p-1 lags. scheme='blocked' keeps time order and
    drops `embargo` rows on each side of the test block.
    """
    X, z = lag_table(y, p)
    n = len(z)
    idx = np.arange(n)
    if scheme == "random":
        idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    errs: List[np.ndarray] = []
    for f in folds:
        test = np.asarray(f)
        if scheme == "random":
            train = np.setdiff1d(idx, test)
        else:
            lo, hi = test.min() - embargo, test.max() + embargo
            train = idx[(idx < lo) | (idx > hi)]
        if len(train) <= p + 2:
            continue
        beta = _ridge_fit(X[train], z[train])
        errs.append(z[test] - _ridge_pred(beta, X[test]))
    return rmse(np.concatenate(errs))


def honest_next_rmse(y_full: np.ndarray, n_train: int, p: int) -> float:
    """Fit the same lag model on the first n_train points, score the rest. No leakage."""
    X, z = lag_table(y_full[:n_train], p)
    beta = _ridge_fit(X, z)
    Xf, zf = lag_table(y_full[n_train - p:], p)
    return rmse(zf - _ridge_pred(beta, Xf))


# --------------------------------------------------------------------------- the estimand


def true_h_step_error(
    model: Callable[..., np.ndarray],
    h: int,
    n_train: int,
    reps: int,
    rng: np.random.Generator,
    metric: str = "rmse",
    period: int = PERIOD,
    **series_kw,
) -> float:
    """What a backtest is TRYING to estimate: expected error of this model at this train size.

    Computable only because the process can be re-drawn. Each rep is an independent series;
    the model is fit on its first n_train points and scored on the next h. This is the yard-
    stick every protocol below is measured against.
    """
    errs = []
    for _ in range(reps):
        y = make_series(rng, n_train + h, period=period, **series_kw)
        actual, pred = forecast_at(y, model, n_train, h, period)
        errs.append(actual - pred)
    e = np.concatenate(errs)
    return rmse(e) if metric == "rmse" else mae(e)


def effective_folds(sd_folds: float, true_sd: float) -> float:
    """k_eff: how many INDEPENDENT folds the k reported ones are worth.

    The analyst publishes sd_across_folds / sqrt(k) as the standard error of the backtest
    mean. Rolling origins share training data and overlapping test windows, so the true
    sampling sd of that mean (measurable here by re-drawing the series) is larger. The
    ratio of variances is the number of independent folds actually present. Same shape as
    Day 170's k-effective for correlated segment estimates.
    """
    if true_sd <= 0:
        return float("nan")
    return float((sd_folds / true_sd) ** 2)


def expected_min_of_m(m: int, rho: float = 0.0, grid: int = 20001, lo: float = -9.0, hi: float = 9.0) -> float:
    """E[min of m equicorrelated standard normals], by quadrature on the independent case.

    An equicorrelated field is sqrt(1-rho) * (independent field) + sqrt(rho) * (common
    shock), and the common shock has mean zero, so E[min] = sqrt(1-rho) * E[min of m iid].
    This is the arithmetic behind best-of-M selection on a shared backtest: candidates that
    all wrap the same base model are correlated, and the correlation SHRINKS the winner's
    curse by exactly sqrt(1-rho).
    """
    from math import erf, sqrt

    z = np.linspace(lo, hi, grid)
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    # E[min] = integral over z of ( -1 + (1-(1-Phi)^m) ) ... use survival form directly:
    # E[min] = -int_{-inf}^{0} P(min > z) dz + int_0^{inf} (1 - P(min <= z))... simpler:
    # E[min] = int (z) * m * (1-Phi)^(m-1) * phi dz
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    dens = m * (1.0 - Phi) ** (m - 1) * phi
    e_iid = float(np.trapezoid(z * dens, z)) if hasattr(np, "trapezoid") else float(np.trapz(z * dens, z))
    return float(sqrt(1.0 - rho) * e_iid)
