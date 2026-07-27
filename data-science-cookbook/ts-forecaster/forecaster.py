from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Forecast:
    fitted: np.ndarray          # in-sample fitted values
    forecast: np.ndarray        # future point forecasts
    lower: np.ndarray           # forecast lower band
    upper: np.ndarray           # forecast upper band
    method: str = ""
    params: dict = field(default_factory=dict)


def holt_winters_add(y: np.ndarray, season: int, h: int,
                     alpha: float = 0.4, beta: float = 0.1, gamma: float = 0.3,
                     z: float = 1.96) -> Forecast:
    """Additive Holt-Winters (triple exponential smoothing), implemented in numpy.

    Captures level + trend + additive seasonality - enough to forecast most business metrics without
    Prophet. Falls back gracefully to Holt (no season) when season <= 1.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if season and season > 1 and n >= 2 * season:
        # initialize level, trend, seasonal
        level = y[:season].mean()
        trend = (y[season:2 * season].mean() - y[:season].mean()) / season
        seasonal = list(y[:season] - level)
        fitted = []
        for t in range(n):
            s = seasonal[t % season]
            if t == 0:
                fitted.append(level + s)
            prev_level = level
            level = alpha * (y[t] - seasonal[t % season]) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            seasonal[t % season] = gamma * (y[t] - level) + (1 - gamma) * seasonal[t % season]
            fitted.append(level + trend + seasonal[t % season])
        fitted = np.array(fitted[:n])
        fc = np.array([level + (i + 1) * trend + seasonal[(n + i) % season] for i in range(h)])
        method = f"Holt-Winters (additive, season={season})"
    else:
        # Holt's linear (level + trend, no season)
        level, trend = y[0], (y[1] - y[0]) if n > 1 else 0.0
        fitted = [level]
        for t in range(1, n):
            prev_level = level
            level = alpha * y[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            fitted.append(level + trend)
        fitted = np.array(fitted[:n])
        fc = np.array([level + (i + 1) * trend for i in range(h)])
        method = "Holt linear (no seasonality)"

    resid = y - fitted
    sigma = np.std(resid[~np.isnan(resid)]) if n > 2 else 0.0
    # widening bands with horizon
    band = z * sigma * np.sqrt(np.arange(1, h + 1))
    return Forecast(fitted, fc, fc - band, fc + band, method,
                    {"alpha": alpha, "beta": beta, "gamma": gamma, "season": season})


def backtest(y: np.ndarray, season: int, test_h: int, **kw) -> dict:
    """Hold out the last test_h points, forecast them, and score MAPE/RMSE - honest accuracy."""
    y = np.asarray(y, dtype=float)
    train, test = y[:-test_h], y[-test_h:]
    fc = holt_winters_add(train, season, test_h, **kw).forecast
    err = test - fc
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(test), 1e-9)) * 100)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mape": round(mape, 2), "rmse": round(rmse, 2), "test": test, "pred": fc}


def sample_series(periods: int = 48, season: int = 12, seed: int = 3) -> pd.Series:
    """Monthly metric with upward trend + yearly seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    trend = 1000 + 8 * t
    seasonal = 120 * np.sin(2 * np.pi * t / season)
    noise = rng.normal(0, 30, periods)
    idx = pd.date_range("2022-01-01", periods=periods, freq="MS")
    return pd.Series(np.round(trend + seasonal + noise), index=idx, name="metric")
