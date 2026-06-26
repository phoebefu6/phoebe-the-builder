"""Sales Forecast Dashboard — core logic.

Takes a historical sales series (a date column + a numeric value column) and
produces a forward forecast with a confidence band, plus an honest backtest
(MAPE on a held-out tail) so you know how much to trust it. The point is to get
the forecast *out of someone's spreadsheet* and into something reproducible.

Primary engine is statsmodels Holt-Winters (Exponential Smoothing) — light,
reliable, and always available. If the optional `prophet` package is installed,
`forecast_sales(..., engine="prophet")` uses it instead. No API keys either way,
so it runs standalone in a notebook or CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DATE_CANDIDATES = ("date", "ds", "day", "month", "period", "timestamp", "week")
VALUE_CANDIDATES = (
    "sales", "y", "revenue", "amount", "value", "qty", "quantity", "total",
)


@dataclass
class ForecastResult:
    engine: str
    history: pd.DataFrame          # columns: ds, y
    forecast: pd.DataFrame         # columns: ds, yhat, yhat_lower, yhat_upper
    periods: int
    freq: str
    seasonal_periods: Optional[int]
    metrics: Dict[str, float] = field(default_factory=dict)


def _guess_column(
    df: pd.DataFrame, candidates: Tuple[str, ...], parse_dates: bool
) -> Optional[str]:
    for c in df.columns:
        if c.lower() in candidates:
            return c
    # Fall back to type-based detection
    for c in df.columns:
        if parse_dates:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return c
        else:
            nums = pd.to_numeric(df[c], errors="coerce")
            if nums.notna().mean() > 0.9:
                return c
    return None


def prepare_series(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Normalize any sales table to a clean ds/y frame at an inferred frequency."""
    if date_col is None:
        date_col = _guess_column(df, DATE_CANDIDATES, parse_dates=True)
    if value_col is None:
        # Avoid picking the date col as the value col
        cand = tuple(c for c in df.columns if c != date_col)
        sub = df[list(cand)]
        value_col = _guess_column(sub, VALUE_CANDIDATES, parse_dates=False)
    if date_col is None or value_col is None:
        raise ValueError("Could not identify a date column and a numeric value column.")

    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df[date_col], errors="coerce"),
            "y": pd.to_numeric(df[value_col], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("ds").groupby("ds", as_index=False)["y"].sum()

    freq = pd.infer_freq(out["ds"]) or _fallback_freq(out["ds"])
    return out.reset_index(drop=True), freq


def _fallback_freq(ds: pd.Series) -> str:
    """Infer a coarse frequency from the median gap between dates."""
    if len(ds) < 2:
        return "D"
    gap = ds.diff().dt.days.median()
    if gap <= 1:
        return "D"
    if gap <= 7:
        return "W"
    if gap <= 31:
        return "MS"
    return "MS"


def _seasonal_periods(freq: str) -> Optional[int]:
    f = (freq or "").upper()
    if f.startswith("D"):
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    return None


def _backtest_mape(
    y: pd.Series, seasonal_periods: Optional[int], holdout: int
) -> float:
    """MAPE on the last `holdout` points, training only on what precedes them."""
    if len(y) <= holdout + 2:
        return float("nan")
    train, test = y.iloc[:-holdout], y.iloc[-holdout:]
    try:
        fitted = _fit_hw(train, seasonal_periods)
        pred = fitted.forecast(holdout)
    except Exception:
        return float("nan")
    actual = test.to_numpy()
    pred = np.asarray(pred)
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _fit_hw(y: pd.Series, seasonal_periods: Optional[int]):
    """Fit Holt-Winters, using seasonality only when there's enough data."""
    use_seasonal = seasonal_periods is not None and len(y) >= 2 * seasonal_periods
    model = ExponentialSmoothing(
        y.to_numpy(dtype=float),
        trend="add",
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_periods if use_seasonal else None,
        initialization_method="estimated",
    )
    return model.fit()


def _forecast_holtwinters(
    series: pd.DataFrame, periods: int, freq: str
) -> ForecastResult:
    sp = _seasonal_periods(freq)
    y = series["y"]
    fitted = _fit_hw(y, sp)
    yhat = np.asarray(fitted.forecast(periods))

    # Confidence band from in-sample residual spread (~95%)
    resid = y.to_numpy(dtype=float) - np.asarray(fitted.fittedvalues)
    sigma = float(np.std(resid)) if len(resid) else 0.0
    lower = yhat - 1.96 * sigma
    upper = yhat + 1.96 * sigma

    future_ds = pd.date_range(
        series["ds"].iloc[-1], periods=periods + 1, freq=freq
    )[1:]
    forecast = pd.DataFrame(
        {"ds": future_ds, "yhat": np.round(yhat, 2),
         "yhat_lower": np.round(lower, 2), "yhat_upper": np.round(upper, 2)}
    )

    holdout = min(max(periods, sp or 6), max(2, len(y) // 4))
    mape = _backtest_mape(y, sp, holdout)
    metrics = {
        "mape_pct": round(mape, 2) if mape == mape else None,  # NaN check
        "history_points": int(len(y)),
        "history_mean": round(float(y.mean()), 2),
        "forecast_mean": round(float(np.mean(yhat)), 2),
    }
    return ForecastResult(
        engine="holt-winters", history=series, forecast=forecast,
        periods=periods, freq=freq, seasonal_periods=sp, metrics=metrics,
    )


def _forecast_prophet(series: pd.DataFrame, periods: int, freq: str) -> ForecastResult:
    from prophet import Prophet  # optional dependency

    m = Prophet(interval_width=0.95)
    m.fit(series.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future).tail(periods)
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    forecast = fc[cols].round(2).reset_index(drop=True)
    sp = _seasonal_periods(freq)
    metrics = {
        "mape_pct": None,
        "history_points": int(len(series)),
        "history_mean": round(float(series["y"].mean()), 2),
        "forecast_mean": round(float(forecast["yhat"].mean()), 2),
    }
    return ForecastResult(
        engine="prophet", history=series, forecast=forecast,
        periods=periods, freq=freq, seasonal_periods=sp, metrics=metrics,
    )


def forecast_sales(
    df: pd.DataFrame,
    periods: int = 12,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    engine: str = "auto",
) -> ForecastResult:
    """Forecast `periods` steps ahead. engine: 'auto' | 'holt-winters' | 'prophet'."""
    series, freq = prepare_series(df, date_col, value_col)
    if len(series) < 4:
        raise ValueError("Need at least 4 historical points to forecast.")

    if engine == "prophet":
        return _forecast_prophet(series, periods, freq)
    return _forecast_holtwinters(series, periods, freq)


def sample_sales(periods: int = 36, random_state: int = 42) -> pd.DataFrame:
    """Deterministic monthly sales with trend + yearly seasonality + noise."""
    rng = np.random.default_rng(random_state)
    dates = pd.date_range("2022-01-01", periods=periods, freq="MS")
    t = np.arange(periods)
    trend = 10000 + 220 * t
    seasonal = 1800 * np.sin(2 * np.pi * (t % 12) / 12) + 900 * np.cos(
        2 * np.pi * (t % 12) / 12
    )
    noise = rng.normal(0, 600, size=periods)
    sales = np.round((trend + seasonal + noise).clip(min=0), 2)
    return pd.DataFrame({"date": dates, "sales": sales})
