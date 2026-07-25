from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Alert:
    """One anomaly on a metric's time series."""

    date: str
    value: float
    kind: str  # threshold | zscore | trend
    severity: str  # info | warning | critical
    message: str


@dataclass
class MetricConfig:
    """Alerting rules for one metric."""

    name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    z_threshold: float = 2.5          # rolling z-score that counts as anomalous
    trend_pct: float = 0.25           # day-over-day change (fraction) that trips a trend alert
    window: int = 7                   # rolling window for baseline
    direction: str = "both"           # both | drop | spike


def _severity_from_z(z: float) -> str:
    az = abs(z)
    if az >= 4:
        return "critical"
    if az >= 3:
        return "warning"
    return "info"


def detect_anomalies(series: pd.Series, cfg: MetricConfig) -> list[Alert]:
    """Flag anomalies via three independent checks: static thresholds, rolling z-score, and
    day-over-day trend. Each check is cheap, explainable, and catches a different failure shape.

    series index = dates (str or datetime), values = the metric.
    """
    alerts: list[Alert] = []
    values = series.astype(float)
    idx = [str(i) for i in series.index]

    # baseline uses the PRIOR window only (shift(1)) so a point is never in its own baseline -
    # otherwise a single outlier hides inside the mean/std it is supposed to be measured against.
    prior = values.shift(1)
    # require a full prior window before trusting z-scores, so the metric's warm-up period
    # doesn't throw false alarms off a 2-3 point baseline.
    roll_mean = prior.rolling(cfg.window, min_periods=cfg.window).mean()
    roll_std = prior.rolling(cfg.window, min_periods=cfg.window).std(ddof=0)

    prev = values.shift(1)

    for i in range(len(values)):
        v = values.iloc[i]
        date = idx[i]

        # 1) static threshold
        if cfg.min_value is not None and v < cfg.min_value:
            alerts.append(Alert(date, v, "threshold", "critical",
                                f"{cfg.name} = {v:.0f} below floor {cfg.min_value:.0f}"))
        if cfg.max_value is not None and v > cfg.max_value:
            alerts.append(Alert(date, v, "threshold", "warning",
                                f"{cfg.name} = {v:.0f} above ceiling {cfg.max_value:.0f}"))

        # 2) rolling z-score (needs a baseline)
        mu, sd = roll_mean.iloc[i], roll_std.iloc[i]
        if pd.notna(mu) and pd.notna(sd) and sd > 0:
            z = (v - mu) / sd
            if abs(z) >= cfg.z_threshold and _direction_ok(z, cfg.direction):
                alerts.append(Alert(date, v, "zscore", _severity_from_z(z),
                                    f"{cfg.name} = {v:.0f} is {z:+.1f}σ from {cfg.window}-pt baseline ({mu:.0f})"))

        # 3) day-over-day trend
        p = prev.iloc[i]
        if pd.notna(p) and p != 0:
            change = (v - p) / abs(p)
            if abs(change) >= cfg.trend_pct and _direction_ok(change, cfg.direction):
                sev = "critical" if abs(change) >= 2 * cfg.trend_pct else "warning"
                alerts.append(Alert(date, v, "trend", sev,
                                    f"{cfg.name} moved {change:+.0%} vs prior point ({p:.0f} → {v:.0f})"))

    return _dedupe_alerts(alerts)


def _direction_ok(delta: float, direction: str) -> bool:
    if direction == "drop":
        return delta < 0
    if direction == "spike":
        return delta > 0
    return True


def _dedupe_alerts(alerts: list[Alert]) -> list[Alert]:
    """Keep the highest-severity alert per (date) but preserve distinct kinds worth seeing."""
    order = {"critical": 0, "warning": 1, "info": 2}
    seen: dict = {}
    for a in alerts:
        key = (a.date, a.kind)
        if key not in seen or order[a.severity] < order[seen[key].severity]:
            seen[key] = a
    return sorted(seen.values(), key=lambda a: (a.date, order[a.severity]))


def summarize(alerts: list[Alert]) -> dict:
    counts = {"critical": 0, "warning": 0, "info": 0}
    by_kind: dict = {}
    for a in alerts:
        counts[a.severity] += 1
        by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
    return {"total": len(alerts), "by_severity": counts, "by_kind": by_kind}


def sample_series() -> pd.Series:
    """A 30-day metric that is stable, then has a sudden drop and a spike - so detectors have prey."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-06-01", periods=30, freq="D")
    base = 1000 + rng.normal(0, 25, size=30)
    base[18] = 640    # sudden drop (incident)
    base[19] = 700
    base[25] = 1480   # spike (bot traffic?)
    return pd.Series(np.round(base), index=[d.strftime("%Y-%m-%d") for d in dates], name="daily_active_users")


SAMPLE_CONFIG = MetricConfig(
    name="daily_active_users",
    min_value=800,
    max_value=1300,
    z_threshold=2.5,
    trend_pct=0.20,
    window=7,
    direction="both",
)
