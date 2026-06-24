from __future__ import annotations

"""Core logic: turn a tidy metric time series + targets into a KPI scorecard.

Execs ask for the same numbers every Monday - "where's revenue, signups, churn
vs target?". Analysts rebuild the same pull by hand each week. This module takes
a long-format frame (date, metric, value) plus a target config and returns, per
metric: the latest value, week-over-week and month-over-month deltas, trend
direction, status vs target, and a RAG (red/amber/green) band - the exact answer
the Monday email needs.

Pure pandas, no UI - shared by the Streamlit app and mountable as a "KPI Tracker"
app on the platform shell.
"""

from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# How close to target still counts as "amber" rather than "red".
AMBER_BAND = 0.10  # within 10% of target = amber, beyond = red


def _pct_change(curr: float, prev: Optional[float]) -> Optional[float]:
    """Percent change curr vs prev; None if prev missing or zero."""
    if prev is None or prev == 0 or pd.isna(prev):
        return None
    return round(100 * (curr - prev) / abs(prev), 1)


def _value_on_or_before(s: pd.Series, asof: pd.Timestamp) -> Optional[float]:
    """Latest value at or before `asof` in a date-indexed series."""
    window = s[s.index <= asof]
    if window.empty:
        return None
    return float(window.iloc[-1])


def _rag(value: float, target: float, direction: str) -> str:
    """RAG band given a value, its target, and whether higher/lower is better."""
    if direction == "down":  # lower is better (e.g. churn) - flip the comparison
        if value <= target:
            return "green"
        return "amber" if value <= target * (1 + AMBER_BAND) else "red"
    # "up": higher is better (e.g. revenue)
    if value >= target:
        return "green"
    return "amber" if value >= target * (1 - AMBER_BAND) else "red"


def score_metric(
    df: pd.DataFrame,
    metric: str,
    target: Optional[float] = None,
    direction: str = "up",
) -> Dict[str, Any]:
    """Score one metric from a long frame [date, metric, value].

    `direction` = "up" (higher is better) or "down" (lower is better).
    """
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"metric not found: {metric!r}")
    sub["date"] = pd.to_datetime(sub["date"])
    s = sub.sort_values("date").set_index("date")["value"].astype(float)

    asof = s.index.max()
    latest = float(s.iloc[-1])
    prev_week = _value_on_or_before(s, asof - pd.Timedelta(days=7))
    prev_month = _value_on_or_before(s, asof - pd.Timedelta(days=30))

    wow = _pct_change(latest, prev_week)
    mom = _pct_change(latest, prev_month)

    # Trend over the trailing window: sign of the simple slope.
    tail = s.tail(8)
    if len(tail) >= 2:
        slope = np.polyfit(range(len(tail)), tail.values, 1)[0]
        trend = "up" if slope > 0 else "down" if slope < 0 else "flat"
    else:
        trend = "flat"

    out: Dict[str, Any] = {
        "metric": metric,
        "asof": asof.date().isoformat(),
        "latest": round(latest, 2),
        "wow_pct": wow,
        "mom_pct": mom,
        "trend": trend,
        "direction": direction,
        "target": target,
    }

    if target is not None:
        out["target_pct"] = round(100 * latest / target, 1) if target else None
        out["rag"] = _rag(latest, target, direction)
        out["on_target"] = out["rag"] == "green"
    else:
        out["rag"] = "none"
        out["on_target"] = None
    return out


def build_scorecard(
    df: pd.DataFrame,
    targets: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Score every metric in `targets`.

    `targets` maps metric -> {"target": float, "direction": "up"|"down"}.
    Metrics present in df but absent from targets are still scored (no RAG).
    """
    rows: List[Dict[str, Any]] = []
    metrics = list(targets.keys()) or sorted(df["metric"].unique())
    for m in metrics:
        cfg = targets.get(m, {})
        rows.append(
            score_metric(df, m, cfg.get("target"), cfg.get("direction", "up"))
        )
    return rows


def scorecard_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count metrics by RAG band - the one-line health read for the exec note."""
    summary = {"green": 0, "amber": 0, "red": 0, "none": 0}
    for r in rows:
        summary[r.get("rag", "none")] += 1
    return summary
