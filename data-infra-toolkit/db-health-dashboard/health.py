from __future__ import annotations

"""Core logic: collect database health metrics, score them RAG, roll up an
overall health score.

No live database is required - `simulate_metrics` produces realistic
Postgres-style telemetry so the dashboard runs anywhere. Swap it for a real
collector (psycopg `pg_stat_*` queries) in production; the scoring layer below
stays identical.
"""

import random
from typing import Dict, List, Optional

import pandas as pd

# Each metric: (label, unit, healthy_when, amber_threshold, red_threshold)
# `direction` says which way is bad: "high" = bigger is worse, "low" = smaller is worse.
METRIC_SPECS: Dict[str, Dict[str, object]] = {
    "cache_hit_ratio": {"label": "Cache hit ratio", "unit": "%", "direction": "low", "amber": 95.0, "red": 90.0},
    "index_hit_ratio": {"label": "Index hit ratio", "unit": "%", "direction": "low", "amber": 95.0, "red": 90.0},
    "conn_pool_usage": {"label": "Connection pool usage", "unit": "%", "direction": "high", "amber": 75.0, "red": 90.0},
    "avg_query_ms": {"label": "Avg query latency", "unit": "ms", "direction": "high", "amber": 100.0, "red": 250.0},
    "slow_queries": {"label": "Slow queries (>1s)", "unit": "", "direction": "high", "amber": 5.0, "red": 15.0},
    "deadlocks": {"label": "Deadlocks (last hr)", "unit": "", "direction": "high", "amber": 1.0, "red": 5.0},
    "replication_lag_s": {"label": "Replication lag", "unit": "s", "direction": "high", "amber": 5.0, "red": 30.0},
    "disk_usage": {"label": "Disk usage", "unit": "%", "direction": "high", "amber": 75.0, "red": 90.0},
    "dead_tuple_ratio": {"label": "Dead tuple ratio", "unit": "%", "direction": "high", "amber": 10.0, "red": 25.0},
    "tps": {"label": "Transactions / sec", "unit": "", "direction": "low", "amber": 50.0, "red": 20.0},
}

STATUS_SCORE = {"green": 100, "amber": 60, "red": 20}


def simulate_metrics(seed: Optional[int] = None, stressed: bool = False) -> Dict[str, float]:
    """Generate a snapshot of DB health metrics.

    Set `stressed=True` to mimic a database under load (degraded numbers) so the
    dashboard's red/amber states are easy to demo.
    """
    rng = random.Random(seed)

    def jitter(base: float, spread: float) -> float:
        return round(base + rng.uniform(-spread, spread), 2)

    if stressed:
        return {
            "cache_hit_ratio": jitter(91.0, 2.0),
            "index_hit_ratio": jitter(92.0, 2.0),
            "conn_pool_usage": jitter(88.0, 6.0),
            "avg_query_ms": jitter(210.0, 40.0),
            "slow_queries": float(rng.randint(8, 20)),
            "deadlocks": float(rng.randint(2, 7)),
            "replication_lag_s": jitter(25.0, 8.0),
            "disk_usage": jitter(86.0, 5.0),
            "dead_tuple_ratio": jitter(18.0, 6.0),
            "tps": jitter(35.0, 10.0),
        }
    return {
        "cache_hit_ratio": jitter(98.5, 1.0),
        "index_hit_ratio": jitter(98.0, 1.0),
        "conn_pool_usage": jitter(45.0, 15.0),
        "avg_query_ms": jitter(55.0, 25.0),
        "slow_queries": float(rng.randint(0, 6)),
        "deadlocks": float(rng.randint(0, 2)),
        "replication_lag_s": jitter(2.0, 2.0),
        "disk_usage": jitter(60.0, 12.0),
        "dead_tuple_ratio": jitter(6.0, 4.0),
        "tps": jitter(120.0, 40.0),
    }


def score_metric(name: str, value: float) -> str:
    """Return 'green' | 'amber' | 'red' for one metric value.

    Unknown metrics fall back to 'green' rather than crashing the dashboard -
    a new metric should never take the whole board down.
    """
    spec = METRIC_SPECS.get(name)
    if spec is None:
        return "green"
    amber = float(spec["amber"])  # type: ignore[arg-type]
    red = float(spec["red"])  # type: ignore[arg-type]

    if spec["direction"] == "high":  # bigger = worse
        if value >= red:
            return "red"
        if value >= amber:
            return "amber"
        return "green"
    # direction == "low": smaller = worse
    if value <= red:
        return "red"
    if value <= amber:
        return "amber"
    return "green"


def build_report(metrics: Dict[str, float]) -> pd.DataFrame:
    """Turn a metrics snapshot into a scored RAG table."""
    rows: List[Dict[str, object]] = []
    for name, value in metrics.items():
        spec = METRIC_SPECS.get(name, {})
        status = score_metric(name, value)
        rows.append(
            {
                "metric": spec.get("label", name),
                "value": value,
                "unit": spec.get("unit", ""),
                "status": status,
                "points": STATUS_SCORE[status],
            }
        )
    return pd.DataFrame(rows)


def overall_health(report: pd.DataFrame) -> Dict[str, object]:
    """Roll the per-metric scores into a single 0-100 health score + grade."""
    if report.empty:
        return {"score": 0, "grade": "N/A", "reds": 0, "ambers": 0}
    score = round(float(report["points"].mean()), 1)
    if score >= 90:
        grade = "Healthy"
    elif score >= 70:
        grade = "Watch"
    else:
        grade = "Critical"
    return {
        "score": score,
        "grade": grade,
        "reds": int((report["status"] == "red").sum()),
        "ambers": int((report["status"] == "amber").sum()),
    }
