"""Pipeline SLA Monitor - core logic.

Turns raw pipeline run history into an SLA scorecard. For each pipeline you
declare four promises:

  * landing_by      - the wall-clock time the run must FINISH by (e.g. "06:00")
  * max_duration_min- the run must not take longer than this
  * freshness_hours - a fresh successful run must exist within this window of `now`
  * min_rows        - a healthy run lands at least this many rows

Every run in the history is graded against those promises. The result is a
per-pipeline compliance %, a breach log, and an overall status you can page on.

Pure standard library - no warehouse connection, runs anywhere.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SLA:
    """The promise a pipeline makes to its consumers."""

    landing_by: time  # run must finish by this time of day
    max_duration_min: int  # run must complete within this many minutes
    freshness_hours: int  # a good run must exist within this window of `now`
    min_rows: int  # a healthy run lands >= this many rows


@dataclass
class Run:
    """One observed execution of a pipeline."""

    pipeline: str
    start: datetime
    end: datetime
    rows: int
    failed: bool = False

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0


@dataclass
class Breach:
    pipeline: str
    run_date: date
    kind: str  # LATE | SLOW | FAILED | LOW_VOLUME | STALE
    detail: str


@dataclass
class PipelineReport:
    pipeline: str
    total_runs: int
    breaches: List[Breach] = field(default_factory=list)
    is_stale: bool = False

    @property
    def breach_count(self) -> int:
        return len(self.breaches)

    @property
    def compliance_pct(self) -> float:
        """Share of runs with zero breaches (staleness counts as a strike)."""
        if self.total_runs == 0:
            return 0.0
        bad_dates = {b.run_date for b in self.breaches}
        clean = self.total_runs - len(bad_dates)
        return round(100.0 * clean / self.total_runs, 1)

    @property
    def status(self) -> str:
        if self.is_stale or self.compliance_pct < 90:
            return "BREACH"
        if self.compliance_pct < 100:
            return "AT_RISK"
        return "HEALTHY"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    runs: List[Run],
    slas: Dict[str, SLA],
    now: Optional[datetime] = None,
) -> List[PipelineReport]:
    """Grade every pipeline's run history against its SLA.

    `now` is the reference point for the freshness check; defaults to the latest
    run end time in the history so the report is reproducible.
    """
    if now is None:
        now = max((r.end for r in runs), default=datetime(2026, 1, 1))

    by_pipeline: Dict[str, List[Run]] = {}
    for r in runs:
        by_pipeline.setdefault(r.pipeline, []).append(r)

    reports: List[PipelineReport] = []
    for name, sla in slas.items():
        prs = sorted(by_pipeline.get(name, []), key=lambda r: r.end)
        report = PipelineReport(pipeline=name, total_runs=len(prs))

        for run in prs:
            rd = run.end.date()
            if run.failed:
                report.breaches.append(
                    Breach(name, rd, "FAILED", "run reported failure")
                )
                continue  # a failed run can't also be late/slow in a useful way

            deadline = datetime.combine(run.end.date(), sla.landing_by)
            if run.end > deadline:
                late_by = (run.end - deadline).total_seconds() / 60.0
                report.breaches.append(
                    Breach(name, rd, "LATE", f"landed {late_by:.0f} min past {sla.landing_by:%H:%M}")
                )
            if run.duration_min > sla.max_duration_min:
                report.breaches.append(
                    Breach(name, rd, "SLOW",
                           f"ran {run.duration_min:.0f} min (limit {sla.max_duration_min})")
                )
            if run.rows < sla.min_rows:
                report.breaches.append(
                    Breach(name, rd, "LOW_VOLUME",
                           f"{run.rows:,} rows (floor {sla.min_rows:,})")
                )

        # Freshness: is there a recent SUCCESSFUL run?
        good = [r for r in prs if not r.failed]
        last_good = good[-1].end if good else None
        if last_good is None or (now - last_good) > timedelta(hours=sla.freshness_hours):
            report.is_stale = True
            age = "never" if last_good is None else f"{(now - last_good).total_seconds()/3600:.1f}h old"
            report.breaches.append(
                Breach(name, now.date(), "STALE",
                       f"no good run within {sla.freshness_hours}h (last: {age})")
            )

        reports.append(report)

    return reports


def scorecard(reports: List[PipelineReport]) -> Dict[str, float]:
    """Fleet-level summary for the top of a dashboard / page alert."""
    n = len(reports)
    if n == 0:
        return {"pipelines": 0, "healthy": 0, "at_risk": 0, "breach": 0, "fleet_compliance": 0.0}
    healthy = sum(1 for r in reports if r.status == "HEALTHY")
    at_risk = sum(1 for r in reports if r.status == "AT_RISK")
    breach = sum(1 for r in reports if r.status == "BREACH")
    fleet = round(sum(r.compliance_pct for r in reports) / n, 1)
    return {
        "pipelines": n,
        "healthy": healthy,
        "at_risk": at_risk,
        "breach": breach,
        "fleet_compliance": fleet,
    }


# ---------------------------------------------------------------------------
# Sample data - a small warehouse's nightly pipelines over ~2 weeks
# ---------------------------------------------------------------------------

SAMPLE_SLAS: Dict[str, SLA] = {
    "orders_daily": SLA(landing_by=time(6, 0), max_duration_min=45, freshness_hours=26, min_rows=8_000),
    "clickstream_daily": SLA(landing_by=time(2, 0), max_duration_min=25, freshness_hours=26, min_rows=50_000),
    "finance_close": SLA(landing_by=time(7, 0), max_duration_min=90, freshness_hours=26, min_rows=500),
    "ml_features": SLA(landing_by=time(5, 30), max_duration_min=60, freshness_hours=26, min_rows=8_000),
}


def make_sample_runs(seed: int = 42) -> List[Run]:
    """Deterministic ~2 weeks of nightly runs with realistic incidents.

    Shaped to show the full range of health at once:
      * clickstream_daily - clean          -> HEALTHY
      * orders_daily      - one late night  -> AT_RISK
      * finance_close     - repeated slow   -> BREACH
      * ml_features       - stops on day 10 -> BREACH (stale)
    """
    rng = random.Random(seed)
    runs: List[Run] = []
    start_day = date(2026, 6, 29)

    for d in range(14):
        day = start_day + timedelta(days=d)

        # orders_daily: healthy except one upstream-delay night -> AT_RISK
        start = datetime.combine(day, time(5, 0)) + timedelta(minutes=rng.randint(-5, 8))
        dur = 30 + rng.randint(-4, 6)
        rows = 10_000 + rng.randint(-800, 1500)
        if d == 5:  # upstream delay -> lands late
            start += timedelta(minutes=80)
        runs.append(Run("orders_daily", start, start + timedelta(minutes=dur), rows))

        # clickstream_daily: consistently on time, fast, full -> HEALTHY
        start = datetime.combine(day, time(0, 40)) + timedelta(minutes=rng.randint(-5, 8))
        dur = 12 + rng.randint(-2, 4)
        rows = 62_000 + rng.randint(-4_000, 8_000)
        runs.append(Run("clickstream_daily", start, start + timedelta(minutes=dur), rows))

        # finance_close: heavy, blows the duration budget several nights -> BREACH
        start = datetime.combine(day, time(5, 30)) + timedelta(minutes=rng.randint(-5, 15))
        dur = 70 + rng.randint(-8, 12)
        if d in (3, 8, 12):
            dur = 105
        runs.append(Run("finance_close", start, start + timedelta(minutes=dur), 900 + rng.randint(-100, 200)))

        # ml_features: stops updating after day 10 -> goes STALE -> BREACH
        if d <= 10:
            start = datetime.combine(day, time(4, 30)) + timedelta(minutes=rng.randint(-5, 10))
            dur = 40 + rng.randint(-5, 10)
            runs.append(Run("ml_features", start, start + timedelta(minutes=dur), 9_500 + rng.randint(-500, 800)))

    return runs
