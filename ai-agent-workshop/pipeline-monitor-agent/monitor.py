from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class PipelineRun:
    job_name: str
    status: str  # "success" | "failed"
    start_time: datetime
    duration_seconds: float
    rows_processed: Optional[int] = None
    expected_interval_minutes: int = 60


@dataclass
class Alert:
    job_name: str
    severity: str  # critical | warning | info
    reason: str
    recommended_action: str
    run_time: datetime


def _median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def _rule_based_action(reason: str) -> str:
    lowered = reason.lower()
    if "failed at" in lowered:
        return "check job logs for stack trace, re-run after fixing root cause"
    if "median of" in lowered and "took" in lowered:
        return "check upstream data volume and warehouse load, consider scaling compute"
    if "hasn't succeeded" in lowered or "no recorded successful run" in lowered:
        return "check scheduler/orchestrator, confirm job wasn't silently disabled or stuck"
    if "under half" in lowered:
        return "check upstream source for partial data or an empty extract"
    return "investigate manually"


def _call_claude_action(reason: str, job_name: str) -> Optional[str]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": f"One-sentence recommended action for a data engineer, given pipeline '{job_name}' alert: {reason}",
            }],
        )
        return response.content[0].text.strip()
    except Exception:
        return None


def check_run(run: PipelineRun, history: List[PipelineRun]) -> Optional[Alert]:
    """Evaluate a single run against its own job's history and return an Alert if something's wrong."""
    job_history = [r for r in history if r.job_name == run.job_name and r.start_time < run.start_time]

    if run.status == "failed":
        reason = f"job '{run.job_name}' failed at {run.start_time.isoformat()}"
        severity = "critical"
    elif job_history:
        durations = [r.duration_seconds for r in job_history if r.status == "success"]
        median_duration = _median(durations)
        if median_duration and run.duration_seconds > median_duration * 2:
            reason = f"job '{run.job_name}' took {run.duration_seconds:.0f}s, over 2x the historical median of {median_duration:.0f}s"
            severity = "warning"
        elif run.rows_processed is not None:
            row_counts = [r.rows_processed for r in job_history if r.rows_processed is not None]
            median_rows = _median(row_counts) if row_counts else None
            if median_rows and run.rows_processed < median_rows * 0.5:
                reason = f"job '{run.job_name}' processed {run.rows_processed} rows, under half the historical median of {median_rows:.0f}"
                severity = "warning"
            else:
                return None
        else:
            return None
    else:
        return None

    action = _call_claude_action(reason, run.job_name) or _rule_based_action(reason)
    return Alert(job_name=run.job_name, severity=severity, reason=reason, recommended_action=action, run_time=run.start_time)


def check_staleness(job_name: str, last_success: Optional[datetime], expected_interval_minutes: int, now: Optional[datetime] = None) -> Optional[Alert]:
    """Flag a job that hasn't succeeded within 2x its expected interval — the 'silent failure' case."""
    now = now or datetime.now()
    if last_success is None:
        reason = f"job '{job_name}' has no recorded successful run"
        severity = "critical"
    elif now - last_success > timedelta(minutes=expected_interval_minutes * 2):
        overdue_minutes = (now - last_success).total_seconds() / 60
        reason = f"job '{job_name}' hasn't succeeded in {overdue_minutes:.0f} min, expected every {expected_interval_minutes} min"
        severity = "critical"
    else:
        return None

    action = _call_claude_action(reason, job_name) or _rule_based_action(reason)
    return Alert(job_name=job_name, severity=severity, reason=reason, recommended_action=action, run_time=now)


def monitor_runs(runs: List[PipelineRun], now: Optional[datetime] = None) -> List[Alert]:
    """Agent loop: evaluate every run for failure/anomaly, then check each job for staleness."""
    runs_sorted = sorted(runs, key=lambda r: r.start_time)
    alerts: List[Alert] = []

    for i, run in enumerate(runs_sorted):
        history = runs_sorted[:i]
        alert = check_run(run, history)
        if alert:
            alerts.append(alert)

    job_names = {r.job_name for r in runs}
    for job_name in job_names:
        job_runs = [r for r in runs_sorted if r.job_name == job_name]
        successes = [r for r in job_runs if r.status == "success"]
        last_success = max((r.start_time for r in successes), default=None)
        interval = job_runs[-1].expected_interval_minutes if job_runs else 60
        stale_alert = check_staleness(job_name, last_success, interval, now=now)
        if stale_alert:
            alerts.append(stale_alert)

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(alerts, key=lambda a: (severity_order[a.severity], a.run_time))
