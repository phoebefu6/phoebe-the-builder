from __future__ import annotations

"""Core logic: a dead-man's-switch for scheduled jobs.

Batch jobs fail silently - the cron line is still in place, but the script
errored and nobody noticed. Here, each job registers an expected heartbeat
interval and pings on every successful run. If a ping doesn't arrive within
`interval + grace`, the job is **overdue** and an alert should fire.

This module is pure logic (no web framework, no clock-of-record) so it's trivial
to unit-test: every function takes `now` explicitly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Job:
    name: str
    interval_seconds: float
    grace_seconds: float = 60.0
    last_ping: Optional[float] = None
    last_alerted: Optional[float] = None

    def deadline(self) -> Optional[float]:
        """Timestamp by which the next ping must arrive."""
        if self.last_ping is None:
            return None
        return self.last_ping + self.interval_seconds + self.grace_seconds

    def state(self, now: float) -> str:
        """'pending' (never pinged), 'healthy', or 'overdue'."""
        if self.last_ping is None:
            return "pending"
        return "overdue" if now > self.deadline() else "healthy"


@dataclass
class Monitor:
    jobs: Dict[str, Job] = field(default_factory=dict)

    def register(self, name: str, interval_seconds: float, grace_seconds: float = 60.0) -> Job:
        """Add or update a job. Re-registering keeps the existing last_ping."""
        existing = self.jobs.get(name)
        if existing:
            existing.interval_seconds = interval_seconds
            existing.grace_seconds = grace_seconds
            return existing
        job = Job(name=name, interval_seconds=interval_seconds, grace_seconds=grace_seconds)
        self.jobs[name] = job
        return job

    def ping(self, name: str, now: float) -> Job:
        """Record a successful run. Clears any prior alert state."""
        job = self.jobs.get(name)
        if job is None:
            # Auto-register unknown jobs with a 1h default so a stray ping is never lost.
            job = self.register(name, interval_seconds=3600.0)
        job.last_ping = now
        job.last_alerted = None
        return job

    def overdue(self, now: float) -> List[Job]:
        return [j for j in self.jobs.values() if j.state(now) == "overdue"]

    def status(self, now: float) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for job in self.jobs.values():
            rows.append(
                {
                    "name": job.name,
                    "state": job.state(now),
                    "interval_seconds": job.interval_seconds,
                    "last_ping": job.last_ping,
                    "seconds_since_ping": None if job.last_ping is None else round(now - job.last_ping, 1),
                    "deadline": job.deadline(),
                }
            )
        return rows

    def alerts_to_fire(self, now: float) -> List[Job]:
        """Overdue jobs that haven't been alerted since their last ping.

        Prevents re-alerting every check cycle: a job alerts once per outage,
        then stays quiet until it pings again.
        """
        out: List[Job] = []
        for job in self.overdue(now):
            if job.last_alerted is None or (job.last_ping is not None and job.last_alerted < job.last_ping):
                job.last_alerted = now
                out.append(job)
        return out
