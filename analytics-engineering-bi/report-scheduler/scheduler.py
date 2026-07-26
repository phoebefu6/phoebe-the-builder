from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


@dataclass
class ScheduledReport:
    """A report that should be sent on a schedule - the thing someone currently does by hand."""

    name: str
    cron: str                       # 5-field cron: min hour dom month dow
    recipients: list[str] = field(default_factory=list)
    metric: str = ""                # column to summarize
    group_by: Optional[str] = None
    agg: str = "sum"


# --------------------------- minimal cron ---------------------------

_DOW = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}


def _parse_field(field_str: str, lo: int, hi: int) -> set:
    """Parse one cron field into the set of matching integers. Supports *, */n, a-b, and lists."""
    values: set = set()
    for part in field_str.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, step_s = part.split("/")
            step = int(step_s)
        if part in ("*", ""):
            rng = range(lo, hi + 1)
        elif "-" in part:
            a, b = part.split("-")
            rng = range(int(a), int(b) + 1)
        else:
            rng = range(int(part), int(part) + 1)
        values.update(v for v in rng if (v - lo) % step == 0)
    return values


def _matches(dt: datetime, cron: str) -> bool:
    minute, hour, dom, month, dow = cron.split()
    dow_norm = ",".join(str(_DOW.get(p.lower(), p)) for p in dow.split(","))
    return (
        dt.minute in _parse_field(minute, 0, 59)
        and dt.hour in _parse_field(hour, 0, 23)
        and dt.day in _parse_field(dom, 1, 31)
        and dt.month in _parse_field(month, 1, 12)
        and (dt.weekday() + 1) % 7 in _parse_field(dow_norm, 0, 6)  # python Mon=0 -> cron Sun=0
    )


def next_runs(cron: str, start: datetime, count: int = 5) -> list[datetime]:
    """Return the next `count` datetimes a cron expression fires, scanning minute by minute."""
    runs: list[datetime] = []
    dt = start.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = 366 * 24 * 60  # cap the scan at ~1 year of minutes
    steps = 0
    while len(runs) < count and steps < limit:
        if _matches(dt, cron):
            runs.append(dt)
        dt += timedelta(minutes=1)
        steps += 1
    return runs


def describe_cron(cron: str) -> str:
    """Plain-English gloss for common cron shapes - so a non-engineer trusts the schedule."""
    common = {
        "0 9 * * 1": "Every Monday at 09:00",
        "0 9 * * *": "Every day at 09:00",
        "0 8 1 * *": "1st of each month at 08:00",
        "0 17 * * 5": "Every Friday at 17:00",
        "*/15 * * * *": "Every 15 minutes",
    }
    return common.get(cron.strip(), f"cron: {cron}")


# --------------------------- report rendering ---------------------------

def render_report(report: ScheduledReport, df: pd.DataFrame, as_of: Optional[datetime] = None) -> str:
    """Render the report body as Markdown - the digest a human would otherwise assemble."""
    as_of = as_of or datetime.now()
    lines = [f"# {report.name}", "", f"_Generated {as_of:%Y-%m-%d %H:%M} · schedule: {describe_cron(report.cron)}_", ""]

    if report.metric and report.metric in df.columns:
        total = df[report.metric].sum()
        lines.append(f"**Total {report.metric}:** {total:,.0f}")
        lines.append("")
        if report.group_by and report.group_by in df.columns:
            agg = getattr(df.groupby(report.group_by)[report.metric], report.agg)().sort_values(ascending=False)
            lines.append(f"## {report.metric} by {report.group_by}")
            lines.append("")
            lines.append(f"| {report.group_by} | {report.agg}({report.metric}) |")
            lines.append("|---|---|")
            for k, v in agg.items():
                lines.append(f"| {k} | {v:,.0f} |")
    else:
        lines.append(f"_(no metric configured, {len(df)} rows in source)_")

    lines += ["", "---", f"_To: {', '.join(report.recipients) or '(no recipients)'}_"]
    return "\n".join(lines)


def build_send_plan(reports: list[ScheduledReport], start: datetime, horizon_days: int = 7) -> pd.DataFrame:
    """Preview every send that WOULD happen in the next N days. A dry run - it never sends email."""
    rows = []
    end = start + timedelta(days=horizon_days)
    for r in reports:
        for run in next_runs(r.cron, start, count=50):
            if run > end:
                break
            rows.append({"report": r.name, "fires_at": run.strftime("%Y-%m-%d %H:%M"),
                         "recipients": len(r.recipients), "schedule": describe_cron(r.cron)})
    return pd.DataFrame(rows).sort_values("fires_at").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["report", "fires_at", "recipients", "schedule"]
    )


SAMPLE_REPORTS = [
    ScheduledReport("Weekly Revenue Digest", "0 9 * * 1", ["exec@acme.com", "finance@acme.com"],
                    metric="amount", group_by="region", agg="sum"),
    ScheduledReport("Daily Signups", "0 9 * * *", ["growth@acme.com"], metric="signups", group_by="channel", agg="sum"),
    ScheduledReport("Monthly Board Pack", "0 8 1 * *", ["board@acme.com"], metric="amount", group_by="plan", agg="sum"),
]


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["US", "EU", "APAC", "US", "EU", "APAC"],
            "plan": ["pro", "free", "team", "team", "pro", "free"],
            "channel": ["paid", "organic", "referral", "paid", "organic", "paid"],
            "amount": [12000, 8000, 5000, 9000, 7000, 3000],
            "signups": [40, 30, 20, 25, 35, 15],
        }
    )
