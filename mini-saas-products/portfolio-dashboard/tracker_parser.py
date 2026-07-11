from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

REPO_URL = "https://github.com/phoebefu6/phoebe-the-builder"

LINE_FOLDERS: Dict[str, str] = {
    "Data Infrastructure Toolkit": "data-infra-toolkit",
    "Business Automation Suite": "automation-suite",
    "Analytics Accelerator": "analytics-accelerator",
    "Document Intelligence": "document-intelligence",
    "AI Agent Workshop": "ai-agent-workshop",
    "Mini SaaS Products": "mini-saas-products",
}

_HEADING = re.compile(r"^#+\s*Month\s+(\d+):\s*([^(\n]+?)(?:\s*\(([\w-]+)/?\))?\s*$")
_ITEM = re.compile(
    r"^- \[(?P<done>[x ])\] Day (?P<day>\d+) — (?P<slug>[\w-]+): (?P<title>[^(\n]+?)(?:\s*\((?P<date>\d{4}-\d{2}-\d{2})\))?\s*$"
)


@dataclass
class Build:
    day: int
    slug: str
    title: str
    product_line: str
    folder: str
    done: bool
    completed: Optional[date] = None

    @property
    def url(self) -> str:
        return f"{REPO_URL}/tree/main/{self.folder}/{self.slug}"


def parse_tracker(text: str) -> List[Build]:
    builds: List[Build] = []
    line_name, folder = "Unknown", "unknown"
    for raw in text.splitlines():
        heading = _HEADING.match(raw.strip())
        if heading:
            line_name = heading.group(2).strip()
            folder = heading.group(3) or LINE_FOLDERS.get(line_name, line_name.lower().replace(" ", "-"))
            continue
        item = _ITEM.match(raw.strip())
        if item:
            builds.append(Build(
                day=int(item.group("day")),
                slug=item.group("slug"),
                title=item.group("title").strip(),
                product_line=line_name,
                folder=folder,
                done=item.group("done") == "x",
                completed=date.fromisoformat(item.group("date")) if item.group("date") else None,
            ))
    return builds


def portfolio_stats(builds: List[Build]) -> Dict[str, object]:
    done = [b for b in builds if b.done and b.completed]
    if not done:
        return {"total": len(builds), "completed": 0}
    dates = sorted(b.completed for b in done)
    start, end = dates[0], dates[-1]
    per_day: Dict[date, int] = {}
    for d in dates:
        per_day[d] = per_day.get(d, 0) + 1
    busiest = max(per_day.items(), key=lambda kv: kv[1])
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 1]
    calendar_days = (end - start).days + 1
    by_line: Dict[str, Dict[str, int]] = {}
    for b in builds:
        row = by_line.setdefault(b.product_line, {"done": 0, "total": 0})
        row["total"] += 1
        row["done"] += int(b.done)
    return {
        "total": len(builds),
        "completed": len(done),
        "start": start,
        "end": end,
        "calendar_days": calendar_days,
        "builds_per_calendar_day": round(len(done) / calendar_days, 2),
        "busiest_day": busiest,
        "active_days": len(per_day),
        "longest_gap_days": max(gaps) if gaps else 0,
        "by_line": by_line,
    }


def cumulative_timeline(builds: List[Build]) -> List[Dict[str, object]]:
    """Daily cumulative completed count from first to last completion date."""
    done = sorted((b for b in builds if b.done and b.completed), key=lambda b: b.completed)
    if not done:
        return []
    start, end = done[0].completed, done[-1].completed
    out: List[Dict[str, object]] = []
    idx, total = 0, 0
    day = start
    while day <= end:
        while idx < len(done) and done[idx].completed == day:
            total += 1
            idx += 1
        out.append({"date": day, "cumulative": total, "ideal": min((day - start).days + 1, len(done))})
        day += timedelta(days=1)
    return out


def to_markdown_table(builds: List[Build]) -> str:
    lines = ["| Day | Project | Product line | Status | Link |", "|---|---|---|---|---|"]
    for b in sorted(builds, key=lambda x: x.day):
        status = f"✅ {b.completed}" if b.done else "⬜ planned"
        lines.append(f"| {b.day} | {b.title} | {b.product_line} | {status} | [{b.slug}]({b.url}) |")
    return "\n".join(lines)
