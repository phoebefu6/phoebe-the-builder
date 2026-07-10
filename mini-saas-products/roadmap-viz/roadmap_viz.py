from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

STATUS_COLORS = {
    "done": "#2e9e5b",
    "in-progress": "#4361ee",
    "planned": "#b0b0b0",
    "at-risk": "#d64545",
}


@dataclass
class RoadmapItem:
    name: str
    lane: str
    start: str          # ISO date "YYYY-MM-DD"
    end: str
    status: str = "planned"

    @property
    def start_date(self) -> date:
        return datetime.strptime(self.start, "%Y-%m-%d").date()

    @property
    def end_date(self) -> date:
        return datetime.strptime(self.end, "%Y-%m-%d").date()

    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days


def to_frame(items: List[RoadmapItem]) -> pd.DataFrame:
    return pd.DataFrame([
        {"lane": it.lane, "item": it.name, "start": it.start, "end": it.end,
         "days": it.duration_days, "status": it.status}
        for it in items
    ])


def render_roadmap(
    items: List[RoadmapItem],
    today: Optional[str] = None,
    title: str = "Product Roadmap",
    save_path: Optional[str] = None,
):
    """Gantt-style timeline: one horizontal bar per item, grouped by lane, colored by status,
    with a 'today' marker. Returns the matplotlib figure."""
    if not items:
        raise ValueError("no roadmap items to render")

    # group by lane, stack rows lane by lane
    lanes: Dict[str, List[RoadmapItem]] = {}
    for it in items:
        lanes.setdefault(it.lane, []).append(it)

    rows = []  # (label, item)
    for lane, lane_items in lanes.items():
        for it in sorted(lane_items, key=lambda x: x.start_date):
            rows.append((f"{lane} · {it.name}", it))

    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.5))
    for i, (label, it) in enumerate(rows):
        y = len(rows) - i - 1
        start_num = mdates.date2num(it.start_date)
        width = max(it.duration_days, 1)
        ax.barh(y, width, left=start_num, height=0.6,
                color=STATUS_COLORS.get(it.status, "#b0b0b0"), edgecolor="white")
        ax.text(start_num + width / 2, y, it.name, ha="center", va="center",
                fontsize=7, color="white")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lbl.split(" · ")[0] for lbl, _ in reversed(rows)], fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    marker = datetime.strptime(today, "%Y-%m-%d").date() if today else date.today()
    ax.axvline(mdates.date2num(marker), color="#111", linestyle="--", linewidth=1.2)
    ax.text(mdates.date2num(marker), len(rows) - 0.3, " today", fontsize=8, color="#111")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STATUS_COLORS.values()]
    ax.legend(handles, STATUS_COLORS.keys(), loc="upper right", fontsize=8, ncol=4)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


SAMPLE_ITEMS: List[RoadmapItem] = [
    RoadmapItem("Auth revamp", "Platform", "2026-01-01", "2026-02-15", "done"),
    RoadmapItem("Mobile app", "Platform", "2026-02-15", "2026-05-30", "in-progress"),
    RoadmapItem("SSO / SAML", "Platform", "2026-06-01", "2026-07-15", "planned"),
    RoadmapItem("Onboarding flow", "Growth", "2026-01-15", "2026-03-01", "done"),
    RoadmapItem("Referral program", "Growth", "2026-03-15", "2026-05-01", "at-risk"),
    RoadmapItem("Pricing experiments", "Growth", "2026-05-15", "2026-07-01", "planned"),
    RoadmapItem("Data warehouse", "Data", "2026-02-01", "2026-04-15", "in-progress"),
    RoadmapItem("Self-serve analytics", "Data", "2026-04-15", "2026-08-01", "planned"),
]

SAMPLE_TODAY = "2026-04-01"
