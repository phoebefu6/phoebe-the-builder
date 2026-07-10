from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# status thresholds: gap = expected_progress - actual_progress
AT_RISK_GAP = 0.10
OFF_TRACK_GAP = 0.25


@dataclass
class KeyResult:
    name: str
    start: float
    current: float
    target: float
    unit: str = ""

    @property
    def progress(self) -> float:
        """Fraction 0-1 of the way from start to target (handles decreasing targets too)."""
        span = self.target - self.start
        if span == 0:
            return 1.0
        return max(0.0, min(1.0, (self.current - self.start) / span))

    def status(self, time_elapsed: float) -> str:
        gap = time_elapsed - self.progress
        if self.progress >= 1.0:
            return "done"
        if gap >= OFF_TRACK_GAP:
            return "off-track"
        if gap >= AT_RISK_GAP:
            return "at-risk"
        return "on-track"

    def fmt(self, v: float) -> str:
        if self.unit == "$":
            return f"${v:,.0f}"
        if self.unit == "%":
            return f"{v:.0f}%"
        return f"{v:,.0f}{self.unit}"


@dataclass
class Objective:
    name: str
    key_results: List[KeyResult] = field(default_factory=list)

    @property
    def progress(self) -> float:
        if not self.key_results:
            return 0.0
        return sum(kr.progress for kr in self.key_results) / len(self.key_results)

    def status(self, time_elapsed: float) -> str:
        statuses = [kr.status(time_elapsed) for kr in self.key_results]
        if all(s == "done" for s in statuses):
            return "done"
        if "off-track" in statuses:
            return "off-track"
        if "at-risk" in statuses:
            return "at-risk"
        return "on-track"


@dataclass
class Advice:
    objective: str
    key_result: str
    status: str
    message: str


def advise(objectives: List[Objective], time_elapsed: float) -> List[Advice]:
    """Flag KRs behind pace. time_elapsed = fraction of the period gone (0-1)."""
    out: List[Advice] = []
    for obj in objectives:
        for kr in obj.key_results:
            st = kr.status(time_elapsed)
            if st in ("at-risk", "off-track"):
                gap = time_elapsed - kr.progress
                out.append(Advice(
                    objective=obj.name, key_result=kr.name, status=st,
                    message=(
                        f"{kr.progress:.0%} done with {time_elapsed:.0%} of the period elapsed "
                        f"({gap:+.0%} behind pace). "
                        + ("Escalate and re-plan — this KR likely misses." if st == "off-track"
                           else "Add focus this week to close the gap.")
                    ),
                ))
    # worst first
    out.sort(key=lambda a: 0 if a.status == "off-track" else 1)
    return out


def quarter_summary(objectives: List[Objective], time_elapsed: float) -> dict:
    krs = [kr for o in objectives for kr in o.key_results]
    from collections import Counter
    counts = Counter(kr.status(time_elapsed) for kr in krs)
    return {
        "objectives": len(objectives),
        "key_results": len(krs),
        "avg_progress": round(sum(kr.progress for kr in krs) / len(krs), 3) if krs else 0.0,
        "on_track": counts.get("on-track", 0) + counts.get("done", 0),
        "at_risk": counts.get("at-risk", 0),
        "off_track": counts.get("off-track", 0),
    }


SAMPLE_OBJECTIVES: List[Objective] = [
    Objective("Grow revenue", [
        KeyResult("MRR", start=80_000, current=104_000, target=120_000, unit="$"),
        KeyResult("Enterprise logos", start=5, current=7, target=12),
    ]),
    Objective("Delight users", [
        KeyResult("NPS", start=32, current=41, target=50),
        KeyResult("Churn rate", start=6.0, current=5.2, target=3.0, unit="%"),
    ]),
    Objective("Ship the platform", [
        KeyResult("Features GA", start=0, current=2, target=8),
        KeyResult("p95 latency", start=800, current=600, target=300, unit="ms"),
    ]),
]

SAMPLE_TIME_ELAPSED = 0.6  # 60% through the quarter
