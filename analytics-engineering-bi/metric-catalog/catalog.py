from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional


@dataclass
class MetricEntry:
    """A catalog entry: what a metric is, who owns it, and how healthy its governance is."""

    name: str
    definition: str
    owner: str = ""
    team: str = ""
    tier: int = 3                       # 1 = board/exec critical ... 3 = nice-to-have
    status: str = "active"              # active | deprecated
    depends_on: list[str] = field(default_factory=list)
    last_reviewed: str = ""             # ISO date
    sla_freshness_hours: Optional[int] = None


@dataclass
class GovernanceIssue:
    metric: str
    severity: str  # high | medium | low
    kind: str
    message: str


class MetricCatalog:
    """A registry of metrics with ownership + governance checks. Persistable to JSON."""

    def __init__(self) -> None:
        self.metrics: dict[str, MetricEntry] = {}

    def register(self, m: MetricEntry) -> None:
        self.metrics[m.name] = m

    def search(self, query: str = "", team: str = "", tier: Optional[int] = None,
               status: str = "") -> list[MetricEntry]:
        q = query.lower()
        out = []
        for m in self.metrics.values():
            if q and q not in m.name.lower() and q not in m.definition.lower():
                continue
            if team and m.team != team:
                continue
            if tier is not None and m.tier != tier:
                continue
            if status and m.status != status:
                continue
            out.append(m)
        return sorted(out, key=lambda m: (m.tier, m.name))

    # ------------------------- governance -------------------------

    def governance_issues(self, stale_days: int = 90, as_of: Optional[date] = None) -> list[GovernanceIssue]:
        """Find the gaps that make a catalog untrustworthy: no owner, stale reviews, broken deps."""
        as_of = as_of or date.today()
        issues: list[GovernanceIssue] = []
        for m in self.metrics.values():
            sev_base = "high" if m.tier == 1 else "medium" if m.tier == 2 else "low"

            if not m.owner and m.status == "active":
                issues.append(GovernanceIssue(m.name, sev_base, "no_owner",
                                              f"Tier-{m.tier} metric has no owner"))
            if m.status == "active" and m.last_reviewed:
                try:
                    reviewed = datetime.strptime(m.last_reviewed, "%Y-%m-%d").date()
                    if (as_of - reviewed).days > stale_days:
                        issues.append(GovernanceIssue(m.name, sev_base, "stale",
                                      f"Not reviewed in {(as_of - reviewed).days} days"))
                except ValueError:
                    pass
            elif m.status == "active" and not m.last_reviewed:
                issues.append(GovernanceIssue(m.name, "low", "never_reviewed", "Never reviewed"))

            for dep in m.depends_on:
                if dep not in self.metrics:
                    issues.append(GovernanceIssue(m.name, "high", "broken_dep",
                                  f"Depends on '{dep}' which is not in the catalog"))
                elif self.metrics[dep].status == "deprecated":
                    issues.append(GovernanceIssue(m.name, "high", "deprecated_dep",
                                  f"Depends on deprecated metric '{dep}'"))
        return issues

    def health(self, stale_days: int = 90, as_of: Optional[date] = None) -> dict:
        total = len(self.metrics)
        active = [m for m in self.metrics.values() if m.status == "active"]
        owned = [m for m in active if m.owner]
        issues = self.governance_issues(stale_days, as_of)
        by_tier = {}
        for m in self.metrics.values():
            by_tier[m.tier] = by_tier.get(m.tier, 0) + 1
        return {
            "total": total,
            "active": len(active),
            "deprecated": total - len(active),
            "owned_pct": round(100 * len(owned) / len(active)) if active else 100,
            "issues": len(issues),
            "high_issues": sum(1 for i in issues if i.severity == "high"),
            "by_tier": dict(sorted(by_tier.items())),
        }

    # ------------------------- persistence -------------------------

    def save(self, path: str) -> None:
        json.dump({"metrics": [asdict(m) for m in self.metrics.values()]}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricCatalog":
        cat = cls()
        for d in json.load(open(path))["metrics"]:
            cat.register(MetricEntry(**d))
        return cat


def sample_catalog() -> MetricCatalog:
    today = date.today()
    old = (today - timedelta(days=200)).strftime("%Y-%m-%d")
    recent = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    cat = MetricCatalog()
    cat.register(MetricEntry("revenue", "Net paid revenue", "finance-lead", "Finance", 1,
                             last_reviewed=recent, sla_freshness_hours=24))
    cat.register(MetricEntry("active_users", "Distinct users with an event", "growth-lead", "Growth", 1,
                             last_reviewed=recent, sla_freshness_hours=6))
    cat.register(MetricEntry("arpu", "Revenue / active users", "", "Finance", 2,
                             depends_on=["revenue", "active_users"], last_reviewed=old))
    cat.register(MetricEntry("nps", "Net promoter score", "cx-lead", "CX", 2, last_reviewed=""))
    cat.register(MetricEntry("legacy_signups", "Old signup metric", "", "Growth", 3, status="deprecated"))
    cat.register(MetricEntry("activation_rate", "Users reaching aha / signups", "growth-lead", "Growth", 2,
                             depends_on=["legacy_signups"], last_reviewed=recent))
    return cat
