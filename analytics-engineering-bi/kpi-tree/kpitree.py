from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KpiTree:
    """A KPI expressed as a product of drivers, e.g. revenue = users × conversion × arpu.

    Multiplicative trees are the common shape for 'why did X move?' - and they decompose exactly.
    """

    name: str
    drivers: list[str] = field(default_factory=list)
    driver_labels: dict = field(default_factory=dict)

    def value(self, factors: dict) -> float:
        v = 1.0
        for d in self.drivers:
            v *= factors[d]
        return v

    def label(self, d: str) -> str:
        return self.driver_labels.get(d, d)


@dataclass
class Contribution:
    driver: str
    label: str
    before: float
    after: float
    pct_change: float          # driver's own % change
    contribution: float        # absolute contribution to the KPI change (LMDI, sums exactly)
    share: float               # share of total KPI change


def _log_mean(a: float, b: float) -> float:
    """Logarithmic mean L(a,b) - the exact weight that makes an LMDI decomposition add up."""
    if a <= 0 or b <= 0:
        return (a + b) / 2  # fallback for non-positive values
    if abs(a - b) < 1e-12:
        return a
    return (a - b) / (math.log(a) - math.log(b))


def decompose(tree: KpiTree, before: dict, after: dict) -> list[Contribution]:
    """Split the KPI's total change into each driver's contribution using LMDI-I.

    Contributions sum EXACTLY to the total change - so 'why did revenue move?' gets a real,
    additive, no-residual answer: this much from users, this much from price, etc.
    """
    kpi0 = tree.value(before)
    kpi1 = tree.value(after)
    weight = _log_mean(kpi1, kpi0)
    total_change = kpi1 - kpi0

    contribs: list[Contribution] = []
    for d in tree.drivers:
        x0, x1 = before[d], after[d]
        if x0 > 0 and x1 > 0:
            contrib = weight * math.log(x1 / x0)
        else:
            contrib = 0.0
        pct = (x1 - x0) / x0 if x0 else 0.0
        contribs.append(
            Contribution(
                driver=d,
                label=tree.label(d),
                before=x0,
                after=x1,
                pct_change=pct,
                contribution=contrib,
                share=contrib / total_change if total_change else 0.0,
            )
        )
    return contribs


def decomposition_summary(tree: KpiTree, before: dict, after: dict) -> dict:
    kpi0, kpi1 = tree.value(before), tree.value(after)
    contribs = decompose(tree, before, after)
    residual = (kpi1 - kpi0) - sum(c.contribution for c in contribs)
    return {
        "kpi": tree.name,
        "before": kpi0,
        "after": kpi1,
        "total_change": kpi1 - kpi0,
        "pct_change": (kpi1 - kpi0) / kpi0 if kpi0 else 0.0,
        "contributions": contribs,
        "residual": residual,   # should be ~0 for LMDI
    }


def narrate(summary: dict) -> str:
    """One-line plain-English driver story, biggest mover first."""
    contribs = sorted(summary["contributions"], key=lambda c: abs(c.contribution), reverse=True)
    dir_word = "rose" if summary["total_change"] >= 0 else "fell"
    parts = []
    for c in contribs:
        verb = "added" if c.contribution >= 0 else "removed"
        parts.append(f"{c.label} {verb} {abs(c.contribution):,.0f} ({c.pct_change:+.0%})")
    return (
        f"{summary['kpi']} {dir_word} {abs(summary['total_change']):,.0f} "
        f"({summary['pct_change']:+.0%}): " + "; ".join(parts) + "."
    )


REVENUE_TREE = KpiTree(
    name="Revenue",
    drivers=["active_users", "conversion", "arpu"],
    driver_labels={
        "active_users": "Active users",
        "conversion": "Conversion rate",
        "arpu": "ARPU",
    },
)

SAMPLE_BEFORE = {"active_users": 100_000, "conversion": 0.04, "arpu": 50.0}
SAMPLE_AFTER = {"active_users": 118_000, "conversion": 0.035, "arpu": 54.0}
