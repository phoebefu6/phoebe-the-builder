from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# metric-diff — "Did this metric really change, or is it just noise?"
#
# Given a metric measured across two periods (last week vs this week,
# control vs variant, before vs after), answer three questions at once:
#   1. How big is the change?            -> absolute + relative delta
#   2. Is it real, or sampling noise?    -> a significance test + CI
#   3. How confident, in plain English?  -> a verdict a PM can read
#
# Two metric shapes cover almost every dashboard number:
#   MEAN  — a continuous metric with row-level samples (AOV, session length,
#           revenue per user). Welch's t-test (unequal variance, safe default).
#   RATE  — a proportion: successes out of trials (conversion, click, churn).
#           Two-proportion z-test.
# ---------------------------------------------------------------------------


@dataclass
class DiffResult:
    """Verdict for one period-over-period comparison."""

    metric: str
    kind: str  # "mean" | "rate"
    baseline: float
    current: float
    abs_delta: float
    rel_delta: float  # fraction, e.g. 0.12 == +12%
    p_value: float
    ci_low: float  # 95% CI on the absolute delta
    ci_high: float
    significant: bool
    direction: str  # "up" | "down" | "flat"
    n_baseline: int
    n_current: int

    @property
    def verdict(self) -> str:
        if not self.significant:
            return (
                f"No real change. {self.metric} moved {self._pct()} but the swing "
                f"is within noise (p={self.p_value:.3f}). Don't act on it yet."
            )
        arrow = "↑" if self.direction == "up" else "↓"
        return (
            f"Real change {arrow}. {self.metric} moved {self._pct()} "
            f"(p={self.p_value:.3f}, 95% CI [{self.ci_low:+.4g}, {self.ci_high:+.4g}]). "
            f"This is unlikely to be sampling noise."
        )

    def _pct(self) -> str:
        return f"{self.rel_delta * 100:+.1f}%"

    def as_row(self) -> dict:
        return {
            "metric": self.metric,
            "baseline": round(self.baseline, 4),
            "current": round(self.current, 4),
            "abs_delta": round(self.abs_delta, 4),
            "rel_delta_%": round(self.rel_delta * 100, 2),
            "p_value": round(self.p_value, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "significant": self.significant,
            "direction": self.direction,
        }


def _direction(abs_delta: float, significant: bool) -> str:
    if not significant:
        return "flat"
    return "up" if abs_delta > 0 else "down"


def diff_mean(
    metric: str,
    baseline: Sequence[float],
    current: Sequence[float],
    alpha: float = 0.05,
) -> DiffResult:
    """Compare the MEAN of a continuous metric across two periods.

    Uses Welch's t-test (does not assume equal variance) — the safe default
    for real-world period-over-period data where sample sizes and spread differ.
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    b = b[~np.isnan(b)]
    c = c[~np.isnan(c)]
    if len(b) < 2 or len(c) < 2:
        raise ValueError("mean diff needs at least 2 observations per period")

    mb, mc = float(b.mean()), float(c.mean())
    abs_delta = mc - mb
    rel_delta = abs_delta / mb if mb != 0 else math.nan

    t_res = stats.ttest_ind(c, b, equal_var=False)
    p = float(t_res.pvalue)

    # 95% CI on the difference of means (Welch-Satterthwaite df).
    vb, vc = b.var(ddof=1), c.var(ddof=1)
    se = math.sqrt(vb / len(b) + vc / len(c))
    if se == 0:
        df = len(b) + len(c) - 2
    else:
        df = (vb / len(b) + vc / len(c)) ** 2 / (
            (vb / len(b)) ** 2 / (len(b) - 1) + (vc / len(c)) ** 2 / (len(c) - 1)
        )
    tcrit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 0.0
    ci_low = abs_delta - tcrit * se
    ci_high = abs_delta + tcrit * se

    sig = p < alpha
    return DiffResult(
        metric=metric,
        kind="mean",
        baseline=mb,
        current=mc,
        abs_delta=abs_delta,
        rel_delta=rel_delta,
        p_value=p,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=sig,
        direction=_direction(abs_delta, sig),
        n_baseline=len(b),
        n_current=len(c),
    )


def diff_rate(
    metric: str,
    baseline_success: int,
    baseline_total: int,
    current_success: int,
    current_total: int,
    alpha: float = 0.05,
) -> DiffResult:
    """Compare a RATE (proportion) across two periods via a two-proportion z-test.

    Example: conversions out of visitors, churned out of active, clicks out of
    impressions. Handles the classic "5.1% vs 5.4% — did it really move?" case.
    """
    if baseline_total <= 0 or current_total <= 0:
        raise ValueError("rate diff needs positive totals in both periods")
    for name, s, n in (
        ("baseline", baseline_success, baseline_total),
        ("current", current_success, current_total),
    ):
        if s < 0 or s > n:
            raise ValueError(f"{name} successes must be within [0, total]")

    pb = baseline_success / baseline_total
    pc = current_success / current_total
    abs_delta = pc - pb
    rel_delta = abs_delta / pb if pb != 0 else math.nan

    # Pooled proportion for the test statistic.
    pool = (baseline_success + current_success) / (baseline_total + current_total)
    se_pool = math.sqrt(pool * (1 - pool) * (1 / baseline_total + 1 / current_total))
    if se_pool == 0:
        p = 1.0
    else:
        z = abs_delta / se_pool
        p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Unpooled SE for the CI on the difference (standard practice).
    se_unpool = math.sqrt(
        pb * (1 - pb) / baseline_total + pc * (1 - pc) / current_total
    )
    zcrit = stats.norm.ppf(1 - alpha / 2)
    ci_low = abs_delta - zcrit * se_unpool
    ci_high = abs_delta + zcrit * se_unpool

    sig = p < alpha
    return DiffResult(
        metric=metric,
        kind="rate",
        baseline=pb,
        current=pc,
        abs_delta=abs_delta,
        rel_delta=rel_delta,
        p_value=p,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=sig,
        direction=_direction(abs_delta, sig),
        n_baseline=baseline_total,
        n_current=current_total,
    )


# ---------------------------------------------------------------------------
# Sample data — a realistic weekly metric review so the demo runs standalone.
# ---------------------------------------------------------------------------

def sample_mean_metric(seed: int = 7) -> tuple[str, List[float], List[float]]:
    """Average order value, last week vs this week. A genuine small lift."""
    rng = np.random.default_rng(seed)
    last_week = rng.normal(48.0, 12.0, size=420).tolist()
    this_week = rng.normal(50.5, 12.0, size=445).tolist()  # ~+5% real lift
    return "Avg Order Value ($)", last_week, this_week


def sample_rate_metric() -> tuple[str, int, int, int, int]:
    """Checkout conversion. A tiny move that looks big but is really noise."""
    # 5.10% -> 5.35% on modest traffic: eyeballs say "up", stats say "wait".
    return "Checkout Conversion", 510, 10000, 535, 10000
