from __future__ import annotations

"""Core logic: A/B test significance, the right way.

Teams eyeball "variant looks higher, ship it" and fool themselves with noise. This
module runs a proper **two-proportion z-test**: conversion rates, lift, a p-value, a
confidence interval on the difference, and a clear significant / not-significant
verdict. Plus a sample-size calculator so you know how much traffic you actually need.

Pure functions (scipy + math), no UI - shared by the Streamlit app and mountable as
an "A/B Test" app on the platform shell.
"""

import math
from dataclasses import dataclass
from typing import Dict

from scipy import stats


@dataclass
class ABResult:
    control_rate: float
    variant_rate: float
    abs_diff: float          # variant - control (proportion points)
    rel_lift: float          # relative lift, %
    z: float
    p_value: float           # two-sided
    significant: bool
    ci_low: float            # CI on the absolute difference
    ci_high: float
    alpha: float
    winner: str              # "variant" | "control" | "no difference"


def run_ab_test(control_n: int, control_conv: int,
                variant_n: int, variant_conv: int, alpha: float = 0.05) -> ABResult:
    """Two-proportion z-test with a CI on the difference.

    Raises ValueError on impossible inputs (zero sample, conversions > visitors).
    """
    if control_n <= 0 or variant_n <= 0:
        raise ValueError("sample sizes must be positive")
    if not (0 <= control_conv <= control_n) or not (0 <= variant_conv <= variant_n):
        raise ValueError("conversions must be between 0 and visitors")

    p1 = control_conv / control_n
    p2 = variant_conv / variant_n
    abs_diff = p2 - p1
    rel_lift = (abs_diff / p1 * 100) if p1 > 0 else float("inf")

    # Pooled standard error for the hypothesis test.
    pooled = (control_conv + variant_conv) / (control_n + variant_n)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / control_n + 1 / variant_n))
    z = abs_diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval on the difference.
    se_diff = math.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / variant_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = abs_diff - z_crit * se_diff
    ci_high = abs_diff + z_crit * se_diff

    significant = p_value < alpha
    if not significant:
        winner = "no difference"
    else:
        winner = "variant" if abs_diff > 0 else "control"

    return ABResult(
        control_rate=p1, variant_rate=p2, abs_diff=abs_diff, rel_lift=rel_lift,
        z=z, p_value=p_value, significant=significant,
        ci_low=ci_low, ci_high=ci_high, alpha=alpha, winner=winner,
    )


def required_sample_size(baseline_rate: float, mde_abs: float,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Visitors needed PER variant to detect an absolute lift `mde_abs`.

    baseline_rate - current conversion rate (0-1)
    mde_abs       - minimum detectable effect, absolute (e.g. 0.02 = +2pp)
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1")
    if mde_abs <= 0:
        raise ValueError("mde must be positive")
    p1 = baseline_rate
    p2 = min(max(baseline_rate + mde_abs, 1e-9), 1 - 1e-9)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pbar * (1 - pbar))
                 + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    n = numerator / (mde_abs ** 2)
    return int(math.ceil(n))


def as_dict(r: ABResult) -> Dict[str, object]:
    return {
        "control_rate": round(r.control_rate, 4),
        "variant_rate": round(r.variant_rate, 4),
        "abs_diff_pp": round(r.abs_diff * 100, 2),
        "rel_lift_pct": round(r.rel_lift, 2),
        "z": round(r.z, 3),
        "p_value": round(r.p_value, 4),
        "significant": r.significant,
        "ci_pp": [round(r.ci_low * 100, 2), round(r.ci_high * 100, 2)],
        "winner": r.winner,
    }
