"""Sample size, power, and minimum-detectable-effect math for A/B tests.

UI-free core so it can be imported by app.py, a notebook, or a pipeline.

Three questions, one module:
  1. How many users do I need?          -> n_for_proportions / n_for_means
  2. What can I detect with what I have? -> mde_for_proportions
  3. How long will that take?            -> duration_days / plan
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------- helpers


def _z(p: float) -> float:
    """Standard normal quantile."""
    return float(stats.norm.ppf(p))


def adjusted_alpha(alpha: float, n_variants: int, correction: str = "none") -> float:
    """Alpha per comparison after correcting for multiple variants.

    n_variants counts every arm including control, so a 3-arm test makes
    2 comparisons against control.
    """
    comparisons = max(1, n_variants - 1)
    if correction == "bonferroni":
        return alpha / comparisons
    if correction == "sidak":
        return 1.0 - (1.0 - alpha) ** (1.0 / comparisons)
    return alpha


def familywise_error(alpha: float, n_variants: int) -> float:
    """Chance of at least one false winner if you run uncorrected."""
    comparisons = max(1, n_variants - 1)
    return 1.0 - (1.0 - alpha) ** comparisons


# ------------------------------------------------------ proportions (A/B)


def n_for_proportions(
    baseline: float,
    effect: float,
    effect_kind: str = "relative",
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    correction: str = "none",
) -> Dict[str, float]:
    """Users per arm to detect a lift on a conversion rate.

    baseline    : current conversion rate, e.g. 0.042
    effect      : 0.10 relative (a 10% lift) or 0.0042 absolute (0.42pp)
    effect_kind : "relative" or "absolute"

    Two-sided two-proportion z-test, equal allocation, pooled variance
    under the null.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a rate strictly between 0 and 1")
    if effect <= 0:
        raise ValueError("effect must be positive")

    if effect_kind == "relative":
        delta = baseline * effect
    elif effect_kind == "absolute":
        delta = effect
    else:
        raise ValueError("effect_kind must be 'relative' or 'absolute'")

    treated = baseline + delta
    if treated >= 1:
        raise ValueError("that lift pushes the treated rate to 100% or above")

    a = adjusted_alpha(alpha, n_variants, correction)
    pooled = (baseline + treated) / 2.0

    z_a = _z(1.0 - a / 2.0)
    z_b = _z(power)
    numerator = (
        z_a * np.sqrt(2.0 * pooled * (1.0 - pooled))
        + z_b * np.sqrt(baseline * (1.0 - baseline) + treated * (1.0 - treated))
    ) ** 2
    n_per_arm = int(np.ceil(numerator / delta**2))

    return {
        "n_per_arm": n_per_arm,
        "n_total": n_per_arm * n_variants,
        "baseline": baseline,
        "treated": treated,
        "absolute_effect": delta,
        "relative_effect": delta / baseline,
        "alpha_per_comparison": a,
        "power": power,
    }


def power_for_proportions(
    baseline: float, treated: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Achieved power for a given arm size - the inverse question."""
    delta = abs(treated - baseline)
    if delta == 0 or n_per_arm < 1:
        return 0.0
    pooled = (baseline + treated) / 2.0
    se_null = np.sqrt(2.0 * pooled * (1.0 - pooled) / n_per_arm)
    se_alt = np.sqrt(
        (baseline * (1.0 - baseline) + treated * (1.0 - treated)) / n_per_arm
    )
    crit = _z(1.0 - alpha / 2.0) * se_null
    return float(stats.norm.sf((crit - delta) / se_alt))


def mde_for_proportions(
    baseline: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    correction: str = "none",
) -> Dict[str, float]:
    """The honest question: given the traffic I actually have, what is the
    smallest lift I could detect?

    Solved by bisection on n_for_proportions rather than the closed-form
    approximation, so the answer is consistent with the sample-size math.
    """
    a = adjusted_alpha(alpha, n_variants, correction)

    def shortfall(rel: float) -> float:
        needed = n_for_proportions(
            baseline, rel, "relative", a, power, n_variants=2, correction="none"
        )["n_per_arm"]
        return needed - n_per_arm

    lo, hi = 1e-5, 0.01
    # grow the bracket until the required n drops below what we have
    while shortfall(hi) > 0:
        hi *= 2.0
        if baseline * (1.0 + hi) >= 1.0 or hi > 50.0:
            return {
                "relative_mde": float("nan"),
                "absolute_mde": float("nan"),
                "treated": float("nan"),
                "n_per_arm": n_per_arm,
                "detectable": False,
            }

    rel = float(optimize.brentq(shortfall, lo, hi, xtol=1e-6))
    return {
        "relative_mde": rel,
        "absolute_mde": baseline * rel,
        "treated": baseline * (1.0 + rel),
        "n_per_arm": n_per_arm,
        "detectable": True,
    }


# ------------------------------------------------------------------ means


def n_for_means(
    sigma: float,
    delta: float,
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    correction: str = "none",
) -> Dict[str, float]:
    """Users per arm to detect a shift in a continuous metric (AOV, session
    length, revenue per user).

    Exact two-sample t-test power via the noncentral t distribution, solved
    iteratively - the normal approximation understates n for small samples.
    """
    if sigma <= 0 or delta <= 0:
        raise ValueError("sigma and delta must be positive")
    a = adjusted_alpha(alpha, n_variants, correction)
    effect_size = delta / sigma  # Cohen's d

    def achieved(n: float) -> float:
        n = max(2.0, n)
        df = 2.0 * n - 2.0
        ncp = effect_size * np.sqrt(n / 2.0)
        crit = stats.t.ppf(1.0 - a / 2.0, df)
        return float(
            stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
        )

    n = 2.0
    while achieved(n) < power and n < 5e7:
        n *= 1.5
    lo, hi = max(2.0, n / 1.5), n
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if achieved(mid) < power:
            lo = mid
        else:
            hi = mid
    n_per_arm = int(np.ceil(hi))

    return {
        "n_per_arm": n_per_arm,
        "n_total": n_per_arm * n_variants,
        "effect_size_d": effect_size,
        "achieved_power": achieved(n_per_arm),
        "alpha_per_comparison": a,
        "power": power,
    }


# --------------------------------------------------------------- duration


def duration_days(
    n_total: int, daily_eligible: float, exposure_share: float = 1.0
) -> float:
    """Calendar days to collect n_total, given daily eligible users and the
    fraction of them you are willing to put in the test."""
    per_day = daily_eligible * exposure_share
    if per_day <= 0:
        return float("inf")
    return n_total / per_day


# ------------------------------------------------------------------- plan


def plan(
    baseline: float,
    effect: float,
    daily_eligible: float,
    effect_kind: str = "relative",
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    correction: str = "none",
    exposure_share: float = 1.0,
    max_days: int = 28,
) -> Dict[str, object]:
    """Full test plan: size, duration, what you could detect in max_days,
    and the warnings that stop a doomed test before it starts."""
    size = n_for_proportions(
        baseline, effect, effect_kind, alpha, power, n_variants, correction
    )
    days = duration_days(int(size["n_total"]), daily_eligible, exposure_share)

    budget_total = int(np.floor(daily_eligible * exposure_share * max_days))
    budget_per_arm = max(1, budget_total // n_variants)
    reachable = mde_for_proportions(
        baseline, budget_per_arm, alpha, power, n_variants, correction
    )

    warnings: List[str] = []
    if days > max_days:
        warnings.append(
            f"Needs {days:.0f} days but the window is {max_days}. "
            f"In {max_days} days at this traffic you can only detect a "
            f"{reachable['relative_mde'] * 100:.1f}% lift, not "
            f"{size['relative_effect'] * 100:.1f}%. Widen the effect you are "
            f"hunting, raise exposure, or pick a higher-traffic surface."
        )
    if days < 7:
        warnings.append(
            f"Only {days:.1f} days of data. Run at least one full week "
            "anyway - weekday and weekend users are different people."
        )
    if baseline < 0.01:
        warnings.append(
            f"Baseline is {baseline * 100:.2f}%. Rare events need enormous "
            "samples and the normal approximation gets shaky. Consider a "
            "coarser upstream metric as the primary."
        )
    if n_variants > 2 and correction == "none":
        fw = familywise_error(alpha, n_variants)
        warnings.append(
            f"{n_variants} arms, {n_variants - 1} comparisons, no correction: "
            f"{fw * 100:.1f}% chance of at least one false winner (you think "
            f"it is {alpha * 100:.0f}%). Apply Bonferroni or Sidak."
        )
    if exposure_share < 1.0:
        warnings.append(
            f"Only {exposure_share * 100:.0f}% of traffic is exposed, which is "
            "what stretches this to {:.0f} days.".format(days)
        )
    warnings.append(
        "This number assumes you look once, at the end. Peeking daily and "
        "stopping on the first significant result inflates your false "
        "positive rate well past alpha. Commit to the horizon or use a "
        "sequential test."
    )

    return {
        "size": size,
        "days": days,
        "days_rounded": int(np.ceil(days)),
        "feasible": bool(days <= max_days),
        "reachable_in_window": reachable,
        "window_days": max_days,
        "budget_per_arm": budget_per_arm,
        "warnings": warnings,
    }


def sensitivity_table(
    baseline: float,
    daily_eligible: float,
    relative_effects: Optional[List[float]] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
    correction: str = "none",
    exposure_share: float = 1.0,
) -> List[Dict[str, float]]:
    """How the ask scales with the lift you claim to be hunting - the table
    that shows why 'let's detect a 2% lift' is not a small request."""
    if relative_effects is None:
        relative_effects = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    rows = []
    for rel in relative_effects:
        s = n_for_proportions(
            baseline, rel, "relative", alpha, power, n_variants, correction
        )
        rows.append(
            {
                "relative_lift": rel,
                "treated_rate": s["treated"],
                "n_per_arm": s["n_per_arm"],
                "n_total": s["n_total"],
                "days": duration_days(
                    int(s["n_total"]), daily_eligible, exposure_share
                ),
            }
        )
    return rows
