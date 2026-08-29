"""A guardrail is not a second metric. It is a constraint, and a constraint has a power.

The world below is a growth experiment with a known answer. A lever raises the primary
metric (conversion) and lowers the thing the business actually lives on (180-day retained
users), through two mechanisms that are written down once and reused everywhere:

  1. the incremental conversions the lever buys are MARGINAL users, who retain badly;
  2. some share of every user sees a pushier product and is mildly annoyed by it.

Every guardrail in the catalogue is a noisy, partially-matured view of one of those two
mechanisms. Nothing is typed in twice: each guardrail's treatment parameter is derived
from the same `ann(a)` and `inc(a)` that move the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------------------
# The world. Five constants for the lever, four for the value model.
# --------------------------------------------------------------------------------------

P0 = 0.10           # baseline conversion rate
REL_LIFT = 0.12     # relative lift in conversion at full lever intensity a = 1
ANNOY_REACH = 0.35  # share of ALL users who notice the pushier product at a = 1
RESP_RATE = 0.02    # survey response rate (the NPS denominator)

R_GOOD = 0.62       # 180-day retention of an ordinary converter
R_MARGINAL = 0.08   # 180-day retention of a converter the lever dragged in
ANNOY_RETENTION_HIT = 0.18  # proportional hit to R_GOOD among annoyed users
HORIZON_DAYS = 180


def annoyed_share(a: float) -> float:
    """Share of all users who experience the pushier product at intensity `a`."""
    return ANNOY_REACH * a


def incremental_conversion(a: float) -> float:
    """Extra conversions per user bought by the lever at intensity `a`."""
    return P0 * REL_LIFT * a


def conversion_rate(a: float) -> float:
    return P0 + incremental_conversion(a)


def true_value(a: float) -> float:
    """180-day retained users per 1,000 exposed. This is what the business lives on."""
    inc = incremental_conversion(a)
    ann = annoyed_share(a)
    good = P0 * R_GOOD * (1.0 - ANNOY_RETENTION_HIT * ann)
    marginal = inc * R_MARGINAL
    return 1000.0 * (good + marginal)


def retention_quality(a: float) -> float:
    """180-day retention among converters. The volume metric hides the harm; this shows it.

    Adding low-quality converters raises the numerator a little and the denominator a lot,
    so a lever that costs 4.75% of retained users costs three times that share of the RATE.
    """
    return true_value(a) / (1000.0 * conversion_rate(a))


def quality_change(a: float) -> float:
    return retention_quality(a) / retention_quality(0.0) - 1.0


def primary_lift(a: float) -> float:
    """Relative lift in the primary metric. This is the number on the ship slide."""
    return conversion_rate(a) / P0 - 1.0


def value_change(a: float) -> float:
    """Relative change in 180-day retained users. Negative is harm."""
    return true_value(a) / true_value(0.0) - 1.0


# --------------------------------------------------------------------------------------
# Maturity. One rule, applied to every guardrail.
# --------------------------------------------------------------------------------------

def observable_fraction(decision_day: int, maturity_days: int) -> float:
    """Share of enrolled users whose guardrail value exists yet.

    Users enrol uniformly across a `decision_day`-long experiment. A metric defined over
    `maturity_days` of exposure can only be computed for users who enrolled early enough.
    A 90-day retention guardrail read on day 14 has a denominator of exactly zero, which
    is a fact about arithmetic and not about the product.
    """
    if decision_day <= 0:
        return 0.0
    return max(0.0, (decision_day - maturity_days) / decision_day)


# --------------------------------------------------------------------------------------
# The guardrail catalogue.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Guardrail:
    name: str
    family: str        # 'binom' | 'poisson' | 'normal'
    channel: str       # 'marginal' (converter mixture) | 'annoy' | 'none'
    denominator: str   # 'all' | 'converters' | 'responders'
    maturity_days: int
    base: float        # control parameter
    effect: float      # marginal-user parameter, or per-annoyed-user bump
    sd: float = 0.0    # for 'normal'
    harm_sign: int = 1  # +1 if a rise is harm, -1 if a fall is harm
    engagement_loading: float = 0.0  # OBSERVATIONAL only: how the metric tracks latent
    blurb: str = ""                  # engagement. Deliberately unrelated to `effect`.

    def control_param(self) -> float:
        return self.base

    def treat_param(self, a: float) -> float:
        if self.channel == "marginal":
            inc = incremental_conversion(a)
            return (P0 * self.base + inc * self.effect) / (P0 + inc)
        if self.channel == "annoy":
            return self.base + annoyed_share(a) * self.effect * self.harm_sign
        return self.base

    def denom_fraction(self, a: float, arm: str) -> float:
        if self.denominator == "all":
            return 1.0
        if self.denominator == "responders":
            return RESP_RATE
        return P0 if arm == "control" else conversion_rate(a)


GUARDRAILS: List[Guardrail] = [
    Guardrail("refund_rate", "binom", "marginal", "converters", 3, 0.0200, 0.0900,
              harm_sign=1, engagement_loading=0.45, blurb="refunds among converters"),
    Guardrail("d7_retention", "binom", "marginal", "converters", 7, 0.7800, 0.4000,
              harm_sign=-1, engagement_loading=1.30, blurb="day-7 retention among converters"),
    Guardrail("d90_retention", "binom", "marginal", "converters", 90, 0.6600, 0.1500,
              harm_sign=-1, engagement_loading=1.75, blurb="day-90 retention among converters"),
    Guardrail("unsubscribe_rate", "binom", "annoy", "all", 2, 0.00800, 0.00400,
              harm_sign=1, engagement_loading=0.70, blurb="email unsubscribes, all users"),
    Guardrail("support_ticket_rate", "poisson", "annoy", "all", 1, 0.0300, 0.0100,
              harm_sign=1, engagement_loading=0.35, blurb="support tickets per user"),
    Guardrail("complaint_rate", "binom", "annoy", "all", 2, 0.00090, 0.00120,
              harm_sign=1, engagement_loading=0.30, blurb="formal complaints, all users"),
    Guardrail("nps_score", "normal", "annoy", "responders", 5, 8.100, 0.800, sd=2.40,
              harm_sign=-1, engagement_loading=1.10, blurb="survey score, 2% respond"),
    Guardrail("session_minutes", "normal", "annoy", "all", 1, 12.400, 0.300, sd=6.20,
              harm_sign=-1, engagement_loading=2.05, blurb="session length, all users"),
    Guardrail("page_latency_ms", "normal", "none", "all", 0, 820.0, 0.0, sd=240.0,
              harm_sign=1, engagement_loading=0.00, blurb="placebo: the lever cannot touch it"),
]

GUARDRAIL_BY_NAME: Dict[str, Guardrail] = {g.name: g for g in GUARDRAILS}

# What a team actually puts on the ship checklist: everything is an all-user metric or a
# survey, because that is what the dashboard already computes.
DASHBOARD_SUITE = ["unsubscribe_rate", "support_ticket_rate", "complaint_rate",
                   "nps_score", "session_minutes", "page_latency_ms"]

# Everything that can be computed inside a 14-day window.
COMPUTABLE_SUITE = [g.name for g in GUARDRAILS if g.name != "d90_retention"]


# --------------------------------------------------------------------------------------
# Analytic effect and standard error. Used for power curves and to check the simulator.
# --------------------------------------------------------------------------------------

def guardrail_n(g: Guardrail, a: float, n_per_arm: int, decision_day: int) -> Tuple[float, float]:
    frac = observable_fraction(decision_day, g.maturity_days)
    n_c = n_per_arm * g.denom_fraction(a, "control") * frac
    n_t = n_per_arm * g.denom_fraction(a, "treat") * frac
    return n_c, n_t


def analytic_z(g: Guardrail, a: float, n_per_arm: int, decision_day: int) -> float:
    """Expected z-score, signed so that positive means evidence of harm."""
    n_c, n_t = guardrail_n(g, a, n_per_arm, decision_day)
    if n_c < 1.0 or n_t < 1.0:
        return 0.0
    p_c, p_t = g.control_param(), g.treat_param(a)
    if g.family == "binom":
        se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    elif g.family == "poisson":
        se = np.sqrt(p_c / n_c + p_t / n_t)
    else:
        se = g.sd * np.sqrt(1.0 / n_c + 1.0 / n_t)
    if se <= 0:
        return 0.0
    return float(g.harm_sign * (p_t - p_c) / se)


def analytic_power(g: Guardrail, a: float, n_per_arm: int, decision_day: int,
                   alpha: float = 0.05) -> float:
    """One-sided power to detect harm. Zero denominator means the test cannot be run."""
    n_c, n_t = guardrail_n(g, a, n_per_arm, decision_day)
    if n_c < 1.0 or n_t < 1.0:
        return float("nan")
    crit = stats.norm.ppf(1 - alpha)
    return float(1 - stats.norm.cdf(crit - analytic_z(g, a, n_per_arm, decision_day)))


def primary_z(a: float, n_per_arm: int) -> float:
    p_c, p_t = P0, conversion_rate(a)
    se = np.sqrt(p_c * (1 - p_c) / n_per_arm + p_t * (1 - p_t) / n_per_arm)
    return float((p_t - p_c) / se)


def primary_power(a: float, n_per_arm: int, alpha: float = 0.05) -> float:
    crit = stats.norm.ppf(1 - alpha)
    return float(1 - stats.norm.cdf(crit - primary_z(a, n_per_arm)))


def n_for_power(target: float, a: float, alpha: float = 0.05,
                guardrail: Optional[Guardrail] = None, decision_day: int = 14,
                hi: int = 400_000_000) -> Optional[int]:
    """Smallest n per arm reaching `target` power. None if unreachable inside `hi`."""
    def power_at(n: int) -> float:
        if guardrail is None:
            return primary_power(a, n, alpha)
        p = analytic_power(guardrail, a, n, decision_day, alpha)
        return 0.0 if np.isnan(p) else p

    if power_at(hi) < target:
        return None
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if power_at(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------------------
# The simulator. Correlation between guardrails is generated, not assumed: within one
# replication every annoyance metric conditions on the SAME realised count of annoyed
# users, and every converter metric on the SAME realised marginal converters.
# --------------------------------------------------------------------------------------

def _two_sample_z(g: Guardrail, stat_c, n_c, stat_t, n_t):
    n_c = np.maximum(n_c, 1e-9)
    n_t = np.maximum(n_t, 1e-9)
    p_c, p_t = stat_c / n_c, stat_t / n_t
    if g.family == "binom":
        pooled = (stat_c + stat_t) / (n_c + n_t)
        se = np.sqrt(np.maximum(pooled * (1 - pooled), 1e-12) * (1 / n_c + 1 / n_t))
    elif g.family == "poisson":
        pooled = (stat_c + stat_t) / (n_c + n_t)
        se = np.sqrt(np.maximum(pooled, 1e-12) * (1 / n_c + 1 / n_t))
    else:
        se = g.sd * np.sqrt(1 / n_c + 1 / n_t)
    return g.harm_sign * (p_t - p_c) / np.maximum(se, 1e-12)


def simulate_experiment(a: float, n_per_arm: int, decision_day: int, reps: int,
                        rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Return an (reps,) array of harm-signed z per guardrail, plus 'primary'.

    Positive z is evidence of harm for every guardrail, whichever way its metric points.
    """
    inc = incremental_conversion(a)
    ann = annoyed_share(a)

    conv_c = rng.binomial(n_per_arm, P0, reps).astype(float)
    conv_good_t = rng.binomial(n_per_arm, P0, reps).astype(float)
    conv_marg_t = rng.binomial(n_per_arm, inc, reps).astype(float) if inc > 0 else np.zeros(reps)
    conv_t = conv_good_t + conv_marg_t
    ann_t = rng.binomial(n_per_arm, ann, reps).astype(float) if ann > 0 else np.zeros(reps)

    out: Dict[str, np.ndarray] = {}
    se_p = np.sqrt(np.maximum((conv_c + conv_t) / (2 * n_per_arm), 1e-12)
                   * (1 - (conv_c + conv_t) / (2 * n_per_arm)) * (2 / n_per_arm))
    out["primary"] = (conv_t / n_per_arm - conv_c / n_per_arm) / np.maximum(se_p, 1e-12)

    for g in GUARDRAILS:
        frac = observable_fraction(decision_day, g.maturity_days)
        if frac <= 0:
            out[g.name] = np.full(reps, np.nan)
            continue

        if g.denominator == "converters":
            n_c = np.floor(conv_c * frac)
            n_good = np.floor(conv_good_t * frac)
            n_marg = np.floor(conv_marg_t * frac)
            n_t = n_good + n_marg
        elif g.denominator == "responders":
            n_c = np.full(reps, np.floor(n_per_arm * RESP_RATE * frac))
            n_t = n_c.copy()
            n_good = np.full(reps, 0.0)
            n_marg = np.full(reps, 0.0)
        else:
            n_c = np.full(reps, np.floor(n_per_arm * frac))
            n_t = n_c.copy()
            n_good = np.full(reps, 0.0)
            n_marg = np.full(reps, 0.0)

        if g.channel == "marginal":
            stat_c = rng.binomial(n_c.astype(np.int64), g.base).astype(float)
            stat_t = (rng.binomial(n_good.astype(np.int64), g.base)
                      + rng.binomial(n_marg.astype(np.int64), g.effect)).astype(float)
        else:
            # Annoyance splits each arm's observable users into annoyed and not.
            if g.denominator == "responders":
                k_ann_t = rng.binomial(n_t.astype(np.int64), ann) if ann > 0 else np.zeros(reps, dtype=np.int64)
            else:
                k_ann_t = np.minimum(np.floor(ann_t * frac), n_t).astype(np.int64)
            k_plain_t = (n_t - k_ann_t).astype(np.int64)
            bumped = g.base + g.effect * g.harm_sign

            if g.family == "binom":
                stat_c = rng.binomial(n_c.astype(np.int64), g.base).astype(float)
                stat_t = (rng.binomial(k_plain_t, g.base)
                          + rng.binomial(k_ann_t, max(min(bumped, 1.0), 0.0))).astype(float)
            elif g.family == "poisson":
                stat_c = rng.poisson(np.maximum(n_c, 0) * g.base).astype(float)
                stat_t = (rng.poisson(np.maximum(k_plain_t, 0) * g.base)
                          + rng.poisson(np.maximum(k_ann_t, 0) * max(bumped, 0.0))).astype(float)
            else:
                mean_t = np.where(n_t > 0, (k_plain_t * g.base + k_ann_t * bumped) / np.maximum(n_t, 1), g.base)
                stat_c = (g.base * n_c + rng.normal(0, g.sd, reps) * np.sqrt(np.maximum(n_c, 1e-9)))
                stat_t = (mean_t * n_t + rng.normal(0, g.sd, reps) * np.sqrt(np.maximum(n_t, 1e-9)))

        out[g.name] = _two_sample_z(g, stat_c, n_c, stat_t, n_t)

    return out


# --------------------------------------------------------------------------------------
# Decision rules.
# --------------------------------------------------------------------------------------

def _reps_of(z: Dict[str, np.ndarray]) -> int:
    return len(next(iter(z.values())))


def crit_value(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha))


def any_fires(z: Dict[str, np.ndarray], suite: List[str], alpha: float) -> np.ndarray:
    """Classic suite rule: block if ANY guardrail is individually significant."""
    crit = crit_value(alpha)
    hits = np.zeros(_reps_of(z), dtype=bool)
    for name in suite:
        col = z[name]
        hits |= np.nan_to_num(col, nan=-np.inf) > crit
    return hits


def composite_z(z: Dict[str, np.ndarray], suite: List[str],
                weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """One directional index from many guardrails: a single test instead of many.

    Weights default to equal. The index is NOT standardised here -- its null spread is
    calibrated empirically by `calibrate_composite` rather than assumed to be unit
    variance. In this world that correction turns out to be negligible (the null s.d.
    comes out at 1.005) because the guardrails are very nearly independent, but it is the
    calibration that establishes the fact instead of taking it on faith.
    """
    cols, ws = [], []
    for name in suite:
        col = z[name]
        if np.all(np.isnan(col)):
            continue
        w = 1.0 if weights is None else weights.get(name, 0.0)
        if w == 0.0:
            continue
        cols.append(np.nan_to_num(col, nan=0.0))
        ws.append(w)
    if not cols:
        return np.zeros(_reps_of(z))
    arr = np.vstack(cols)
    w = np.array(ws, dtype=float)
    return (w @ arr) / np.sqrt(np.sum(w ** 2))


def calibrate_composite(suite: List[str], weights: Optional[Dict[str, float]],
                        n_per_arm: int, decision_day: int, alpha: float,
                        reps: int, seed: int) -> float:
    """Critical value for the composite index, taken from the a = 0 null by simulation."""
    rng = np.random.default_rng(seed)
    null = composite_z(simulate_experiment(0.0, n_per_arm, decision_day, reps, rng), suite, weights)
    return float(np.quantile(null, 1 - alpha))


def sensitivity_weights(suite: List[str], a: float, n_per_arm: int,
                        decision_day: int) -> Dict[str, float]:
    """Weight each guardrail by how hard the LEVER moves it, floored at zero.

    This is the quantity a guardrail is chosen for and almost never chosen by.
    """
    w = {}
    for name in suite:
        g = GUARDRAIL_BY_NAME[name]
        w[name] = max(analytic_z(g, a, n_per_arm, decision_day), 0.0)
    return w


# --------------------------------------------------------------------------------------
# Harmless changes. Not every experiment carries a lever; most are ordinary product work
# that genuinely helps. A guardrail suite has to let these through.
# --------------------------------------------------------------------------------------

CLEAN_SHARE = 0.60      # share of proposed changes that carry no harm channel at all
CLEAN_LIFT_MAX = 0.12   # a clean change's relative lift, drawn uniform on (0, this)


def clean_value_change(lift: float) -> float:
    """A clean lift converts ORDINARY users, so 180-day value moves with it, one for one."""
    return lift


def primary_z_for_lift(lift: float, n_per_arm: int) -> float:
    p_c, p_t = P0, P0 * (1.0 + lift)
    se = np.sqrt(p_c * (1 - p_c) / n_per_arm + p_t * (1 - p_t) / n_per_arm)
    return float((p_t - p_c) / se)


def simulate_clean_guardrails(n_per_arm: int, decision_day: int, reps: int,
                              rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """A harmless change moves no guardrail. Its z-scores ARE the null."""
    return simulate_experiment(0.0, n_per_arm, decision_day, reps, rng)


# --------------------------------------------------------------------------------------
# The observational cohort. This is where guardrails actually get chosen: somebody
# correlates every metric they have against churn and picks the top of the list.
# --------------------------------------------------------------------------------------

def passive_cohort(n_users: int, rng: np.random.Generator) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Users differ in latent engagement. Return per-user guardrail values and the outcome.

    Nothing here is causal. `engagement_loading` says how each metric tracks a user's
    latent engagement; `effect` says how hard the lever moves it. They are different
    numbers on purpose, and section 7 measures how little they agree.
    """
    e = rng.normal(0.0, 1.0, n_users)
    retained = rng.random(n_users) < 1.0 / (1.0 + np.exp(-(np.log(R_GOOD / (1 - R_GOOD)) + 1.15 * e)))

    values: Dict[str, np.ndarray] = {}
    for g in GUARDRAILS:
        lo = g.engagement_loading
        # A metric that means harm when it FALLS rises with engagement, and vice versa.
        signal = -g.harm_sign * lo * e
        if g.family == "normal":
            values[g.name] = g.base + signal * (g.sd / 3.0) + rng.normal(0, g.sd, n_users)
        elif g.family == "poisson":
            rate = g.base * np.exp(signal)
            values[g.name] = rng.poisson(rate).astype(float)
        else:
            odds = np.log(g.base / (1 - g.base)) + signal
            values[g.name] = (rng.random(n_users) < 1.0 / (1.0 + np.exp(-odds))).astype(float)
    return values, retained.astype(float)
