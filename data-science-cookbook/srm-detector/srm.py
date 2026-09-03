"""Sample ratio mismatch: the detectors, the worlds they are measured on, and
the analytic bias each world carries.

The library is deliberately split three ways because the three things are
independent and get confused constantly:

* ``detectors`` answer "are these two counts consistent with the intended
  split" and never see a conversion.
* ``simulate`` produces the counts AND the outcomes, from a world whose true
  effect and true bias are known in closed form.
* ``analytic`` gives the closed forms, so the simulation is checked against
  arithmetic rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# 1. Detectors
#
# Every detector maps (count_a, count_b, intended_share_of_a) to a number that
# is compared against a threshold, and flags when it comes out BELOW that
# threshold. For the five statistical tests the number is a p-value and the
# threshold is an alpha. The two "eyeball" rules are the ones teams actually
# use, and they have no p-value at all - they are wrapped to return 0.0 when
# they fire and 1.0 when they do not, so that the same harness can measure a
# rule of thumb and a hypothesis test on one axis. That wrapping is the whole
# point: it makes "outside 49/51" comparable to "p < 0.0005", and the
# comparison is not flattering to either default.
# --------------------------------------------------------------------------


def chi2_stat(count_a: int, count_b: int, share: float = 0.5, yates: bool = False) -> float:
    """Pearson chi-square goodness-of-fit statistic on one degree of freedom."""
    n = count_a + count_b
    if n == 0:
        return 0.0
    exp_a = n * share
    exp_b = n * (1.0 - share)
    dev = abs(count_a - exp_a)
    if yates:
        dev = max(0.0, dev - 0.5)
    return dev * dev / exp_a + dev * dev / exp_b


def p_chi2(count_a: int, count_b: int, share: float = 0.5) -> float:
    return float(stats.chi2.sf(chi2_stat(count_a, count_b, share), 1))


def p_chi2_yates(count_a: int, count_b: int, share: float = 0.5) -> float:
    return float(stats.chi2.sf(chi2_stat(count_a, count_b, share, yates=True), 1))


def p_g_test(count_a: int, count_b: int, share: float = 0.5) -> float:
    """Likelihood-ratio (G) test. Same asymptotic reference distribution."""
    n = count_a + count_b
    if n == 0:
        return 1.0
    exp = np.array([n * share, n * (1.0 - share)], dtype=float)
    obs = np.array([count_a, count_b], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(obs > 0, obs * np.log(obs / exp), 0.0)
    g = 2.0 * float(terms.sum())
    return float(stats.chi2.sf(g, 1))


def p_normal_z(count_a: int, count_b: int, share: float = 0.5) -> float:
    """Two-sided normal approximation on the observed share of arm A."""
    n = count_a + count_b
    if n == 0:
        return 1.0
    se = np.sqrt(share * (1.0 - share) / n)
    z = (count_a / n - share) / se
    return float(2.0 * stats.norm.sf(abs(z)))


def p_binom_exact(count_a: int, count_b: int, share: float = 0.5) -> float:
    """Two-sided exact binomial. The reference every approximation is judged against."""
    n = count_a + count_b
    if n == 0:
        return 1.0
    return float(stats.binomtest(int(count_a), int(n), share).pvalue)


def eyeball_abs(count_a: int, count_b: int, share: float = 0.5, tol: float = 0.01) -> float:
    """'Flag if the split is outside 49/51.' Returns 0.0 when it fires."""
    n = count_a + count_b
    if n == 0:
        return 1.0
    return 0.0 if abs(count_a / n - share) > tol else 1.0


def eyeball_ratio(count_a: int, count_b: int, share: float = 0.5, tol: float = 0.01) -> float:
    """'Flag if arm A / arm B is outside 0.99-1.01.' Returns 0.0 when it fires."""
    if count_b == 0:
        return 0.0
    ratio = count_a / count_b
    target = share / (1.0 - share)
    return 0.0 if abs(ratio / target - 1.0) > tol else 1.0


Detector = Callable[[int, int, float], float]

DETECTORS: Dict[str, Detector] = {
    "chi2": p_chi2,
    "chi2_yates": p_chi2_yates,
    "g_test": p_g_test,
    "normal_z": p_normal_z,
    "eyeball_1pct_abs": eyeball_abs,
    "eyeball_1pct_ratio": eyeball_ratio,
}

# Exact binomial is held out of the default dict because it is O(n) per call
# and the sweeps run it thousands of times; it is used where it is the point.
EXACT: Dict[str, Detector] = {"binom_exact": p_binom_exact}

# The two thresholds in circulation. 0.05 is the reflex; 0.0005 is the one
# large-scale experimentation platforms publish, and section 6 measures why.
ALPHA_REFLEX = 0.05
ALPHA_PLATFORM = 0.0005


def chi2_critical(alpha: float) -> float:
    return float(stats.chi2.ppf(1.0 - alpha, 1))


# --------------------------------------------------------------------------
# 2. The world
#
# Two strata, because a one-stratum world cannot express the thing that makes
# SRM dangerous: that the users a broken assignment loses are not a random
# sample of the users it keeps. Low-intent users convert at p_low and are the
# ones a slow redirect drops.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    per_arm: int = 100_000
    low_share: float = 0.30
    p_low: float = 0.02
    p_high: float = 0.13428571428571429  # gives an exact 0.10 blended base rate
    true_rel_lift: float = 0.05

    @property
    def base_rate(self) -> float:
        return self.low_share * self.p_low + (1.0 - self.low_share) * self.p_high


MECHANISMS = (
    "healthy",
    "mcar_loss",
    "selective_loss",
    "balanced_selective",
)


def simulate(
    world: World,
    mechanism: str,
    rate: float,
    trials: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Run ``trials`` independent experiments and return per-trial arrays.

    ``rate`` means a different thing per mechanism, deliberately - each is the
    natural knob for that failure - and :func:`count_loss_of` converts all of
    them onto the one axis that a detector can see.

    * ``healthy`` - nothing is lost.
    * ``mcar_loss`` - the treatment arm loses ``rate`` of its records at
      random (a logging endpoint dropping requests). Missing completely at
      random: the split moves, the effect estimate does not.
    * ``selective_loss`` - the treatment arm loses ``rate`` of its LOW-INTENT
      users (a redirect they bounce out of). The split moves less, and the
      effect estimate moves.
    * ``balanced_selective`` - the same selective loss in treatment, plus the
      identical NUMBER of users removed from control at random. The split does
      not move at all. The effect estimate moves exactly as much.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; expected one of {MECHANISMS}")

    # Assignment randomises the TRAFFIC between arms - it does not hand each
    # arm a fixed quota. That is the whole reason the arm counts have a null
    # distribution to test in the first place, and a simulator that gives each
    # arm exactly per_arm users has no null: every p-value comes out at 1.0 and
    # the healthy world reports a false-positive rate of zero.
    n_total = 2 * world.per_arm
    assigned_c = rng.binomial(n_total, 0.5, trials)
    assigned_t = n_total - assigned_c

    low_c = rng.binomial(assigned_c, world.low_share)
    high_c = assigned_c - low_c
    low_t = rng.binomial(assigned_t, world.low_share)
    high_t = assigned_t - low_t

    if mechanism == "mcar_loss":
        keep = 1.0 - rate
        low_t = rng.binomial(low_t, keep)
        high_t = rng.binomial(high_t, keep)
    elif mechanism in ("selective_loss", "balanced_selective"):
        dropped = rng.binomial(low_t, rate)
        low_t = low_t - dropped
        if mechanism == "balanced_selective":
            # Remove the same COUNT from control, chosen at random across its
            # strata -> hypergeometric, not binomial.
            take = np.minimum(dropped, assigned_c)
            take_low = rng.hypergeometric(low_c, high_c, take)
            low_c = low_c - take_low
            high_c = high_c - (take - take_low)

    p_low_t = world.p_low * (1.0 + world.true_rel_lift)
    p_high_t = world.p_high * (1.0 + world.true_rel_lift)

    conv_c = rng.binomial(low_c, world.p_low) + rng.binomial(high_c, world.p_high)
    conv_t = rng.binomial(low_t, p_low_t) + rng.binomial(high_t, p_high_t)

    n_c = low_c + high_c
    n_t = low_t + high_t
    rate_c = np.divide(conv_c, n_c, out=np.zeros(trials, float), where=n_c > 0)
    rate_t = np.divide(conv_t, n_t, out=np.zeros(trials, float), where=n_t > 0)
    est = np.divide(rate_t, rate_c, out=np.zeros(trials, float), where=rate_c > 0) - 1.0

    return {
        "n_ctrl": n_c,
        "n_trt": n_t,
        "rate_ctrl": rate_c,
        "rate_trt": rate_t,
        "est_rel_lift": est,
        "share_ctrl": n_c / np.maximum(n_c + n_t, 1),
    }


# --------------------------------------------------------------------------
# 3. Closed forms
# --------------------------------------------------------------------------


def count_loss_of(world: World, mechanism: str, rate: float) -> float:
    """Expected fraction of the treatment arm's records that go missing.

    This is the only thing a detector can see, and it is what puts three
    mechanisms with wildly different consequences onto one axis.
    """
    if mechanism == "healthy":
        return 0.0
    if mechanism == "mcar_loss":
        return rate
    if mechanism in ("selective_loss", "balanced_selective"):
        return world.low_share * rate
    raise ValueError(mechanism)


def expected_share(world: World, mechanism: str, rate: float) -> float:
    """Expected observed share of the CONTROL arm."""
    loss = count_loss_of(world, mechanism, rate)
    if mechanism == "balanced_selective":
        return 0.5  # equal counts removed from both arms
    return 1.0 / (2.0 - loss)


def analytic_est_lift(world: World, mechanism: str, rate: float) -> float:
    """Closed-form relative lift an infinite version of this world reports."""
    w, pl, ph = world.low_share, world.p_low, world.p_high
    lift = world.true_rel_lift

    if mechanism in ("healthy", "mcar_loss"):
        return lift  # both arms keep their stratum mix

    # Treatment loses `rate` of its low-intent users.
    low_t = w * (1.0 - rate)
    high_t = 1.0 - w
    obs_t = (low_t * pl * (1.0 + lift) + high_t * ph * (1.0 + lift)) / (low_t + high_t)

    if mechanism == "selective_loss":
        obs_c = w * pl + (1.0 - w) * ph
    else:  # balanced_selective: control loses the same count, at random
        removed = w * rate
        obs_c = (w * pl + (1.0 - w) * ph) * (1.0 - removed) / (1.0 - removed)
    return obs_t / obs_c - 1.0


def _power_one_prop(n: int, delta: float, alpha: float, share: float = 0.5) -> float:
    """Power of the two-sided share test to see a share of ``share + delta``."""
    se0 = np.sqrt(share * (1.0 - share) / n)
    p1 = share + delta
    se1 = np.sqrt(p1 * (1.0 - p1) / n)
    crit = stats.norm.isf(alpha / 2.0) * se0
    return float(stats.norm.sf((crit - abs(delta)) / se1) + stats.norm.cdf((-crit - abs(delta)) / se1))


def mdd_share(n_total: int, alpha: float, power: float = 0.80, share: float = 0.5) -> float:
    """Minimum detectable deviation in the observed split, as an absolute share."""
    lo, hi = 1e-9, 0.49
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _power_one_prop(n_total, mid, alpha, share) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def mde_rel_lift(per_arm: int, base: float, alpha: float, power: float = 0.80) -> float:
    """Minimum detectable RELATIVE lift for the experiment itself."""
    z_a = stats.norm.isf(alpha / 2.0)
    z_b = stats.norm.isf(1.0 - power)

    lo, hi = 1e-9, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        p2 = base * (1.0 + mid)
        se_null = np.sqrt(2.0 * base * (1.0 - base) / per_arm)
        se_alt = np.sqrt((base * (1.0 - base) + p2 * (1.0 - p2)) / per_arm)
        if (z_a * se_null + z_b * se_alt) > base * mid:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def loss_for_share_deviation(deviation: float) -> float:
    """Fraction of ONE arm that must vanish to move the split by ``deviation``.

    share = 1 / (2 - loss)  ->  loss = 2 - 1/share
    """
    share = 0.5 + abs(deviation)
    return 2.0 - 1.0 / share


# --------------------------------------------------------------------------
# 4. Segmented assignment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    name: str
    share: float
    base_rate: float


DEFAULT_SEGMENTS: Tuple[Segment, ...] = (
    Segment("chrome", 0.62, 0.110),
    Segment("android", 0.23, 0.095),
    Segment("safari", 0.15, 0.062),
)


def simulate_segmented(
    per_arm: int,
    segments: Tuple[Segment, ...],
    broken: Optional[str],
    loss: float,
    trials: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Assignment where the loss is confined to ONE segment.

    Returns per-trial control counts and treatment counts, per segment, plus
    the totals a platform-level SRM check would actually be handed.
    """
    n_seg = len(segments)
    n_c = np.zeros((n_seg, trials), dtype=np.int64)
    n_t = np.zeros((n_seg, trials), dtype=np.int64)
    n_total = 2 * per_arm
    for i, seg in enumerate(segments):
        # Traffic in this segment, then randomised between arms - same reason
        # as in `simulate`: the arm counts need a null distribution.
        seg_total = rng.binomial(n_total, seg.share, trials)
        n_c[i] = rng.binomial(seg_total, 0.5)
        n_t[i] = seg_total - n_c[i]
        if broken is not None and seg.name == broken and loss > 0:
            n_t[i] = rng.binomial(n_t[i], 1.0 - loss)
    return {
        "n_ctrl_seg": n_c,
        "n_trt_seg": n_t,
        "n_ctrl": n_c.sum(axis=0),
        "n_trt": n_t.sum(axis=0),
    }


# --------------------------------------------------------------------------
# 5. Applying detectors over arrays
# --------------------------------------------------------------------------


def flag_rate(
    n_a: np.ndarray,
    n_b: np.ndarray,
    detector: Detector,
    alpha: float,
    share: float = 0.5,
    limit: Optional[int] = None,
) -> float:
    """Fraction of trials in which ``detector`` fires at ``alpha``."""
    take = len(n_a) if limit is None else min(limit, len(n_a))
    fired = 0
    for i in range(take):
        if detector(int(n_a[i]), int(n_b[i]), share) < alpha:
            fired += 1
    return fired / take


def vector_p_chi2(n_a: np.ndarray, n_b: np.ndarray, share: float = 0.5) -> np.ndarray:
    """Vectorised chi-square p-values - the sweeps need thousands of these."""
    n = (n_a + n_b).astype(float)
    exp_a = n * share
    exp_b = n * (1.0 - share)
    dev = np.abs(n_a - exp_a)
    stat = dev * dev / exp_a + dev * dev / exp_b
    return stats.chi2.sf(stat, 1)


def sequential_srm_fpr(
    per_arm_final: int,
    looks: int,
    alpha: float,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """False-positive rate of checking SRM at every look and stopping on the first flag.

    The same optional-stopping arithmetic Day 164 priced for the effect test
    applies unchanged to the health check, and nobody applies it there.
    """
    step = per_arm_final // looks
    n_c = np.zeros(trials, dtype=np.int64)
    n_t = np.zeros(trials, dtype=np.int64)
    fired = np.zeros(trials, dtype=bool)
    for _ in range(looks):
        arrivals = 2 * step
        to_c = rng.binomial(arrivals, 0.5, trials)
        n_c = n_c + to_c
        n_t = n_t + (arrivals - to_c)
        p = vector_p_chi2(n_c, n_t)
        fired |= p < alpha
    return float(fired.mean())
