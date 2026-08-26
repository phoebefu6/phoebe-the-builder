"""Expected value is a number. It is not a decision.

Somebody asks which option to take, so you build the spreadsheet: probability
times payoff, one row per option, pick the biggest.  That arithmetic is right
and the decision it produces can still be wrong, in four separate ways this
module measures:

  * **Plugging in averages.** The payoff is not linear in the inputs, so the
    number you get from average inputs is not the average outcome. It is a
    number that may never occur.
  * **Comparing point estimates.** The option with the higher expected value
    can be the worse one most of the time. EV ranks the mean; nobody
    experiences the mean once.
  * **Repeating the bet.** A gamble with positive expected value can have
    negative growth when the payoffs multiply rather than add. Maximise EV
    across a sequence and you go broke while the average goes up.
  * **Deciding at all.** Sometimes the answer is to find out first, and
    whether that is worth doing is itself computable - and almost never
    computed.

Everything here is simulation and closed form on one authored worked example:
a build-versus-buy decision with four uncertain inputs, and a repeated bet.
The example is authored; the arithmetic on it is asserted in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

RNG_SEED = 20260827
N_SIMS = 200_000


# --------------------------------------------------------------------------
# The inputs, as ranges rather than points
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Input:
    """One uncertain quantity, elicited the way people actually give them.

    A person can say "about 40, could be 25, could be 70". They cannot say
    "lognormal with sigma 0.31". `low` and `high` are the 10th and 90th
    percentiles, which is the elicitation people can actually do.
    """

    name: str
    low: float
    mid: float
    high: float
    unit: str
    note: str = ""

    def draw(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """A PERT-ish draw: a beta on [low, high] peaked at mid.

        Chosen over a triangular because real estimates have thin tails and
        a triangular's corners create artefacts in the tornado.
        """
        if self.high <= self.low:
            return np.full(n, self.mid)
        span = self.high - self.low
        mode = (self.mid - self.low) / span
        mode = min(max(mode, 1e-6), 1 - 1e-6)
        conc = 4.0
        a = 1 + conc * mode
        b = 1 + conc * (1 - mode)
        return self.low + rng.beta(a, b, n) * span


#: A build-versus-buy decision. Four inputs, none of them known.
INPUTS: Tuple[Input, ...] = (
    Input("seats", 12, 32, 90, "seats",
          "how many people end up actually using it"),
    Input("hours_saved", 0.4, 1.4, 3.0, "hours/seat/week",
          "the benefit, per seat, per week"),
    Input("hourly_cost", 45, 70, 95, "currency/hour",
          "fully loaded"),
    Input("build_months", 3.0, 6.0, 15.0, "months",
          "engineering time before anything works"),
)

INPUTS_BY_NAME: Dict[str, Input] = {i.name: i for i in INPUTS}

#: Costs that are known well enough to treat as fixed.
BUILD_TEAM_COST_PER_MONTH = 30_000.0
BUY_LICENCE_PER_SEAT_YEAR = 1_800.0
BUY_ONBOARDING = 80_000.0
HORIZON_YEARS = 3.0
#: The vendor licenses a fixed pool; benefit above it is not delivered, and
#: this cap is what makes `buy` flat in seats while `build` keeps scaling.
BUY_SEAT_CAP = 20.0
#: A built tool needs upkeep whether or not anyone uses it.
BUILD_MAINTENANCE_PER_YEAR = 45_000.0


# --------------------------------------------------------------------------
# The payoff, which is not linear
# --------------------------------------------------------------------------


def value_of_option(option: str, seats: np.ndarray, hours_saved: np.ndarray,
                    hourly_cost: np.ndarray, build_months: np.ndarray) -> np.ndarray:
    """Three-year net value of each option, in currency.

    Two nonlinearities do the damage, and both are ordinary:

      * `buy` caps delivered benefit at the licensed seat count, so the
        payoff is a `min()` - concave, and averages overstate it.
      * `build` delivers nothing until it ships, so benefit is earned over
        `HORIZON_YEARS - build_months/12`, floored at zero. A build that
        overruns past the horizon returns pure cost.
    """
    weekly = seats * hours_saved * hourly_cost
    annual_benefit = weekly * 46.0            # working weeks

    if option == "buy":
        delivered = np.minimum(seats, BUY_SEAT_CAP) * hours_saved * hourly_cost * 46.0
        cost = BUY_ONBOARDING + np.minimum(seats, BUY_SEAT_CAP) * \
            BUY_LICENCE_PER_SEAT_YEAR * HORIZON_YEARS
        return delivered * HORIZON_YEARS - cost

    if option == "build":
        live_years = np.maximum(HORIZON_YEARS - build_months / 12.0, 0.0)
        cost = build_months * BUILD_TEAM_COST_PER_MONTH + \
            BUILD_MAINTENANCE_PER_YEAR * live_years
        return annual_benefit * live_years - cost

    if option == "defer":
        return np.zeros_like(seats)

    raise ValueError(f"unknown option: {option}")


OPTIONS: Tuple[str, ...] = ("build", "buy", "defer")


@lru_cache(maxsize=8)
def simulate(n: int = N_SIMS, seed: int = RNG_SEED) -> Dict[str, np.ndarray]:
    """One shared set of draws, every option evaluated on each."""
    rng = np.random.default_rng(seed)
    draws = {i.name: i.draw(rng, n) for i in INPUTS}
    out = {opt: value_of_option(opt, **draws) for opt in OPTIONS}
    out.update({f"input:{k}": v for k, v in draws.items()})
    return out


def midpoints() -> Dict[str, float]:
    return {i.name: i.mid for i in INPUTS}


# --------------------------------------------------------------------------
# 1. The flaw of averages
# --------------------------------------------------------------------------


def naive_point_estimate() -> Dict[str, float]:
    """What the spreadsheet says: the middle of each range in the formula.

    "Middle" here is the elicited most-likely value - the number a person
    actually types - which for a skewed range is the mode, not the mean.
    """
    mids = {k: np.array([v]) for k, v in midpoints().items()}
    return {opt: float(value_of_option(opt, **mids)[0]) for opt in OPTIONS}


def input_means() -> Dict[str, float]:
    """The simulated mean of each input, which is not its elicited middle."""
    sims = simulate()
    return {i.name: float(sims[f"input:{i.name}"].mean()) for i in INPUTS}


def estimate_at_input_means() -> Dict[str, float]:
    means = {k: np.array([v]) for k, v in input_means().items()}
    return {opt: float(value_of_option(opt, **means)[0]) for opt in OPTIONS}


def true_expected_value() -> Dict[str, float]:
    sims = simulate()
    return {opt: float(sims[opt].mean()) for opt in OPTIONS}


def mode_vs_mean() -> Dict[str, Dict[str, float]]:
    """Error one: the number people type is the mode, not the mean.

    A range like 3-6-15 months has a mean well above 6. This error has
    nothing to do with the payoff being nonlinear; it is upstream of it,
    and it is the reason a naive estimate and a mean-input estimate differ.
    """
    mids, means = midpoints(), input_means()
    return {
        k: {"elicited_mid": mids[k], "actual_mean": means[k],
            "shift": means[k] - mids[k]}
        for k in mids
    }


def flaw_of_averages() -> Dict[str, Dict[str, float]]:
    """Error two: f(E[x]) is not E[f(x)], even with the right means in.

    Measured at the true input MEANS so the mode/mean error above cannot
    contaminate it - what is left is purely the payoff's curvature.
    Jensen: concave payoff, point estimate too high. The spreadsheet has no
    way to tell you which case it is in.
    """
    at_means, true = estimate_at_input_means(), true_expected_value()
    naive = naive_point_estimate()
    out = {}
    for opt in OPTIONS:
        gap = at_means[opt] - true[opt]
        out[opt] = {
            "elicited_mid_estimate": naive[opt],
            "at_input_means": at_means[opt],
            "true_ev": true[opt],
            "jensen_gap": gap,
            "jensen_overstates_by": (gap / abs(true[opt])) if true[opt] else 0.0,
        }
    return out


def probability_of_the_point_estimate(tolerance: float = 0.05) -> Dict[str, float]:
    """How often the real outcome lands within +-5% of the spreadsheet number."""
    sims, naive = simulate(), naive_point_estimate()
    out = {}
    for opt in OPTIONS:
        if opt == "defer":
            continue
        target = naive[opt]
        band = abs(target) * tolerance
        out[opt] = float(np.mean(np.abs(sims[opt] - target) <= band))
    return out


# --------------------------------------------------------------------------
# 2. Ranking the mean is not ranking the option
# --------------------------------------------------------------------------


def beats(a: str, b: str) -> float:
    """P(option a delivers more than option b) on the same draw."""
    sims = simulate()
    return float(np.mean(sims[a] > sims[b]))


def ranking_conflict() -> Dict[str, object]:
    """Where 'higher EV' and 'wins more often' disagree."""
    ev = true_expected_value()
    best_ev = max(OPTIONS, key=lambda o: ev[o])
    pairs = {}
    conflicts = []
    for a in OPTIONS:
        for b in OPTIONS:
            if a == b:
                continue
            pairs[(a, b)] = beats(a, b)
            if ev[a] > ev[b] and pairs[(a, b)] < 0.5:
                conflicts.append((a, b, ev[a] - ev[b], pairs[(a, b)]))
    return {"ev": ev, "best_by_ev": best_ev, "pairs": pairs, "conflicts": conflicts}


def downside(option: str, q: float = 0.1) -> Dict[str, float]:
    sims = simulate()[option]
    return {
        "p10": float(np.quantile(sims, q)),
        "median": float(np.median(sims)),
        "p90": float(np.quantile(sims, 1 - q)),
        "p_loss": float(np.mean(sims < 0)),
    }


# --------------------------------------------------------------------------
# 3. Which input actually decides it
# --------------------------------------------------------------------------


def tornado(option_a: str = "build", option_b: str = "buy") -> List[Tuple[str, float, float, float]]:
    """Swing in (A - B) when one input moves P10->P90, others held at mid.

    One-at-a-time, which is what a tornado chart is. Its limitation is
    reported alongside it by `interaction_share`.
    """
    base = midpoints()
    rows = []
    for inp in INPUTS:
        vals = {}
        for label, v in (("low", inp.low), ("high", inp.high)):
            args = {k: np.array([base[k]]) for k in base}
            args[inp.name] = np.array([v])
            vals[label] = float(value_of_option(option_a, **args)[0] -
                                value_of_option(option_b, **args)[0])
        rows.append((inp.name, vals["low"], vals["high"], abs(vals["high"] - vals["low"])))
    return sorted(rows, key=lambda r: -r[3])


def interaction_share(option_a: str = "build", option_b: str = "buy") -> Dict[str, float]:
    """How much of the real variance the one-at-a-time picture misses.

    Compares the variance a tornado implies (inputs varied singly, effects
    added) against the variance of the full joint simulation.
    """
    sims = simulate()
    diff = sims[option_a] - sims[option_b]
    joint_var = float(np.var(diff))
    swings = [r[3] for r in tornado(option_a, option_b)]
    # A P10-P90 swing is ~2.56 sd for a normal; treat each swing as a sd
    # contribution and add in quadrature, which is what OAT implicitly does.
    oat_var = float(sum((s / 2.563) ** 2 for s in swings))
    return {
        "joint_variance": joint_var,
        "oat_variance": oat_var,
        "ratio": joint_var / oat_var if oat_var else math.inf,
    }


def switching_point(input_name: str, option_a: str = "build",
                    option_b: str = "buy") -> Optional[float]:
    """The value of one input at which the recommendation flips.

    More useful than any point estimate: it converts "which option" into
    "what would have to be true", which is a question people can check.
    """
    inp = INPUTS_BY_NAME[input_name]
    base = midpoints()

    def gap(x: float) -> float:
        args = {k: np.array([base[k]]) for k in base}
        args[input_name] = np.array([x])
        return float(value_of_option(option_a, **args)[0] -
                     value_of_option(option_b, **args)[0])

    lo, hi = inp.low, inp.high
    if gap(lo) * gap(hi) > 0:
        return None  # no flip anywhere in the plausible range
    for _ in range(200):
        mid = (lo + hi) / 2
        if gap(lo) * gap(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def switching_points() -> Dict[str, Optional[float]]:
    return {i.name: switching_point(i.name) for i in INPUTS}


# --------------------------------------------------------------------------
# 4. What it is worth to find out first
# --------------------------------------------------------------------------


def evpi() -> Dict[str, float]:
    """Expected value of perfect information.

    E[max over options given the truth] - max over options of E[.]
    It is the ceiling on what any study, pilot or spike can be worth. If a
    proposed investigation costs more than this, it cannot pay for itself
    however good it is - and that is decidable before commissioning it.
    """
    sims = simulate()
    stacked = np.vstack([sims[o] for o in OPTIONS])
    best_known = float(np.max([sims[o].mean() for o in OPTIONS]))
    with_clairvoyance = float(np.max(stacked, axis=0).mean())
    return {
        "best_without_information": best_known,
        "with_perfect_information": with_clairvoyance,
        "evpi": with_clairvoyance - best_known,
    }


def evppi(input_name: str, bins: int = 40) -> float:
    """Expected value of perfect information about ONE input.

    Resolving everything is not on offer. This is what learning a single
    quantity is worth, which is what actually gets commissioned - and the
    parts do not sum to the whole.
    """
    sims = simulate()
    x = sims[f"input:{input_name}"]
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    total, n = 0.0, len(x)
    for k in range(bins):
        m = idx == k
        if not m.any():
            continue
        total += m.sum() * max(sims[o][m].mean() for o in OPTIONS)
    best_known = max(sims[o].mean() for o in OPTIONS)
    return float(total / n - best_known)


def information_value() -> Dict[str, float]:
    out = {i.name: evppi(i.name) for i in INPUTS}
    out["_all (EVPI)"] = evpi()["evpi"]
    return out


# --------------------------------------------------------------------------
# 5. Repeating the bet: the average goes up and you go broke
# --------------------------------------------------------------------------

#: A gamble with unambiguously positive expected value, per round.
UP, DOWN, P_UP = 1.5, 0.6, 0.5


def ensemble_growth() -> float:
    """Average multiplier per round. Above 1 means positive expected value."""
    return P_UP * UP + (1 - P_UP) * DOWN


def time_average_growth() -> float:
    """Per-round growth an individual trajectory actually experiences.

    The geometric mean. When payoffs multiply, this is the number that
    decides whether you end up with more than you started with - and it can
    be below 1 while the arithmetic mean is above it.
    """
    return math.exp(P_UP * math.log(UP) + (1 - P_UP) * math.log(DOWN))


def trajectories(rounds: int = 250, n: int = 20_000, fraction: float = 1.0,
                 seed: int = RNG_SEED + 5) -> Dict[str, float]:
    """Bet `fraction` of the bankroll each round, `rounds` times.

    fraction=1.0 is what maximising expected value per round tells you to do.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n, rounds)) < P_UP
    mult = np.where(wins, 1 + fraction * (UP - 1), 1 - fraction * (1 - DOWN))
    wealth = np.prod(mult, axis=1)
    return {
        "fraction": fraction,
        "mean": float(wealth.mean()),
        "median": float(np.median(wealth)),
        "p_below_start": float(np.mean(wealth < 1.0)),
        "p_ruin_99pct": float(np.mean(wealth < 0.01)),
    }


def kelly_fraction() -> float:
    """The fraction maximising expected log wealth. Closed form for two outcomes."""
    b, a = UP - 1.0, 1.0 - DOWN
    f = (P_UP * b - (1 - P_UP) * a) / (a * b)
    return max(0.0, min(1.0, f))


def sizing_comparison(rounds: int = 250) -> List[Dict[str, float]]:
    k = kelly_fraction()
    return [trajectories(rounds=rounds, fraction=f)
            for f in (1.0, 0.5, k, k / 2)]


@lru_cache(maxsize=1)
def summary() -> Dict[str, object]:
    return {
        "naive": naive_point_estimate(),
        "true_ev": true_expected_value(),
        "conflict": ranking_conflict(),
        "evpi": evpi(),
        "ensemble": ensemble_growth(),
        "time_average": time_average_growth(),
        "kelly": kelly_fraction(),
    }
