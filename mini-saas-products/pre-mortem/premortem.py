"""Running a pre-mortem, and scoring what it produces.

A pre-mortem is cheap and it works: assume the project already failed, then
write down why.  Prospective hindsight beats "what could go wrong" because
a completed event is easier to explain than a hypothetical one.

The problem is what happens next.  The output is a list of failure modes,
and almost every organisation scores that list on a 5x5 **risk matrix** -
likelihood band times impact band.  That step is not a convenience, it is
a lossy transform, and Cox (2008) showed it cannot rank risks correctly:
qualitative matrices compress ranges, invert orderings, and are provably
unable to reproduce the ordering of the quantitative risks they came from.

This module does both halves.  It builds the plan arithmetic a pre-mortem
is supposed to expose - a twelve-step plan of 95% steps does not succeed
95% of the time - and then measures, on one authored set of failure modes,
how often a standard matrix mis-ranks them against their own expected
loss, how much range collapses into a single cell, and how differently
two equally conventional scales score identical risks.

The plan and the failure modes are **authored, not sampled**: they are the
worked example. Everything computed about them - the conjunction
arithmetic, the correlation gap, the mis-ranking rate, the prevention
ordering - is arithmetic on that example and is asserted in the tests.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

RNG_SEED = 20260826


# --------------------------------------------------------------------------
# The plan: a conjunction, not a single event
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    name: str
    p_success: float


#: A migration nobody would call risky, step by step. Every number is the
#: kind of confidence a competent engineer states out loud.
PLAN: Tuple[Step, ...] = (
    Step("Stand up the new warehouse", 0.99),
    Step("Replicate historical data", 0.96),
    Step("Row counts reconcile", 0.93),
    Step("Rewrite the 40 heaviest queries", 0.92),
    Step("Semantics match on the top 20 metrics", 0.90),
    Step("dbt models build green", 0.95),
    Step("Downstream dashboards repoint", 0.94),
    Step("Access rules survive the move", 0.97),
    Step("Nightly SLA still met", 0.91),
    Step("Cost lands inside budget", 0.88),
    Step("Cutover weekend runs clean", 0.93),
    Step("Two weeks with no rollback", 0.95),
)


def plan_success(steps: Sequence[Step] = PLAN) -> float:
    """P(every step works), assuming independence."""
    return float(np.prod([s.p_success for s in steps]))


def weakest_step_success(steps: Sequence[Step] = PLAN) -> float:
    return min(s.p_success for s in steps)


def steps_to_coin_flip(steps: Sequence[Step] = PLAN) -> int:
    """How many steps of the plan's *average* quality reach 50/50."""
    avg = float(np.mean([s.p_success for s in steps]))
    return math.ceil(math.log(0.5) / math.log(avg))


def correlated_plan_success(rho_shock: float = 0.12,
                            steps: Sequence[Step] = PLAN,
                            n: int = 400_000) -> Dict[str, float]:
    """The same plan when the steps share a common cause.

    Independence is the optimistic assumption, not the neutral one. A
    common shock - the one engineer who knows the old warehouse leaves, the
    vendor slips - fires with probability `rho_shock` and, when it does,
    each step's failure probability triples. Survival falls below the
    product rule, which is already far below anyone's stated confidence.
    """
    rng = np.random.default_rng(RNG_SEED)
    p_fail = np.array([1.0 - s.p_success for s in steps])
    shock = rng.random(n) < rho_shock
    draws = rng.random((n, len(steps)))
    scaled = np.where(shock[:, None], np.minimum(p_fail * 3.0, 1.0), p_fail)
    survived = float((draws >= scaled).all(axis=1).mean())
    return {
        "independent": plan_success(steps),
        "correlated": survived,
        "shock_probability": rho_shock,
        "gap": plan_success(steps) - survived,
    }


# --------------------------------------------------------------------------
# What the pre-mortem produces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureMode:
    """One way the project is dead in six months.

    `probability` and `loss` are the quantities a matrix throws away.
    `prevention_cost` and `prevention_effect` are what nobody records, and
    they are the two numbers that actually decide what to do this week.
    """

    id: str
    cause: str
    probability: float
    loss: float               # currency, if it happens
    prevention_cost: float    # currency, spent now, whether or not it happens
    prevention_effect: float  # fraction of probability removed, 0-1
    surfaced_by: str = "prospective hindsight"

    @property
    def expected_loss(self) -> float:
        return self.probability * self.loss

    @property
    def prevention_value(self) -> float:
        """Expected loss avoided, net of what avoiding it costs."""
        return self.probability * self.prevention_effect * self.loss - self.prevention_cost

    @property
    def prevention_ratio(self) -> float:
        if self.prevention_cost <= 0:
            return math.inf
        return (self.probability * self.prevention_effect * self.loss) / self.prevention_cost


#: Fourteen causes, written the way a pre-mortem produces them: a specific
#: mechanism, not a category. Authored - this is the worked example.
MODES: Tuple[FailureMode, ...] = (
    FailureMode("F01", "Metric definitions silently differ between the two engines",
                0.55, 400_000, 25_000, 0.70),
    FailureMode("F02", "The one engineer who knows the legacy warehouse leaves mid-migration",
                0.18, 900_000, 40_000, 0.55),
    FailureMode("F03", "Query costs land 3x over budget and finance halts the project",
                0.30, 600_000, 15_000, 0.80),
    FailureMode("F04", "Historical data fails to reconcile and nobody can say which side is right",
                0.35, 500_000, 30_000, 0.65),
    FailureMode("F05", "Timezone handling changes and every daily report shifts by a day",
                0.40, 120_000, 6_000, 0.85),
    FailureMode("F06", "Access rules do not survive the move; someone sees the salary table",
                0.08, 4_000_000, 45_000, 0.75),
    FailureMode("F07", "Nightly SLA slips and the exec dashboard is empty at 8am",
                0.45, 90_000, 12_000, 0.70),
    FailureMode("F08", "Cutover weekend overruns and Monday opens on a half-migrated warehouse",
                0.22, 1_100_000, 60_000, 0.60),
    FailureMode("F09", "Downstream dashboards repoint but three are missed for a quarter",
                0.50, 60_000, 5_000, 0.90),
    FailureMode("F10", "Vendor slips the contract date and the window closes",
                0.15, 700_000, 20_000, 0.30),
    FailureMode("F11", "A float rounding change moves reported revenue by a cent per row",
                0.28, 250_000, 8_000, 0.80),
    FailureMode("F12", "Nobody owns the rollback and the decision takes four days",
                0.20, 800_000, 10_000, 0.75),
    FailureMode("F13", "The migration succeeds and nobody uses the new warehouse",
                0.33, 350_000, 18_000, 0.45),
    FailureMode("F14", "A regulator asks for lineage during the transition and there is none",
                0.06, 2_500_000, 35_000, 0.70),
)


def total_expected_loss(modes: Sequence[FailureMode] = MODES) -> float:
    return float(sum(m.expected_loss for m in modes))


# --------------------------------------------------------------------------
# Risk matrices - two of them, both entirely conventional
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    """An ordinal risk scale: bin edges for probability and for impact.

    Both scales below are the shape found in real corporate risk
    templates. Nothing here is a straw man; the point is that two
    defensible scales disagree.
    """

    name: str
    p_edges: Tuple[float, ...]
    loss_edges: Tuple[float, ...]
    note: str

    def p_band(self, p: float) -> int:
        """1..5. One-indexed on purpose: a band of 0 would zero the product,
        so the lowest-likelihood row would score 0 at every impact."""
        return int(np.searchsorted(self.p_edges, p, side="right")) + 1

    def loss_band(self, loss: float) -> int:
        return int(np.searchsorted(self.loss_edges, loss, side="right")) + 1

    def score(self, m: FailureMode) -> int:
        """The universal formula: likelihood band times impact band."""
        return self.p_band(m.probability) * self.loss_band(m.loss)

    def cell(self, m: FailureMode) -> Tuple[int, int]:
        return (self.p_band(m.probability), self.loss_band(m.loss))


SCALES: Tuple[Scale, ...] = (
    Scale("corporate-5x5",
          (0.05, 0.20, 0.50, 0.80),
          (50_000, 250_000, 1_000_000, 5_000_000),
          "even-ish probability bands, order-of-magnitude impact bands"),
    Scale("audit-5x5",
          (0.10, 0.25, 0.50, 0.75),
          (100_000, 500_000, 2_000_000, 10_000_000),
          "the same shape with the bands drawn one notch differently"),
)

SCALES_BY_NAME = {s.name: s for s in SCALES}


# --------------------------------------------------------------------------
# Cox's result, measured
# --------------------------------------------------------------------------


def inversions(scale: Scale, modes: Sequence[FailureMode] = MODES) -> List[Tuple[str, str, float]]:
    """Pairs the matrix ranks the wrong way round.

    An inversion is a pair where the matrix score of A is strictly greater
    than that of B, while A's expected loss is strictly smaller. The matrix
    is not being imprecise here; it is being wrong about the direction.
    """
    out = []
    for a, b in itertools.combinations(modes, 2):
        sa, sb = scale.score(a), scale.score(b)
        if sa == sb:
            continue
        hi, lo = (a, b) if sa > sb else (b, a)
        if hi.expected_loss < lo.expected_loss:
            out.append((hi.id, lo.id, lo.expected_loss / hi.expected_loss))
    return sorted(out, key=lambda r: -r[2])


def ranking_quality(scale: Scale, modes: Sequence[FailureMode] = MODES) -> Dict[str, float]:
    """How well the matrix reproduces the ordering it is a proxy for."""
    pairs = list(itertools.combinations(modes, 2))
    ranked = [(a, b) for a, b in pairs if scale.score(a) != scale.score(b)]
    inv = len(inversions(scale, modes))
    tied = len(pairs) - len(ranked)
    return {
        "pairs": len(pairs),
        "ordered_by_matrix": len(ranked),
        "tied_by_matrix": tied,
        "inversions": inv,
        "inversion_rate": inv / len(ranked) if ranked else 0.0,
        "undecided_rate": tied / len(pairs),
    }


def range_compression(scale: Scale, modes: Sequence[FailureMode] = MODES) -> Dict[str, object]:
    """How much true risk collapses into one matrix cell.

    Two failure modes sharing a cell are, to every downstream reader,
    the same risk.
    """
    cells: Dict[Tuple[int, int], List[FailureMode]] = {}
    for m in modes:
        cells.setdefault(scale.cell(m), []).append(m)
    worst_cell, worst_ratio, worst_pair = None, 1.0, None
    for cell, group in cells.items():
        if len(group) < 2:
            continue
        lo = min(group, key=lambda m: m.expected_loss)
        hi = max(group, key=lambda m: m.expected_loss)
        ratio = hi.expected_loss / lo.expected_loss
        if ratio > worst_ratio:
            worst_cell, worst_ratio, worst_pair = cell, ratio, (hi.id, lo.id)
    return {
        "occupied_cells": len(cells),
        "shared_cells": sum(1 for g in cells.values() if len(g) > 1),
        "worst_cell": worst_cell,
        "worst_ratio": worst_ratio,
        "worst_pair": worst_pair,
    }


def scale_disagreement(modes: Sequence[FailureMode] = MODES) -> Dict[str, object]:
    """Two conventional scales, one set of risks, two different orders."""
    a, b = SCALES
    flips = []
    for x, y in itertools.combinations(modes, 2):
        ax, ay = a.score(x), a.score(y)
        bx, by = b.score(x), b.score(y)
        if ax == ay or bx == by:
            continue
        if (ax > ay) != (bx > by):
            flips.append((x.id, y.id))
    top_a = max(modes, key=lambda m: (a.score(m), m.expected_loss)).id
    top_b = max(modes, key=lambda m: (b.score(m), m.expected_loss)).id
    return {"flips": flips, "n_flips": len(flips),
            "top_by_a": top_a, "top_by_b": top_b,
            "same_top": top_a == top_b}


def ordinal_product_is_meaningless(scale: Scale) -> Dict[str, object]:
    """The arithmetic itself: band numbers are labels, not quantities.

    Band 4 is not twice band 2. Multiplying two ordinal labels produces a
    number with no unit, and the score's implied indifference curves are an
    artefact of where the bands were drawn.
    """
    equal_score = {}
    for p_band in range(1, len(scale.p_edges) + 2):
        for l_band in range(1, len(scale.loss_edges) + 2):
            equal_score.setdefault(p_band * l_band, []).append((p_band, l_band))
    collisions = {k: v for k, v in equal_score.items() if len(v) > 1}
    return {
        "distinct_scores": len(equal_score),
        "cells": (len(scale.p_edges) + 1) * (len(scale.loss_edges) + 1),
        "colliding_scores": len(collisions),
        "example": (12, equal_score.get(12, [])),
    }


# --------------------------------------------------------------------------
# What to actually do on Monday
# --------------------------------------------------------------------------


def by_expected_loss(modes: Sequence[FailureMode] = MODES) -> List[FailureMode]:
    return sorted(modes, key=lambda m: -m.expected_loss)


def by_matrix(scale: Scale, modes: Sequence[FailureMode] = MODES) -> List[FailureMode]:
    return sorted(modes, key=lambda m: (-scale.score(m), -m.expected_loss))


def by_prevention_value(modes: Sequence[FailureMode] = MODES) -> List[FailureMode]:
    """Expected loss avoided minus what avoiding it costs.

    This is the only one of the three orderings that answers the question
    the meeting is actually held to answer, because it is the only one that
    knows what prevention costs.
    """
    return sorted(modes, key=lambda m: -m.prevention_value)


def by_prevention_ratio(modes: Sequence[FailureMode] = MODES) -> List[FailureMode]:
    """Loss avoided per unit spent - the order to work down under a budget."""
    return sorted(modes, key=lambda m: -m.prevention_ratio)


def ordering_disagreement(scale: Optional[Scale] = None,
                          modes: Sequence[FailureMode] = MODES) -> Dict[str, object]:
    scale = scale or SCALES[0]
    m_order = [m.id for m in by_matrix(scale, modes)]
    v_order = [m.id for m in by_prevention_value(modes)]
    e_order = [m.id for m in by_expected_loss(modes)]
    top_moves = [
        (mid, m_order.index(mid), v_order.index(mid))
        for mid in (m.id for m in modes)
    ]
    biggest = max(top_moves, key=lambda r: abs(r[1] - r[2]))
    return {
        "matrix_top3": m_order[:3],
        "expected_loss_top3": e_order[:3],
        "prevention_top3": v_order[:3],
        "matrix_equals_prevention": m_order == v_order,
        "biggest_move": biggest,
        "negative_value_modes": [m.id for m in modes if m.prevention_value < 0],
    }


def budget_allocation(budget: float, modes: Sequence[FailureMode] = MODES) -> Dict[str, object]:
    """Spend a fixed budget two ways and compare the loss avoided.

    Greedy by ratio against greedy by the matrix's own priority order.
    """
    def spend(order: Sequence[FailureMode]) -> Tuple[List[str], float, float]:
        left, bought, avoided = budget, [], 0.0
        for m in order:
            if m.prevention_cost <= left and m.prevention_value > 0:
                left -= m.prevention_cost
                bought.append(m.id)
                avoided += m.probability * m.prevention_effect * m.loss
        return bought, avoided, budget - left

    by_ratio = spend(by_prevention_ratio(modes))
    by_mat = spend(by_matrix(SCALES[0], modes))
    return {
        "budget": budget,
        "ratio_bought": by_ratio[0], "ratio_avoided": by_ratio[1], "ratio_spent": by_ratio[2],
        "matrix_bought": by_mat[0], "matrix_avoided": by_mat[1], "matrix_spent": by_mat[2],
        "advantage": by_ratio[1] - by_mat[1],
    }


def optimal_allocation(budget: float, modes: Sequence[FailureMode] = MODES) -> Dict[str, object]:
    """The exact best set of preventions for the money. Brute force, 2^14.

    This is the point the ordering exercise misses entirely. "Which risk is
    top?" is the wrong question, because prevention is bought under a
    budget - and choosing a set under a budget is a knapsack, not a sort.
    No ranking, however well constructed, is guaranteed to find the best
    set: greedy-by-ratio can lose to the matrix at a tight budget, and both
    lose to the exact answer.
    """
    n = len(modes)
    best_set, best_avoided = (), 0.0
    for mask in range(1 << n):
        cost = avoided = 0.0
        for i in range(n):
            if mask >> i & 1:
                m = modes[i]
                cost += m.prevention_cost
                if cost > budget:
                    break
                avoided += m.probability * m.prevention_effect * m.loss
        else:
            if cost <= budget and avoided > best_avoided:
                best_avoided, best_set = avoided, tuple(
                    modes[i].id for i in range(n) if mask >> i & 1)
    return {"budget": budget, "bought": list(best_set), "avoided": best_avoided}


def allocation_comparison(budgets: Sequence[float] = (50_000, 100_000, 150_000, 200_000),
                          modes: Sequence[FailureMode] = MODES) -> List[Dict[str, object]]:
    """Matrix order vs ratio order vs the exact answer, at each budget."""
    rows = []
    for b in budgets:
        greedy = budget_allocation(b, modes)
        best = optimal_allocation(b, modes)
        rows.append({
            "budget": b,
            "matrix": greedy["matrix_avoided"],
            "ratio": greedy["ratio_avoided"],
            "optimal": best["avoided"],
            "matrix_shortfall": best["avoided"] - greedy["matrix_avoided"],
            "ratio_shortfall": best["avoided"] - greedy["ratio_avoided"],
            "ratio_beats_matrix": greedy["ratio_avoided"] > greedy["matrix_avoided"],
            "optimal_set": best["bought"],
        })
    return rows


# --------------------------------------------------------------------------
# The record a pre-mortem should leave behind
# --------------------------------------------------------------------------

REQUIRED_FIELDS = ("cause", "probability", "loss", "prevention_cost", "prevention_effect")


def lint(mode: Dict[str, object]) -> Dict[str, bool]:
    """A failure mode that cannot be acted on, field by field.

    A cause without a probability cannot be ranked. Without a loss it
    cannot be compared. Without a prevention cost and effect it cannot be
    decided - which is the only reason the meeting happened.
    """
    p = mode.get("probability")
    eff = mode.get("prevention_effect")
    return {
        "has_mechanism": bool(mode.get("cause")) and len(str(mode.get("cause", ""))) > 25,
        "has_probability": isinstance(p, (int, float)) and 0.0 < float(p) < 1.0,
        "has_loss": isinstance(mode.get("loss"), (int, float)) and float(mode["loss"]) > 0,
        "has_prevention_cost": isinstance(mode.get("prevention_cost"), (int, float)),
        "has_prevention_effect": isinstance(eff, (int, float)) and 0.0 < float(eff) <= 1.0,
    }


def actionable(mode: Dict[str, object]) -> bool:
    return all(lint(mode).values())


#: How a pre-mortem's output usually arrives, before anyone asks for numbers.
RAW_NOTES: Tuple[Dict[str, object], ...] = (
    {"cause": "Data quality issues"},
    {"cause": "Metric definitions silently differ between the two engines",
     "probability": 0.55, "loss": 400_000, "prevention_cost": 25_000,
     "prevention_effect": 0.70},
    {"cause": "Key person risk", "probability": 0.18},
    {"cause": "The one engineer who knows the legacy warehouse leaves mid-migration",
     "probability": 0.18, "loss": 900_000, "prevention_cost": 40_000,
     "prevention_effect": 0.55},
    {"cause": "Budget overrun", "probability": 0.3, "loss": 600_000},
    {"cause": "Query costs land 3x over budget and finance halts the project",
     "probability": 0.30, "loss": 600_000, "prevention_cost": 15_000,
     "prevention_effect": 0.80},
    {"cause": "Scope creep"},
    {"cause": "Timezone handling changes and every daily report shifts by a day",
     "probability": 0.40, "loss": 120_000, "prevention_cost": 6_000,
     "prevention_effect": 0.85},
)


def notes_report() -> Dict[str, object]:
    checks = [lint(n) for n in RAW_NOTES]
    return {
        "n": len(RAW_NOTES),
        "actionable": sum(1 for n in RAW_NOTES if actionable(n)),
        "per_field": {f: sum(1 for c in checks if c[f]) for f in
                      ("has_mechanism", "has_probability", "has_loss",
                       "has_prevention_cost", "has_prevention_effect")},
        "vague": [str(n["cause"]) for n in RAW_NOTES if not lint(n)["has_mechanism"]],
    }


@lru_cache(maxsize=1)
def summary() -> Dict[str, object]:
    scale = SCALES[0]
    return {
        "plan_success": plan_success(),
        "weakest_step": weakest_step_success(),
        "steps_to_coin_flip": steps_to_coin_flip(),
        "expected_loss": total_expected_loss(),
        "ranking": ranking_quality(scale),
        "compression": range_compression(scale),
        "disagreement": scale_disagreement(),
    }
