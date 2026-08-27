"""Cost of delay: an ordering is not a schedule.

Every prioritisation method emits an *order*. What a business actually pays is the
delay cost incurred by the *schedule* that order produces. This module models both
so the gap between them can be measured rather than argued about.

Nothing here is stochastic except `noise_sweep`, which seeds its own generator.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HORIZON = 40.0  # weeks; the planning window everything is quoted against


# ---------------------------------------------------------------- cost of delay


@dataclass(frozen=True)
class CoD:
    """Cost of delay as a *rate over time*, which is what it actually is.

    `kind` is one of:
      linear   - a constant rate r. The only shape WSJF's arithmetic assumes.
      deadline - nothing until t_break, then r2 per week. A fixed date.
      step     - r per week until t_break, then r2. An escalating clause.
      window   - r at t=0 decaying with constant tau. A market opportunity.

    `cum(c)` is the total cost of finishing at week c: the integral of the rate
    from 0 to c. That integral, not the rate, is what appears on the P&L.
    """

    kind: str
    r: float = 0.0
    r2: float = 0.0
    tau: float = 0.0
    t_break: float = 0.0

    def rate(self, t: float) -> float:
        if self.kind == "linear":
            return self.r
        if self.kind == "deadline":
            return 0.0 if t < self.t_break else self.r2
        if self.kind == "step":
            return self.r if t < self.t_break else self.r2
        if self.kind == "window":
            return self.r * math.exp(-t / self.tau)
        raise ValueError(self.kind)

    def cum(self, c: float) -> float:
        if c <= 0.0:
            return 0.0
        if self.kind == "linear":
            return self.r * c
        if self.kind == "deadline":
            return self.r2 * max(0.0, c - self.t_break)
        if self.kind == "step":
            return self.r * min(c, self.t_break) + self.r2 * max(0.0, c - self.t_break)
        if self.kind == "window":
            return self.r * self.tau * (1.0 - math.exp(-c / self.tau))
        raise ValueError(self.kind)

    def mean_rate(self, horizon: float = HORIZON) -> float:
        """The rate a team would quote if asked to average over the window."""
        return self.cum(horizon) / horizon

    def peak_rate(self, horizon: float = HORIZON) -> float:
        """The worst week in the window - the number an anxious team quotes."""
        return max(self.rate(t) for t in _grid(horizon))


def _grid(horizon: float, n: int = 401) -> List[float]:
    return [horizon * i / (n - 1) for i in range(n)]


# ------------------------------------------------------------------- the backlog


@dataclass(frozen=True)
class Item:
    key: str
    name: str
    duration: float       # calendar weeks for one team - what delay is paid in
    person_weeks: float   # effort - what an estimate is usually given in
    cod: CoD
    reach: float
    impact: float
    confidence: float

    @property
    def effort_pm(self) -> float:
        """RICE effort, person-months."""
        return self.person_weeks / 4.0

    @property
    def rice(self) -> float:
        return self.reach * self.impact * self.confidence / self.effort_pm

    @property
    def rice_duration_denominator(self) -> float:
        """Same numerator, but divided by calendar duration instead of effort."""
        return self.reach * self.impact * self.confidence / (self.duration / 4.0)


def backlog() -> Dict[str, Item]:
    """One quarter of a real-looking B2B SaaS backlog. 9 items, 40 weeks of work."""
    raw = [
        # key   name                 dur  pw   cod                                                       reach impact conf
        ("A", "sso-saml",            6,  12, CoD("linear", r=38.0),                                        400, 3.0, 0.90),
        ("B", "soc2-evidence",       4,   4, CoD("deadline", r2=180.0, t_break=26.0),                      150, 2.0, 0.80),
        ("C", "usage-billing",       8,   8, CoD("linear", r=52.0),                                         900, 2.0, 0.80),
        ("D", "onboarding-revamp",   3,   6, CoD("window", r=70.0, tau=10.0),                             2000, 1.0, 0.90),
        ("E", "api-rate-limits",     2,   2, CoD("linear", r=9.0),                                          300, 0.5, 1.00),
        ("F", "data-export",         1,   1, CoD("linear", r=6.0),                                          250, 0.5, 1.00),
        ("G", "mobile-push",         5,  10, CoD("window", r=30.0, tau=25.0),                              1500, 1.0, 0.60),
        ("H", "audit-log",           4,   4, CoD("step", r=5.0, r2=45.0, t_break=20.0),                     200, 1.0, 0.90),
        ("I", "search-rebuild",      7,   7, CoD("linear", r=22.0),                                        1800, 2.0, 0.50),
    ]
    return {
        k: Item(k, n, float(d), float(pw), cod, float(re), float(im), float(cf))
        for (k, n, d, pw, cod, re, im, cf) in raw
    }


def linearised(items: Dict[str, Item]) -> Dict[str, Item]:
    """The same backlog with every shape replaced by a constant rate of the same
    total value over the horizon. This is the world WSJF's theorem lives in."""
    out = {}
    for k, it in items.items():
        out[k] = Item(
            it.key, it.name, it.duration, it.person_weeks,
            CoD("linear", r=it.cod.mean_rate()),
            it.reach, it.impact, it.confidence,
        )
    return out


PRECEDENCE: Tuple[Tuple[str, str], ...] = (
    ("H", "B"),  # you cannot evidence an audit trail you have not built
    ("E", "C"),  # metering has to hold up before it can be billed on
)


# ------------------------------------------------------------------- scheduling


def completions(order: Sequence[str], items: Dict[str, Item]) -> Dict[str, float]:
    """Single team, no interruption: week each item is done."""
    t = 0.0
    out: Dict[str, float] = {}
    for k in order:
        t += items[k].duration
        out[k] = t
    return out


def cost_of(order: Sequence[str], items: Dict[str, Item]) -> float:
    c = completions(order, items)
    return sum(items[k].cod.cum(c[k]) for k in order)


def cost_breakdown(order: Sequence[str], items: Dict[str, Item]) -> Dict[str, float]:
    c = completions(order, items)
    return {k: items[k].cod.cum(c[k]) for k in order}


def parallel_schedule(order: Sequence[str], items: Dict[str, Item], teams: int
                      ) -> Dict[str, float]:
    """List-scheduling: walk the order, give each item to the team free soonest."""
    free = [0.0] * teams
    out: Dict[str, float] = {}
    for k in order:
        m = min(range(teams), key=lambda i: (free[i], i))
        free[m] += items[k].duration
        out[k] = free[m]
    return out


def parallel_cost(order: Sequence[str], items: Dict[str, Item], teams: int) -> float:
    c = parallel_schedule(order, items, teams)
    return sum(items[k].cod.cum(c[k]) for k in order)


# ------------------------------------------------------------------- orderings


def _rank(items: Dict[str, Item], score: Callable[[Item], float]) -> List[str]:
    """Descending by score, ties broken by key so the result is deterministic."""
    return sorted(items, key=lambda k: (-score(items[k]), k))


def order_cd3_initial(items: Dict[str, Item]) -> List[str]:
    """WSJF / CD3 with the cost of delay elicited as 'what does a week cost us
    right now'. This is the number a room actually produces."""
    return _rank(items, lambda it: it.cod.rate(0.0) / it.duration)


def order_cd3_mean(items: Dict[str, Item]) -> List[str]:
    """The same named method, cost of delay averaged over the planning window."""
    return _rank(items, lambda it: it.cod.mean_rate() / it.duration)


def order_cd3_peak(items: Dict[str, Item]) -> List[str]:
    """The same named method, cost of delay taken at its worst week."""
    return _rank(items, lambda it: it.cod.peak_rate() / it.duration)


def order_rice(items: Dict[str, Item]) -> List[str]:
    return _rank(items, lambda it: it.rice)


def order_rice_duration(items: Dict[str, Item]) -> List[str]:
    return _rank(items, lambda it: it.rice_duration_denominator)


def order_value_first(items: Dict[str, Item]) -> List[str]:
    """Highest total cost of delay over the horizon. No denominator at all."""
    return _rank(items, lambda it: it.cod.cum(HORIZON))


def order_shortest_first(items: Dict[str, Item]) -> List[str]:
    return _rank(items, lambda it: -it.duration)


def order_effort_first(items: Dict[str, Item]) -> List[str]:
    return _rank(items, lambda it: -it.person_weeks)


def order_hippo(items: Dict[str, Item]) -> List[str]:
    """The order the loudest stakeholder in the room asked for: the two things a
    customer complained about last week, then the visible redesign, then the rest."""
    stated = ["G", "D", "I", "A", "C", "F", "E", "H", "B"]
    return [k for k in stated if k in items] + sorted(set(items) - set(stated))


ORDERINGS: Dict[str, Callable[[Dict[str, Item]], List[str]]] = {
    "cd3_initial": order_cd3_initial,
    "cd3_mean": order_cd3_mean,
    "cd3_peak": order_cd3_peak,
    "rice": order_rice,
    "rice_duration": order_rice_duration,
    "value_first": order_value_first,
    "shortest_first": order_shortest_first,
    "effort_first": order_effort_first,
    "hippo": order_hippo,
}


# ---------------------------------------------------------- exhaustive optimum


def _feasible_positions(perm: Sequence[int], keys: Sequence[str],
                        edges: Sequence[Tuple[str, str]]) -> bool:
    pos = {keys[i]: p for p, i in enumerate(perm)}
    return all(pos[a] < pos[b] for a, b in edges)


def sweep(items: Dict[str, Item],
          edges: Optional[Sequence[Tuple[str, str]]] = None,
          teams: int = 1) -> Dict[str, object]:
    """Enumerate every ordering. Returns the best, the worst, and the exact mean
    over all feasible orderings - the mean *is* the random-baseline cost, computed
    rather than sampled."""
    keys = sorted(items)
    durs = [items[k].duration for k in keys]
    cums = [items[k].cod.cum for k in keys]
    n = len(keys)
    best = math.inf
    worst = -math.inf
    total = 0.0
    count = 0
    best_perm: Tuple[int, ...] = tuple(range(n))
    worst_perm: Tuple[int, ...] = tuple(range(n))
    for perm in itertools.permutations(range(n)):
        if edges and not _feasible_positions(perm, keys, edges):
            continue
        if teams == 1:
            t = 0.0
            tot = 0.0
            for i in perm:
                t += durs[i]
                tot += cums[i](t)
        else:
            free = [0.0] * teams
            tot = 0.0
            for i in perm:
                m = min(range(teams), key=lambda j: (free[j], j))
                free[m] += durs[i]
                tot += cums[i](free[m])
        total += tot
        count += 1
        if tot < best:
            best, best_perm = tot, perm
        if tot > worst:
            worst, worst_perm = tot, perm
    return {
        "best": best,
        "best_order": [keys[i] for i in best_perm],
        "worst": worst,
        "worst_order": [keys[i] for i in worst_perm],
        "mean": total / count,
        "count": count,
    }


def optimal_two_team_assignment(items: Dict[str, Item]) -> Dict[str, object]:
    """For a purely linear backlog, WSPT *within* a team is optimal, so the exact
    two-team optimum is a search over the 2**n assignments only."""
    for it in items.values():
        if it.cod.kind != "linear":
            raise ValueError("only exact for linear cost of delay")
    keys = sorted(items)
    n = len(keys)
    best = math.inf
    best_split: Tuple[List[str], List[str]] = ([], [])
    for mask in range(1 << n):
        groups: List[List[str]] = [[], []]
        for i, k in enumerate(keys):
            groups[(mask >> i) & 1].append(k)
        tot = 0.0
        for g in groups:
            g_sorted = sorted(g, key=lambda k: (-items[k].cod.r / items[k].duration, k))
            t = 0.0
            for k in g_sorted:
                t += items[k].duration
                tot += items[k].cod.cum(t)
        if tot < best:
            best = tot
            best_split = ([k for k in groups[0]], [k for k in groups[1]])
    return {"best": best, "split": best_split}


# -------------------------------------------------------------- estimate noise


def noise_sweep(items: Dict[str, Item], sigma: float, trials: int,
                seed: int = 20260827) -> Dict[str, object]:
    """Durations are estimates. Rank on the *estimate*, pay on the *truth*.

    Returns the realised-cost distribution and how often the ordering changed.
    """
    import random

    rng = random.Random(seed)
    truth_order = order_cd3_mean(items)
    truth_cost = cost_of(truth_order, items)
    costs: List[float] = []
    changed = 0
    for _ in range(trials):
        noisy = {}
        for k, it in items.items():
            f = math.exp(rng.gauss(0.0, sigma) - 0.5 * sigma * sigma)
            noisy[k] = Item(it.key, it.name, max(0.25, it.duration * f),
                            it.person_weeks, it.cod, it.reach, it.impact, it.confidence)
        o = order_cd3_mean(noisy)
        if o != truth_order:
            changed += 1
        costs.append(cost_of(o, items))  # ranked on noise, paid on truth
    costs_sorted = sorted(costs)
    return {
        "truth_order": truth_order,
        "truth_cost": truth_cost,
        "mean": sum(costs) / len(costs),
        "p50": costs_sorted[len(costs) // 2],
        "p90": costs_sorted[int(0.90 * (len(costs) - 1))],
        "max": costs_sorted[-1],
        "reorder_rate": changed / trials,
        "costs": costs,
    }


def kendall_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Number of pairs the two orderings disagree about."""
    pa = {k: i for i, k in enumerate(a)}
    pb = {k: i for i, k in enumerate(b)}
    keys = list(pa)
    d = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = keys[i], keys[j]
            if (pa[x] < pa[y]) != (pb[x] < pb[y]):
                d += 1
    return d


def repair_precedence(order: Sequence[str],
                      edges: Sequence[Tuple[str, str]]) -> List[str]:
    """The repair a team actually performs: walk the preferred order, take the
    highest-ranked item whose prerequisites are already done."""
    remaining = list(order)
    done: List[str] = []
    while remaining:
        for k in remaining:
            if all(a in done for a, b in edges if b == k):
                done.append(k)
                remaining.remove(k)
                break
        else:  # pragma: no cover - would mean a cycle
            raise ValueError("cyclic precedence")
    return done


def all_costs(items: Dict[str, Item],
              edges: Optional[Sequence[Tuple[str, str]]] = None) -> List[float]:
    """Every ordering's cost. The exact population, not a sample."""
    keys = sorted(items)
    durs = [items[k].duration for k in keys]
    cums = [items[k].cod.cum for k in keys]
    n = len(keys)
    out: List[float] = []
    for perm in itertools.permutations(range(n)):
        if edges and not _feasible_positions(perm, keys, edges):
            continue
        t = 0.0
        tot = 0.0
        for i in perm:
            t += durs[i]
            tot += cums[i](t)
        out.append(tot)
    return out


def percentile_of(costs: Sequence[float], value: float) -> float:
    """Fraction of orderings strictly cheaper than `value`. 0.88 means the method
    is beaten by 88% of the orderings you could have drawn out of a hat."""
    return sum(1 for c in costs if c < value) / len(costs)
