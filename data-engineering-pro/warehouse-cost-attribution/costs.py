"""A warehouse invoice is a JOINT cost, and a joint cost has no unique owner.

The month below is simulated so the true structure is known. Six teams share ten tables,
three upstream models and one reservation. Three things make the cost joint rather than
separable, and all three are ordinary:

  1. CACHE       - the second read of a table on a given day is nearly free, so what a
                   query costs depends on who else ran it;
  2. SHARED BUILD- an upstream model is built once and consumed by everyone who needs it;
  3. RESERVATION - a fixed floor that exists the moment anybody uses the warehouse at all.

Everything downstream is derived from one characteristic function, `coalition_cost(S)`:
what the invoice WOULD have been if only the teams in S existed. With six teams that is
64 subsets and 720 orderings, so Shapley values and the core are computed exactly rather
than sampled.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------------------
# Rates. Roughly BigQuery on-demand plus a reservation.
# --------------------------------------------------------------------------------------

SCAN_RATE = 5.00 / 1024      # $ per GB scanned ($5/TB)
STORAGE_RATE = 0.020         # $ per GB-month
CACHE_RATE = 0.02            # a cached re-read costs 2% of a scan
RESERVED_FLOOR = 5_200.00    # $/month for the reservation, owed the moment anyone uses it
WORKDAYS = 22
COLD_SLOTS_PER_DAY = 4       # distinct query shapes per table per day that miss the cache


@dataclass(frozen=True)
class Table:
    name: str
    scan_gb: float
    storage_gb: float


@dataclass(frozen=True)
class Model:
    """An upstream dbt model. Built once per day if ANY consumer needs it."""
    name: str
    build_gb: float
    consumers: FrozenSet[str]


@dataclass(frozen=True)
class Team:
    name: str
    reads: Dict[str, int]        # table -> reads per month
    owned: bool = True           # False = nobody claims it (scheduled jobs)
    label: str = ""


TABLES: Dict[str, Table] = {t.name: t for t in [
    Table("events_raw",        9800.0, 214_000.0),
    Table("sessions",          3100.0,  61_000.0),
    Table("orders",             620.0,  14_500.0),
    Table("customers",          210.0,   4_800.0),
    Table("subscriptions",      340.0,   7_100.0),
    Table("marketing_touch",   4200.0,  78_000.0),
    Table("ledger",             150.0,   3_900.0),
    Table("feature_store",     6400.0, 131_000.0),
    Table("support_tickets",    280.0,   5_200.0),
    Table("experiment_assign", 2600.0,  47_000.0),
]}

MODELS: List[Model] = [
    Model("dim_customer",  1400.0, frozenset({"analytics", "growth", "finance", "exec_reporting"})),
    Model("fct_orders",    2100.0, frozenset({"analytics", "finance", "exec_reporting"})),
    Model("user_features", 5300.0, frozenset({"ml_platform", "growth"})),
]

TEAMS: List[Team] = [
    Team("analytics", {"events_raw": 260, "sessions": 180, "orders": 140,
                       "customers": 90, "experiment_assign": 60},
         label="ad-hoc analysis and dashboards"),
    Team("growth", {"events_raw": 310, "marketing_touch": 220, "sessions": 120,
                    "experiment_assign": 190},
         label="experimentation and campaigns"),
    Team("finance", {"orders": 110, "ledger": 240, "subscriptions": 130, "customers": 40},
         label="revenue and close"),
    Team("ml_platform", {"feature_store": 340, "events_raw": 150, "sessions": 90},
         label="training and feature pipelines"),
    Team("exec_reporting", {"orders": 30, "customers": 25, "subscriptions": 20,
                            "ledger": 15},
         label="the weekly board pack"),
    Team("scheduled_unowned", {"events_raw": 120, "marketing_touch": 130,
                               "support_tickets": 95, "feature_store": 60},
         owned=False, label="jobs whose owner has left"),
]

TEAM_NAMES: List[str] = [t.name for t in TEAMS]
TEAM_BY_NAME: Dict[str, Team] = {t.name: t for t in TEAMS}


# --------------------------------------------------------------------------------------
# The characteristic function. Everything else is derived from this one function.
# --------------------------------------------------------------------------------------

def _reads_by_table(members: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for name in members:
        for table, n in TEAM_BY_NAME[name].reads.items():
            out[table] = out.get(table, 0) + n
    return out


def coalition_cost(members: Sequence[str]) -> float:
    """The invoice this coalition would have generated on its own.

    Not additive, by construction: a table read by two teams is scanned once a day and
    cached thereafter, its storage is paid once, and a shared model is built once.
    """
    members = list(members)
    if not members:
        return 0.0

    total = RESERVED_FLOOR
    for table, reads in _reads_by_table(members).items():
        t = TABLES[table]
        days_touched = min(reads, WORKDAYS * COLD_SLOTS_PER_DAY)  # the rest hit the cache
        cached = reads - days_touched                 # every other read is nearly free
        total += days_touched * t.scan_gb * SCAN_RATE
        total += cached * t.scan_gb * SCAN_RATE * CACHE_RATE
        total += t.storage_gb * STORAGE_RATE          # stored because SOMEBODY needs it

    live = set(members)
    for m in MODELS:
        if m.consumers & live:                        # built once, for whoever needs it
            total += WORKDAYS * m.build_gb * SCAN_RATE
    return total


INVOICE = coalition_cost(TEAM_NAMES)


def all_coalitions() -> List[Tuple[str, ...]]:
    out: List[Tuple[str, ...]] = []
    for k in range(len(TEAM_NAMES) + 1):
        out.extend(itertools.combinations(TEAM_NAMES, k))
    return out


COALITION_COST: Dict[FrozenSet[str], float] = {
    frozenset(c): coalition_cost(c) for c in all_coalitions()
}


def v(S: Sequence[str]) -> float:
    return COALITION_COST[frozenset(S)]


# --------------------------------------------------------------------------------------
# Attribution methods. Each is a sentence somebody says in a cost review.
# --------------------------------------------------------------------------------------

def _normalise(shares: Dict[str, float], total: float = None) -> Dict[str, float]:
    """Scale an allocation so it adds to the invoice. Every method must bill the bill."""
    total = INVOICE if total is None else total
    s = sum(shares.values())
    if s <= 0:
        return {k: total / len(shares) for k in shares}
    return {k: val * total / s for k, val in shares.items()}


def method_direct_bytes() -> Dict[str, float]:
    """'Bill each team the bytes its own queries scanned.' Ignores that most were cached."""
    return _normalise({t.name: sum(TABLES[tb].scan_gb * n for tb, n in t.reads.items())
                       for t in TEAMS})


def method_query_count() -> Dict[str, float]:
    """'Split it by number of queries.' The rule a spreadsheet reaches for first."""
    return _normalise({t.name: float(sum(t.reads.values())) for t in TEAMS})


def method_equal_split() -> Dict[str, float]:
    """'Six teams, six ways.'"""
    return {n: INVOICE / len(TEAM_NAMES) for n in TEAM_NAMES}


def method_standalone() -> Dict[str, float]:
    """'What would you have cost on your own?' Over-recovers, because sharing is real."""
    return _normalise({n: v([n]) for n in TEAM_NAMES})


def method_marginal() -> Dict[str, float]:
    """'What would we save if you stopped?' Under-recovers, for exactly the same reason."""
    return _normalise({n: INVOICE - v([m for m in TEAM_NAMES if m != n]) for n in TEAM_NAMES})


def raw_marginal() -> Dict[str, float]:
    """Marginal cost BEFORE normalising -- the number that does not add up to the bill."""
    return {n: INVOICE - v([m for m in TEAM_NAMES if m != n]) for n in TEAM_NAMES}


def raw_standalone() -> Dict[str, float]:
    return {n: v([n]) for n in TEAM_NAMES}


def method_first_toucher() -> Dict[str, float]:
    """'Whoever ran the query that warmed the cache pays for the scan.'

    Teams are ordered by name, which is exactly as principled as ordering them by who
    happens to run at 06:00.
    """
    shares = {n: 0.0 for n in TEAM_NAMES}
    order = sorted(TEAM_NAMES)
    for table, t in TABLES.items():
        readers = [(n, TEAM_BY_NAME[n].reads.get(table, 0)) for n in order]
        readers = [(n, r) for n, r in readers if r > 0]
        if not readers:
            continue
        reads = sum(r for _, r in readers)
        days = min(reads, WORKDAYS * COLD_SLOTS_PER_DAY)
        scan_cost = days * t.scan_gb * SCAN_RATE + t.storage_gb * STORAGE_RATE
        cache_cost = (reads - days) * t.scan_gb * SCAN_RATE * CACHE_RATE
        shares[readers[0][0]] += scan_cost
        for n, r in readers:
            shares[n] += cache_cost * r / reads
    for m in MODELS:
        live = sorted(m.consumers)
        shares[live[0]] += WORKDAYS * m.build_gb * SCAN_RATE
    for n in TEAM_NAMES:
        shares[n] += RESERVED_FLOOR / len(TEAM_NAMES)
    return _normalise(shares)


def sampled_shapley(draws: int, seed: int = 0) -> Dict[str, float]:
    """Shapley by Monte Carlo over orderings, for the case where 2^n is out of reach."""
    rng = np.random.default_rng(seed)
    phi = {name: 0.0 for name in TEAM_NAMES}
    for _ in range(draws):
        running: List[str] = []
        prev = 0.0
        for name in rng.permutation(TEAM_NAMES):
            running.append(str(name))
            cur = coalition_cost(running)
            phi[str(name)] += cur - prev
            prev = cur
    return {k: val / draws for k, val in phi.items()}


def shapley() -> Dict[str, float]:
    """The average marginal contribution over all 720 orderings. Computed, not sampled.

    Uniquely satisfies efficiency, symmetry, the dummy axiom and additivity -- which is a
    strong claim about the METHOD and, as section 6 shows, a much weaker one about whether
    anybody would accept the bill.
    """
    n = len(TEAM_NAMES)
    phi = {name: 0.0 for name in TEAM_NAMES}
    for order in itertools.permutations(TEAM_NAMES):
        running: List[str] = []
        prev = 0.0
        for name in order:
            running.append(name)
            cur = v(running)
            phi[name] += cur - prev
            prev = cur
    fact = float(math.factorial(n))
    return {k: val / fact for k, val in phi.items()}


METHODS = {
    "direct_bytes": method_direct_bytes,
    "query_count": method_query_count,
    "equal_split": method_equal_split,
    "standalone": method_standalone,
    "marginal": method_marginal,
    "first_toucher": method_first_toucher,
    "shapley": shapley,
}


# --------------------------------------------------------------------------------------
# The core. Not one allocation -- the SET of allocations no coalition would walk out of.
# --------------------------------------------------------------------------------------

def _core_constraints() -> Tuple[np.ndarray, np.ndarray]:
    """x(S) <= v(S) for every proper coalition. Nobody pays more than going it alone."""
    idx = {n: i for i, n in enumerate(TEAM_NAMES)}
    rows, rhs = [], []
    for S in all_coalitions():
        if not S or len(S) == len(TEAM_NAMES):
            continue
        row = np.zeros(len(TEAM_NAMES))
        for n in S:
            row[idx[n]] = 1.0
        rows.append(row)
        rhs.append(v(S))
    return np.array(rows), np.array(rhs)


def core_is_nonempty() -> bool:
    A, b = _core_constraints()
    res = linprog(c=np.zeros(len(TEAM_NAMES)), A_ub=A, b_ub=b,
                  A_eq=np.ones((1, len(TEAM_NAMES))), b_eq=[INVOICE],
                  bounds=[(0, None)] * len(TEAM_NAMES), method="highs")
    return bool(res.success)


def core_range(team: str) -> Tuple[Optional[float], Optional[float]]:
    """The least and most this team could pay across every allocation in the core."""
    A, b = _core_constraints()
    i = TEAM_NAMES.index(team)
    c = np.zeros(len(TEAM_NAMES))
    c[i] = 1.0
    lo = linprog(c=c, A_ub=A, b_ub=b, A_eq=np.ones((1, len(TEAM_NAMES))), b_eq=[INVOICE],
                 bounds=[(0, None)] * len(TEAM_NAMES), method="highs")
    hi = linprog(c=-c, A_ub=A, b_ub=b, A_eq=np.ones((1, len(TEAM_NAMES))), b_eq=[INVOICE],
                 bounds=[(0, None)] * len(TEAM_NAMES), method="highs")
    if not (lo.success and hi.success):
        return None, None
    return float(lo.x[i]), float(hi.x[i])


def in_core(alloc: Dict[str, float], tol: float = 1e-6) -> bool:
    if abs(sum(alloc.values()) - INVOICE) > 1e-4:
        return False
    for S in all_coalitions():
        if not S or len(S) == len(TEAM_NAMES):
            continue
        if sum(alloc[n] for n in S) > v(S) + tol:
            return False
    return True


def core_violations(alloc: Dict[str, float], tol: float = 1e-6) -> List[Tuple[Tuple[str, ...], float]]:
    """Which coalitions would object, and by how much."""
    out = []
    for S in all_coalitions():
        if not S or len(S) == len(TEAM_NAMES):
            continue
        excess = sum(alloc[n] for n in S) - v(S)
        if excess > tol:
            out.append((S, excess))
    return sorted(out, key=lambda r: -r[1])


# --------------------------------------------------------------------------------------
# What cannot be attributed to anybody at all.
# --------------------------------------------------------------------------------------

def unowned_cost() -> float:
    """The bill that disappears if the jobs nobody claims are switched off."""
    owned = [t.name for t in TEAMS if t.owned]
    return INVOICE - v(owned)


def reservation_share() -> float:
    """The floor. Owed by everyone jointly and by nobody in particular."""
    return RESERVED_FLOOR


def unattributable() -> Dict[str, float]:
    return {"unowned_jobs": unowned_cost(), "reservation_floor": reservation_share()}


# --------------------------------------------------------------------------------------
# Cache order: the same query, priced by who happened to run first.
# --------------------------------------------------------------------------------------

def query_cost_by_position(table: str, position: int) -> float:
    """Cost charged to the n-th read of a table on one day, under direct-bytes billing."""
    t = TABLES[table]
    full = t.scan_gb * SCAN_RATE
    return full if position == 0 else full * CACHE_RATE


def cache_ratio(table: str) -> float:
    return 1.0 / CACHE_RATE


@dataclass
class Allocation:
    name: str
    shares: Dict[str, float] = field(default_factory=dict)
