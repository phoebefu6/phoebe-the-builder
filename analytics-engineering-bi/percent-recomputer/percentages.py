"""Percentage tables that add up, and the reason no method makes them fair.

`round(value / total * 100, 1)` per row is the wrong shape, and this time not
because of the residual - Day 143 (`currency-rounder`) covered the residual and
named the row that absorbs it. The deeper problem is that turning shares into
displayable numbers is an **apportionment** problem, the same one as handing out
seats in a parliament, and apportionment has a proved impossibility at the centre
of it.

Balinski and Young (1982): no apportionment method can both stay within the quota
(every row gets either the floor or the ceiling of its exact share) and avoid the
Alabama paradox (adding one unit to the total taking a unit away from some row).
Not "no known method". No method. So the choice is not between a correct method
and buggy ones - it is between two named failure modes, and the only wrong move
is picking without knowing which one you took.

What that means for a percentage column
---------------------------------------
1. Rounding each row independently does not preserve the total. Three equal rows
   at one decimal place print 33.3 three times and the column reads 99.9. Every
   row is correctly rounded.
2. Every fix is a method with a name and a known bias. Largest remainder stays in
   quota and suffers the Alabama paradox. D'Hondt never suffers it and violates
   the upper quota, systematically in favour of large rows. Sainte-Laguë splits
   the difference. Huntington-Hill (the US House method) gives every row at least
   one unit, so it *cannot* print 0.0% for a row that exists.
3. Three failures are specific to percentages rather than seats, and none of them
   is a rounding question:
   * a denominator of 7 cannot produce 33.3% - only multiples of 100/7 exist, so
     one decimal place claims a sample you do not have
   * a grouped table cannot be consistent at both levels: rounded rows summing to
     rounded subtotals summing to a rounded grand total is over-determined
   * a share of a signed total (a P&L with a loss-making line) is not a share -
     it can exceed 100%, go negative, and reorder under a sign flip

So the return value is a verdict over every method, not a column of numbers:

    consistent - every method returns the same allocation
    residual   - the methods agree to within one unit and disagree about *which*
                 row holds it; the column adds up either way
    contested  - the methods differ by more than a unit, or a paradox or a quota
                 violation fires. Two defensible tables exist
    undefined  - there is no share to display: the base is zero, or signed, or
                 the precision asked for does not exist at this denominator

Standard library only: `math`, `dataclasses`, `enum`, `fractions`, `itertools`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One line of a table. `value` is a count, an amount, a population."""

    label: str
    value: float


@dataclass(frozen=True)
class Table:
    """A table plus what is being asked of it.

    `units` is the integer budget to hand out. For a percentage column at `d`
    decimal places that is `100 * 10**d` - so a 1 dp column is 1000 units of a
    tenth of a percent each, and every method below is an apportionment method
    whether or not it was written for seats.
    """

    name: str
    rows: Tuple[Row, ...]
    units: int
    kind: str = "percent"  # "percent" | "seats"
    decimals: int = 1
    group_of: Dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def total(self) -> float:
        return sum(r.value for r in self.rows)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(r.label for r in self.rows)

    def quotas(self) -> Tuple[Fraction, ...]:
        """Exact shares in units, as fractions - never floats.

        The quota is the only unarguable number in the whole file: it is what the
        row is owed before anybody rounds. Every method is judged against it.
        """
        t = Fraction(0)
        vals = [Fraction(str(r.value)) for r in self.rows]
        for v in vals:
            t += v
        if t == 0:
            return tuple(Fraction(0) for _ in vals)
        return tuple(v * self.units / t for v in vals)

    def is_definable(self) -> Tuple[bool, str]:
        """Whether a share exists at all, before any rounding question."""
        if not self.rows:
            return False, "no rows"
        if any(not math.isfinite(r.value) for r in self.rows):
            return False, "a value is not finite"
        if any(r.value < 0 for r in self.rows) and any(r.value > 0 for r in self.rows):
            return False, ("values have mixed signs: a share of a signed total is not a "
                           "share - it can exceed 100%, go negative, and reorder if the sign flips")
        if self.total == 0:
            return False, "the base is zero, so no row has a share"
        if self.total < 0:
            return False, "the base is negative, so every share is negative"
        return True, ""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class Verdict(Enum):
    CONSISTENT = "consistent"
    RESIDUAL = "residual"
    CONTESTED = "contested"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class Allocation:
    """One method's answer: an integer count of units per row."""

    method: str
    units: Tuple[int, ...]
    notes: Tuple[str, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.units)

    def percents(self, table: Table) -> Tuple[float, ...]:
        """Units rendered as percentages at the table's decimal place."""
        scale = 10 ** table.decimals
        return tuple(u / scale for u in self.units)

    def sums_to(self, table: Table) -> bool:
        return self.total == table.units


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str  # blocking | silent | advisory
    message: str
    method: Optional[str] = None


@dataclass(frozen=True)
class Audit:
    table: Table
    verdict: Verdict
    allocations: Tuple[Allocation, ...]
    findings: Tuple[Finding, ...]

    @property
    def by_method(self) -> Dict[str, Allocation]:
        return {a.method: a for a in self.allocations}

    def disagreeing_rows(self) -> Tuple[str, ...]:
        """Labels where two methods that both sum correctly differ."""
        good = [a for a in self.allocations if a.sums_to(self.table)]
        out = []
        for i, label in enumerate(self.table.labels):
            if len({a.units[i] for a in good}) > 1:
                out.append(label)
        return tuple(out)

    def max_row_gap(self) -> int:
        """Largest difference in units any single row sees across methods."""
        good = [a for a in self.allocations if a.sums_to(self.table)]
        if len(good) < 2:
            return 0
        return max(
            max(a.units[i] for a in good) - min(a.units[i] for a in good)
            for i in range(len(self.table.rows))
        )


# ---------------------------------------------------------------------------
# The methods
# ---------------------------------------------------------------------------
#
# Two families. The naive ones round each row on its own and are the only ones
# that can fail to sum. The rest are apportionment methods and always sum
# exactly - they differ in *who* they favour, which is not a rounding detail but
# a policy choice that nobody in a BI pipeline knows they are making.


def _shares(table: Table) -> List[Fraction]:
    return list(table.quotas())


def naive_half_up(table: Table) -> Allocation:
    """What every dashboard ships: round each row, hope the column adds up."""
    out = []
    for q in _shares(table):
        # half away from zero, the SQL/Excel convention
        out.append(int(math.floor(q + Fraction(1, 2))) if q >= 0 else -int(math.floor(-q + Fraction(1, 2))))
    return Allocation("naive_half_up", tuple(out))


def naive_half_even(table: Table) -> Allocation:
    """Python's own `round`, and therefore pandas' default: half to even.

    Included because it is the same call in the same language and it produces a
    different digit from `naive_half_up` on any exact half - and an exact half is
    common, not rare, once the denominator is a power of two or a round number.
    """
    out = []
    for q in _shares(table):
        f = math.floor(q)
        rem = q - f
        if rem > Fraction(1, 2):
            out.append(int(f) + 1)
        elif rem < Fraction(1, 2):
            out.append(int(f))
        else:
            out.append(int(f) if int(f) % 2 == 0 else int(f) + 1)
    return Allocation("naive_half_even", tuple(out))


def largest_remainder(table: Table) -> Allocation:
    """Hamilton / Hare-Niemeyer: floor everybody, then hand out what is left.

    Stays inside the quota by construction - every row gets its floor or its
    ceiling, never more. Pays for that with the Alabama paradox, and with a tie
    break that is pure convention: rows with equal remainders are separated here
    by input order, so sorting the input changes the table.
    """
    quotas = _shares(table)
    floors = [int(math.floor(q)) for q in quotas]
    left = table.units - sum(floors)
    order = sorted(range(len(quotas)), key=lambda i: (-(quotas[i] - floors[i]), i))
    notes = []
    rems = [quotas[i] - floors[i] for i in range(len(quotas))]
    if left > 0 and len(set(rems)) < len(rems):
        notes.append("tied remainders were separated by input order, which is a convention, not a rule")
    for i in order[:max(left, 0)]:
        floors[i] += 1
    return Allocation("largest_remainder", tuple(floors), tuple(notes))


def _highest_averages(table: Table, priority: Callable[[float, int], float], name: str,
                      min_one: bool = False) -> Allocation:
    """The divisor-method engine: hand out one unit at a time to the top priority.

    `priority(value, already_awarded)` is the whole difference between D'Hondt,
    Sainte-Lague, Adams and Huntington-Hill. Rows with a zero value never get a
    unit, whatever the formula says about dividing by zero.
    """
    n = len(table.rows)
    awarded = [0] * n
    notes: List[str] = []
    live = [i for i in range(n) if table.rows[i].value > 0]
    if not live:
        return Allocation(name, tuple(awarded), ("no row has a positive value",))
    if min_one and table.units < len(live):
        return Allocation(
            name, tuple(awarded),
            (f"{name} awards every row at least one unit and there are only "
             f"{table.units} units for {len(live)} rows, so it has no answer here",),
        )
    for _ in range(table.units):
        best, best_p = None, None
        for i in live:
            p = priority(table.rows[i].value, awarded[i])
            if best_p is None or p > best_p:
                best, best_p = i, p
        awarded[best] += 1
    if min_one:
        notes.append("every row with a positive value gets at least one unit, so this method "
                     "cannot print a zero for a row that exists")
    return Allocation(name, tuple(awarded), tuple(notes))


def jefferson(table: Table) -> Allocation:
    """Jefferson / D'Hondt: divide by (awarded + 1). Rounds down, favours the big."""
    return _highest_averages(table, lambda v, a: v / (a + 1), "jefferson_dhondt")


def webster(table: Table) -> Allocation:
    """Webster / Sainte-Lague: divide by (2*awarded + 1). Nearest, least biased."""
    return _highest_averages(table, lambda v, a: v / (2 * a + 1), "webster_sainte_lague")


def adams(table: Table) -> Allocation:
    """Adams: divide by awarded. Rounds up, favours the small, needs units >= rows."""
    return _highest_averages(table, lambda v, a: math.inf if a == 0 else v / a, "adams", min_one=True)


def huntington_hill(table: Table) -> Allocation:
    """Huntington-Hill: divide by the geometric mean. The US House method since 1941."""
    return _highest_averages(
        table, lambda v, a: math.inf if a == 0 else v / math.sqrt(a * (a + 1)),
        "huntington_hill", min_one=True,
    )


def last_row_dump(table: Table) -> Allocation:
    """The hack: round naively, then push the whole residual onto the last row.

    Sums correctly and lies about one specific row by the entire residual, with no
    marking. On a long tail the last row is usually the smallest, so the lie is
    largest relative to the number it replaces.
    """
    base = list(naive_half_up(table).units)
    residual = table.units - sum(base)
    if base:
        base[-1] += residual
    note = (f"row {table.labels[-1]!r} carries the whole residual of {residual:+d} units",) if residual else ()
    return Allocation("last_row_dump", tuple(base), note)


def largest_row_dump(table: Table) -> Allocation:
    """The other hack: push the residual onto the biggest row, where it shows least."""
    base = list(naive_half_up(table).units)
    residual = table.units - sum(base)
    if base:
        i = max(range(len(base)), key=lambda k: (table.rows[k].value, -k))
        base[i] += residual
        note = (f"row {table.labels[i]!r} carries the residual of {residual:+d} units "
                "because it is the largest, which is where it is least visible",) if residual else ()
    else:
        note = ()
    return Allocation("largest_row_dump", tuple(base), note)


METHODS: Dict[str, Callable[[Table], Allocation]] = {
    "naive_half_up": naive_half_up,
    "naive_half_even": naive_half_even,
    "largest_remainder": largest_remainder,
    "jefferson_dhondt": jefferson,
    "webster_sainte_lague": webster,
    "adams": adams,
    "huntington_hill": huntington_hill,
    "last_row_dump": last_row_dump,
    "largest_row_dump": largest_row_dump,
}

METHOD_KIND: Dict[str, str] = {
    "naive_half_up": "independent rounding",
    "naive_half_even": "independent rounding",
    "largest_remainder": "apportionment (quota)",
    "jefferson_dhondt": "apportionment (divisor)",
    "webster_sainte_lague": "apportionment (divisor)",
    "adams": "apportionment (divisor)",
    "huntington_hill": "apportionment (divisor)",
    "last_row_dump": "residual hack",
    "largest_row_dump": "residual hack",
}


# ---------------------------------------------------------------------------
# The paradoxes, computed rather than cited
# ---------------------------------------------------------------------------


def quota_violations(table: Table, alloc: Allocation) -> Tuple[Tuple[str, int, float], ...]:
    """Rows awarded outside [floor(quota), ceil(quota)].

    Staying inside the quota is the one fairness property a reader assumes without
    being told: a row owed 12.4 units gets 12 or 13. Divisor methods break it.
    """
    out = []
    for label, u, q in zip(table.labels, alloc.units, table.quotas()):
        lo, hi = math.floor(q), math.ceil(q)
        if u < lo or u > hi:
            out.append((label, u, float(q)))
    return tuple(out)


def alabama(table: Table, method: str) -> Tuple[Tuple[str, int, int], ...]:
    """Rows that LOSE a unit when the total budget goes UP by one.

    Named for the 1880 census, where Alabama's House seats fell from 8 to 7 as the
    House grew from 299 seats to 300. Nothing about Alabama changed.
    """
    before = METHODS[method](table)
    bigger = Table(table.name, table.rows, table.units + 1, table.kind, table.decimals,
                   table.group_of, table.note)
    after = METHODS[method](bigger)
    return tuple(
        (label, b, a)
        for label, b, a in zip(table.labels, before.units, after.units)
        if a < b
    )


def population_paradox(before: Table, after: Table, method: str) -> Tuple[Tuple[str, str, float, float], ...]:
    """A row growing FASTER than another hands it a unit anyway.

    The standard formulation compares growth *rates* pairwise, not shares against
    the total: state A grows faster than state B, and A loses a seat to B. Returns
    (loser, gainer, loser_growth_rate, gainer_growth_rate).

    Every divisor method is immune to this by construction. Largest remainder is
    not, which is the other half of the Balinski-Young trade: the method that
    respects the quota is the one that can punish growth.
    """
    if before.labels != after.labels:
        raise ValueError("population paradox needs the same rows before and after")
    a0, a1 = METHODS[method](before), METHODS[method](after)
    n = len(before.rows)
    rates = []
    for i in range(n):
        v0, v1 = before.rows[i].value, after.rows[i].value
        rates.append(math.inf if v0 == 0 else v1 / v0)
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rates[i] > rates[j] and a1.units[i] < a0.units[i] and a1.units[j] > a0.units[j]:
                out.append((before.labels[i], before.labels[j], rates[i], rates[j]))
    return tuple(out)


def new_state_paradox(table: Table, new_row: Row, extra_units: int, method: str) -> Tuple[Tuple[str, int, int], ...]:
    """Adding a row - with its own fair share of extra units - moves the others.

    Named for Oklahoma joining the Union in 1907: the House grew by the seats
    Oklahoma was owed, and New York still lost one to Maine.
    """
    before = METHODS[method](table)
    grown = Table(table.name, table.rows + (new_row,), table.units + extra_units,
                  table.kind, table.decimals, table.group_of, table.note)
    after = METHODS[method](grown)
    return tuple(
        (label, b, a)
        for label, b, a in zip(table.labels, before.units, after.units)
        if a != b
    )


def size_bias(table: Table, alloc: Allocation) -> float:
    """Mean units-per-unit-of-value for the big half minus the small half.

    Positive means the method over-serves large rows. Reported as a ratio against
    the exact quota so it is comparable across tables: 1.0 is neutral.
    """
    pairs = [(r.value, u, float(q)) for r, u, q in zip(table.rows, alloc.units, table.quotas())
             if r.value > 0 and q > 0]
    if len(pairs) < 2:
        return 1.0
    pairs.sort(key=lambda p: p[0])
    half = len(pairs) // 2
    small = pairs[:half]
    big = pairs[len(pairs) - half:]
    s = sum(u / q for _, u, q in small) / len(small)
    b = sum(u / q for _, u, q in big) / len(big)
    return b - s


def representable_step(table: Table) -> Optional[float]:
    """The finest percentage step this denominator can actually produce.

    If every value is a whole count and they sum to n, a row's share is k/n and the
    only percentages that exist are multiples of 100/n. Asking for a decimal place
    finer than that prints a digit the data cannot carry.
    """
    if any(float(r.value) != int(r.value) for r in table.rows):
        return None
    n = int(round(table.total))
    if n <= 0:
        return None
    return 100.0 / n


def subtotal_clash(table: Table) -> Tuple[Tuple[str, int, int], ...]:
    """Groups whose rounded rows do not sum to the group's own rounded share.

    A two-level table is over-determined: rows must sum to subtotals, subtotals to
    the grand total, and every number must be a rounded version of its own share.
    Three constraints, one set of integers - usually no solution.
    """
    if not table.group_of:
        return ()
    rows_alloc = largest_remainder(table)
    per_group: Dict[str, int] = {}
    for label, u in zip(table.labels, rows_alloc.units):
        g = table.group_of.get(label, label)
        per_group[g] = per_group.get(g, 0) + u
    group_values: Dict[str, float] = {}
    for r in table.rows:
        g = table.group_of.get(r.label, r.label)
        group_values[g] = group_values.get(g, 0.0) + r.value
    group_table = Table(table.name + "-groups",
                        tuple(Row(g, v) for g, v in group_values.items()),
                        table.units, table.kind, table.decimals)
    group_alloc = largest_remainder(group_table)
    out = []
    for g, own in zip(group_table.labels, group_alloc.units):
        if per_group[g] != own:
            out.append((g, per_group[g], own))
    return tuple(out)


# ---------------------------------------------------------------------------
# Findings - twenty named mechanisms
# ---------------------------------------------------------------------------

FINDING_CODES: Tuple[Tuple[str, str, str], ...] = (
    ("NO_SHARE_DEFINED", "blocking", "the base is zero or negative, so no row has a share"),
    ("MIXED_SIGN_BASE", "blocking", "values have mixed signs, so a percentage of the total is not a share"),
    ("UNREPRESENTABLE_PRECISION", "blocking", "the decimal place asked for is finer than the denominator can produce"),
    ("METHOD_HAS_NO_ANSWER", "blocking", "a method that guarantees every row a unit has fewer units than rows"),
    ("COLUMN_DOES_NOT_SUM", "silent", "independently rounded rows do not sum to the total"),
    ("MODE_DIVERGENCE", "silent", "half-up and half-even give different digits for the same row"),
    ("METHOD_DISAGREEMENT", "silent", "two methods that both sum correctly award a row differently"),
    ("ROW_GAP_ABOVE_ONE", "silent", "a row differs by more than one unit across methods"),
    ("RESIDUAL_ROW_UNMARKED", "silent", "one row silently carries the whole residual"),
    ("QUOTA_VIOLATION", "silent", "a row is awarded outside the floor and ceiling of its exact share"),
    ("ALABAMA_PARADOX", "silent", "raising the total by one unit takes a unit away from a row"),
    ("POPULATION_PARADOX", "silent", "a row growing faster than another loses a unit to it"),
    ("NEW_STATE_PARADOX", "silent", "adding a row with its own share of units moves the existing rows"),
    ("TIE_ORDER_DEPENDENCE", "silent", "tied remainders are separated by input order"),
    ("ZERO_ROW_NONZERO_VALUE", "silent", "a row that exists prints as zero"),
    ("SIZE_BIAS", "silent", "the method systematically over-serves large or small rows"),
    ("SUBTOTAL_CLASH", "silent", "rounded rows do not sum to the group's own rounded share"),
    ("DECIMAL_PLACE_INSTABILITY", "silent", "which row absorbs the residual changes with the decimal place"),
    ("MIN_ONE_GUARANTEE", "advisory", "a method cannot print zero for a row that exists"),
    ("BALINSKI_YOUNG", "advisory", "the quota-versus-paradox trade is proved, and this names which side each method took"),
)
SEVERITY_OF: Dict[str, str] = {c: s for c, s, _ in FINDING_CODES}
DESCRIPTION_OF: Dict[str, str] = {c: d for c, _, d in FINDING_CODES}


def _f(code: str, message: str, method: Optional[str] = None) -> Finding:
    return Finding(code, SEVERITY_OF[code], message, method)


def audit(table: Table) -> Audit:
    """Run every method over one table and report what they disagree about."""
    findings: List[Finding] = []
    ok, why = table.is_definable()
    if not ok:
        code = "MIXED_SIGN_BASE" if "mixed signs" in why else "NO_SHARE_DEFINED"
        return Audit(table, Verdict.UNDEFINED, (), (_f(code, why),))

    allocs = tuple(METHODS[name](table) for name in METHODS)
    by = {a.method: a for a in allocs}
    quotas = table.quotas()
    scale = 10 ** table.decimals

    # --- can this precision exist at all -----------------------------------
    step = representable_step(table)
    if step is not None and table.kind == "percent":
        asked = 1.0 / scale
        if step > asked * 1.0000001:
            n = int(round(table.total))
            achievable = [round(k * step, 4) for k in range(min(n, 6) + 1)]
            findings.append(_f(
                "UNREPRESENTABLE_PRECISION",
                f"the denominator is {n}, so the only shares that exist are multiples of "
                f"{step:.4g} points ({achievable}{' ...' if n > 6 else ''}); a column printed to "
                f"{table.decimals} decimal place(s) implies a step of {asked:g} and a sample "
                f"{'far ' if step > 20 * asked else ''}larger than {n}",
            ))

    # --- the naive column ---------------------------------------------------
    naive = by["naive_half_up"]
    if not naive.sums_to(table):
        drift = naive.total - table.units
        findings.append(_f(
            "COLUMN_DOES_NOT_SUM",
            f"every row is correctly rounded and the column sums to "
            f"{naive.total / scale:g}{'%' if table.kind == 'percent' else ''} instead of "
            f"{table.units / scale:g}{'%' if table.kind == 'percent' else ''} "
            f"({drift:+d} units)",
            "naive_half_up",
        ))
    even = by["naive_half_even"]
    diff_rows = [table.labels[i] for i in range(len(table.rows)) if even.units[i] != naive.units[i]]
    if diff_rows:
        findings.append(_f(
            "MODE_DIVERGENCE",
            f"half-up and half-even differ on {len(diff_rows)} row(s): {diff_rows}. Python's `round` and "
            f"pandas use half-even; SQL and spreadsheets use half-up. Same data, same language, different digit",
        ))

    # --- who disagrees, and by how much ------------------------------------
    contested_rows = []
    summing = [a for a in allocs if a.sums_to(table)]
    for i, label in enumerate(table.labels):
        vals = {a.units[i] for a in summing}
        if len(vals) > 1:
            contested_rows.append((label, min(vals), max(vals)))
    if contested_rows:
        worst = max(contested_rows, key=lambda r: r[2] - r[1])
        findings.append(_f(
            "METHOD_DISAGREEMENT",
            f"{len(contested_rows)} of {len(table.rows)} rows are awarded differently by methods that all "
            f"sum correctly; the widest is {worst[0]!r} at {worst[1]}-{worst[2]} units "
            f"({(worst[2] - worst[1]) / scale:g} points apart)",
        ))
        if worst[2] - worst[1] > 1:
            findings.append(_f(
                "ROW_GAP_ABOVE_ONE",
                f"row {worst[0]!r} differs by {worst[2] - worst[1]} units across methods, which is more than a "
                f"rounding residual - this is a policy difference, not a rounding one",
            ))

    for name in ("last_row_dump", "largest_row_dump"):
        for note in by[name].notes:
            if "residual" in note:
                findings.append(_f("RESIDUAL_ROW_UNMARKED", note, name))

    for note in by["largest_remainder"].notes:
        if "tied remainders" in note:
            findings.append(_f("TIE_ORDER_DEPENDENCE", note, "largest_remainder"))

    # --- quota, paradoxes, bias --------------------------------------------
    for a in allocs:
        # A method that has no answer here already reported that; charging it a
        # quota violation on its empty allocation would be counting a non-answer
        # twice.
        if not a.sums_to(table):
            continue
        v = quota_violations(table, a)
        if v:
            label, u, q = v[0]
            findings.append(_f(
                "QUOTA_VIOLATION",
                f"{a.method} awards {label!r} {u} units against an exact share of {q:.4g} "
                f"({len(v)} row(s) outside floor-to-ceiling)",
                a.method,
            ))

    for name in ("largest_remainder", "jefferson_dhondt", "webster_sainte_lague"):
        hits = alabama(table, name)
        if hits:
            label, b, a_ = hits[0]
            findings.append(_f(
                "ALABAMA_PARADOX",
                f"{name}: raising the total from {table.units} to {table.units + 1} units drops {label!r} "
                f"from {b} to {a_}. Nothing about {label!r} changed",
                name,
            ))

    for a in allocs:
        if not a.sums_to(table):
            continue
        for i, label in enumerate(table.labels):
            if a.units[i] == 0 and table.rows[i].value > 0:
                findings.append(_f(
                    "ZERO_ROW_NONZERO_VALUE",
                    f"{a.method} prints {label!r} as 0 with {table.rows[i].value:g} of "
                    f"{table.total:g} ({100 * table.rows[i].value / table.total:.4g}% of the base)",
                    a.method,
                ))
                break

    for a in allocs:
        if not a.sums_to(table) or len(table.rows) < 4:
            continue
        bias = size_bias(table, a)
        if abs(bias) > 0.02:
            side = "large" if bias > 0 else "small"
            findings.append(_f(
                "SIZE_BIAS",
                f"{a.method} serves {side} rows better: mean units-per-quota differs by {bias:+.3f} "
                f"between the big and small halves",
                a.method,
            ))

    for name in ("adams", "huntington_hill"):
        for note in by[name].notes:
            if "no answer" in note:
                findings.append(_f("METHOD_HAS_NO_ANSWER", note, name))
            elif "at least one unit" in note:
                findings.append(_f("MIN_ONE_GUARANTEE", note, name))

    clash = subtotal_clash(table)
    if clash:
        g, rows_sum, own = clash[0]
        findings.append(_f(
            "SUBTOTAL_CLASH",
            f"group {g!r}: its rounded rows sum to {rows_sum / scale:g} and its own rounded share is "
            f"{own / scale:g}. A two-level table is over-determined - rows, subtotals and the grand total "
            f"cannot all be rounded versions of their own shares",
        ))

    # --- does the answer survive a change of decimal place -----------------
    if table.kind == "percent" and table.decimals > 0:
        coarser = Table(table.name, table.rows, table.units // 10, table.kind,
                        table.decimals - 1, table.group_of, table.note)
        fine = largest_remainder(table).units
        coarse = largest_remainder(coarser).units
        fine_top = max(range(len(fine)), key=lambda i: fine[i] - math.floor(quotas[i]))
        coarse_q = coarser.quotas()
        coarse_top = max(range(len(coarse)), key=lambda i: coarse[i] - math.floor(coarse_q[i]))
        if fine_top != coarse_top:
            findings.append(_f(
                "DECIMAL_PLACE_INSTABILITY",
                f"at {table.decimals} dp the extra unit goes to {table.labels[fine_top]!r}; at "
                f"{table.decimals - 1} dp it goes to {table.labels[coarse_top]!r}. The row that looks "
                f"generously treated is a function of the display format",
            ))

    findings.append(_f(
        "BALINSKI_YOUNG",
        "no method avoids both quota violation and the Alabama paradox (Balinski and Young, 1982). "
        "largest_remainder keeps the quota and takes the paradox; the divisor methods take the "
        "violation and are paradox-free. This table cannot be given a method that is neither",
    ))

    # --- verdict ------------------------------------------------------------
    hard = {"QUOTA_VIOLATION", "ALABAMA_PARADOX", "POPULATION_PARADOX", "NEW_STATE_PARADOX",
            "SUBTOTAL_CLASH", "ROW_GAP_ABOVE_ONE", "ZERO_ROW_NONZERO_VALUE",
            "UNREPRESENTABLE_PRECISION"}
    codes = {f.code for f in findings}
    if codes & hard:
        verdict = Verdict.CONTESTED
    elif "METHOD_DISAGREEMENT" in codes or "COLUMN_DOES_NOT_SUM" in codes or "MODE_DIVERGENCE" in codes:
        verdict = Verdict.RESIDUAL
    else:
        verdict = Verdict.CONSISTENT

    seen, uniq = set(), []
    for f in findings:
        key = (f.code, f.message)
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return Audit(table, verdict, allocs, tuple(uniq))


# ---------------------------------------------------------------------------
# The corpus: ten tables, each a real shape of table
# ---------------------------------------------------------------------------
#
# Values in the paradox tables were found by exhaustive search over small integer
# tables, not quoted from a textbook - `evidence.py` re-derives each one.


def percent_table(name: str, pairs: Sequence[Tuple[str, float]], decimals: int = 1,
                  groups: Optional[Dict[str, str]] = None, note: str = "") -> Table:
    return Table(name, tuple(Row(a, b) for a, b in pairs), 100 * 10 ** decimals,
                 "percent", decimals, dict(groups or {}), note)


def seat_table(name: str, pairs: Sequence[Tuple[str, float]], seats: int, note: str = "") -> Table:
    return Table(name, tuple(Row(a, b) for a, b in pairs), seats, "seats", 0, {}, note)


THIRDS = percent_table("thirds", [("alpha", 1000), ("beta", 1000), ("gamma", 1000)],
                       note="three equal rows: the smallest table that fails to sum")

TRAFFIC = percent_table(
    "traffic",
    [("organic", 48213), ("direct", 21877), ("paid-search", 12044), ("referral", 6231),
     ("social", 3187), ("email", 1502), ("affiliate", 214), ("qr-print", 29)],
    note="a real acquisition table with a long tail; the last row is 0.02% of the base",
)

SURVEY7 = percent_table(
    "survey-n7", [("yes", 3), ("no", 3), ("unsure", 1)],
    note="seven respondents, printed to one decimal place",
)

COMMITTEE = seat_table(
    "committee-7", [("engineering", 22), ("operations", 39), ("legal", 4)], 7,
    note="Alabama paradox: legal loses its only seat when the committee grows to 8",
)

CENSUS_BEFORE = seat_table(
    "census-before", [("north", 302), ("east", 25), ("south", 259)], 13,
    note="the before table for the population paradox",
)
CENSUS_AFTER = seat_table(
    "census-after", [("north", 434), ("east", 27), ("south", 325)], 13,
    note="south grew 25.5% and east 8%, and south hands east a seat",
)

NEWCOMER = seat_table(
    "newcomer-19", [("west", 122), ("centre", 57), ("hills", 40), ("coast", 104)], 19,
    note="new-state paradox: adding a 103-value row with its own 6 seats moves centre and hills",
)
NEWCOMER_ROW = Row("newtown", 103)
NEWCOMER_EXTRA = 6

COUNCIL = seat_table(
    "council-9", [("blue", 5709), ("red", 2908), ("green", 2492),
                  ("yellow", 920), ("grey", 911)], 9,
    note="nine seats, five parties: three methods, three different answers, and blue ranges 3 to 5",
)

QUARTERS = percent_table(
    "quarters", [("north", 250), ("east", 250), ("south", 250), ("west", 250)],
    note="the table that works: four equal rows and a denominator that divides",
)

POWERS = percent_table(
    "instances", [("t3.nano", 100), ("t3.micro", 100), ("t3.small", 200),
                  ("t3.medium", 400), ("t3.large", 800)],
    note="every share lands on an exact half, so the rounding mode picks the digit",
)

QUEUES = seat_table(
    "queues-13", [("billing", 57), ("api", 90), ("mobile", 18), ("web", 19), ("platform", 395)], 13,
    note="Sainte-Lague is paradox-free and here it awards platform 10 seats against a share of 8.87",
)

SHIFTS = seat_table(
    "shifts-17", [("sre-eu", 2), ("sre-apac", 8), ("sre-us", 190), ("core", 300)], 17,
    note="Huntington-Hill awards core 9 shifts against a share of 10.2 - below its own floor",
)

SHORTLIST = seat_table(
    "shortlist-3", [("ana", 41), ("ben", 33), ("cai", 22), ("dee", 18), ("eli", 9)], 3,
    note="three places, five candidates: the methods that guarantee everyone a seat have no answer",
)

PNL = percent_table(
    "pnl", [("subscriptions", 820000), ("services", 240000), ("hardware", -95000),
            ("other", 15000)],
    note="a P&L line is negative, so there is no share to display",
)

ZERO_BASE = percent_table("zero-base", [("draft", 0), ("review", 0), ("done", 0)],
                          note="nothing has happened yet")

GROUPED = percent_table(
    "regions-grouped",
    [("london", 2133), ("manchester", 2684), ("leeds", 1045),
     ("berlin", 3444), ("munich", 4122), ("hamburg", 1469)],
    groups={"london": "UK", "manchester": "UK", "leeds": "UK",
            "berlin": "DE", "munich": "DE", "hamburg": "DE"},
    note="rows, subtotals and a grand total, all rounded to one decimal place",
)

CORPUS: Tuple[Table, ...] = (
    QUARTERS, THIRDS, POWERS, TRAFFIC, SURVEY7, GROUPED, COMMITTEE, CENSUS_AFTER,
    NEWCOMER, COUNCIL, QUEUES, SHIFTS, SHORTLIST, PNL, ZERO_BASE,
)


@dataclass(frozen=True)
class CorpusReport:
    audits: Tuple[Audit, ...]
    verdicts: Dict[str, int]
    finding_counts: Dict[str, int]
    method_sum_failures: Dict[str, int]
    method_quota_failures: Dict[str, int]

    @property
    def total(self) -> int:
        return len(self.audits)

    def clean(self) -> Tuple[Audit, ...]:
        return tuple(a for a in self.audits if a.verdict is Verdict.CONSISTENT)


def audit_corpus(tables: Sequence[Table] = CORPUS) -> CorpusReport:
    audits = tuple(audit(t) for t in tables)
    verdicts = {v.value: 0 for v in Verdict}
    finding_counts = {c: 0 for c, _, _ in FINDING_CODES}
    sum_fail = {m: 0 for m in METHODS}
    quota_fail = {m: 0 for m in METHODS}
    for a in audits:
        verdicts[a.verdict.value] += 1
        for f in a.findings:
            finding_counts[f.code] += 1
        for al in a.allocations:
            if not al.sums_to(a.table):
                sum_fail[al.method] += 1
            if quota_violations(a.table, al):
                quota_fail[al.method] += 1
    # The two paired paradoxes are properties of a pair of tables, so they are
    # counted here rather than inside a single-table audit.
    if population_paradox(CENSUS_BEFORE, CENSUS_AFTER, "largest_remainder"):
        finding_counts["POPULATION_PARADOX"] += 1
    if new_state_paradox(NEWCOMER, NEWCOMER_ROW, NEWCOMER_EXTRA, "largest_remainder"):
        finding_counts["NEW_STATE_PARADOX"] += 1
    return CorpusReport(audits, verdicts, finding_counts, sum_fail, quota_fail)


def no_method_is_clean() -> Dict[str, Tuple[int, int, int]]:
    """Per method: (tables where it fails to sum, quota violations, Alabama hits).

    Every column has a nonzero entry somewhere, which is the file's whole claim.
    """
    out: Dict[str, Tuple[int, int, int]] = {}
    for m in METHODS:
        s = q = al = 0
        for t in CORPUS:
            ok, _ = t.is_definable()
            if not ok:
                continue
            a = METHODS[m](t)
            if not a.sums_to(t):
                s += 1
                continue  # no answer, or a column that does not add up: not a quota question
            if quota_violations(t, a):
                q += 1
            if m in ("largest_remainder", "jefferson_dhondt", "webster_sainte_lague",
                     "adams", "huntington_hill") and alabama(t, m):
                al += 1
        out[m] = (s, q, al)
    return out


__all__ = [
    "Row", "Table", "Verdict", "Allocation", "Finding", "Audit", "CorpusReport",
    "METHODS", "METHOD_KIND", "FINDING_CODES", "SEVERITY_OF", "DESCRIPTION_OF",
    "CORPUS", "THIRDS", "TRAFFIC", "SURVEY7", "COMMITTEE", "CENSUS_BEFORE",
    "CENSUS_AFTER", "NEWCOMER", "NEWCOMER_ROW", "NEWCOMER_EXTRA", "COUNCIL", "QUARTERS",
    "POWERS", "SHORTLIST", "QUEUES", "SHIFTS",
    "GROUPED", "PNL", "ZERO_BASE", "audit", "audit_corpus", "no_method_is_clean",
    "percent_table", "seat_table", "quota_violations", "alabama",
    "population_paradox", "new_state_paradox", "size_bias", "representable_step",
    "subtotal_clash", "naive_half_up", "naive_half_even", "largest_remainder",
    "jefferson", "webster", "adams", "huntington_hill", "last_row_dump",
    "largest_row_dump",
]
