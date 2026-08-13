"""Currency rounding and allocation that reports what it could not preserve.

A rounding function returns a number. It cannot return the fact that the rows no
longer add up, that the line each penny landed on depends on how the ledger was
sorted, or that the amount it produced is not payable in the currency it is
denominated in.

Core ideas
----------
1. Quantising a single amount is easy and almost never the bug.
2. Quantising a *set* of amounts that must reconcile to a stated total is the bug.
   Independent rounding does not preserve a sum. Allocation does, at the cost of
   moving the error onto named rows.
3. The verdict is three-valued: `exact` (nothing was rounded away),
   `reconciled` (a residual existed and was allocated to identified rows), and
   `irreconcilable` (no allocation in this currency's minor unit can hit the
   stated total).

Standard library only: `decimal`, `fractions`, `dataclasses`, `enum`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import (
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
    ROUND_HALF_DOWN,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    localcontext,
)
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Currencies
# --------------------------------------------------------------------------

# ISO 4217 assigns each currency an *exponent*: the power of ten relating the
# major unit to the minor unit. Two is the common case and not the only one.
#
# `step` is the smallest amount that can appear in the books. `cash_step` is the
# smallest amount that can change hands in physical money, which is a different
# number in several countries and is the reason a receipt total and a cash total
# can legally disagree.


@dataclass(frozen=True)
class Currency:
    code: str
    exponent: int  # ISO 4217 exponent
    step: Decimal  # smallest bookable increment
    cash_step: Optional[Decimal]  # smallest increment payable in cash, if different
    note: str = ""

    @property
    def has_cash_gap(self) -> bool:
        return self.cash_step is not None and self.cash_step != self.step

    @property
    def step_is_power_of_ten(self) -> bool:
        """True when `step` is exactly 10**-exponent.

        False for currencies whose real subdivision is not a tenth-power, where
        ISO's exponent describes the *number of digits printed* rather than the
        set of amounts that exist.
        """
        return self.step == Decimal(1).scaleb(-self.exponent)


CURRENCIES: Dict[str, Currency] = {
    "USD": Currency("USD", 2, Decimal("0.01"), Decimal("0.01"), "the assumed default"),
    "EUR": Currency("EUR", 2, Decimal("0.01"), Decimal("0.01"), "cent is the coin"),
    "GBP": Currency("GBP", 2, Decimal("0.01"), Decimal("0.01"), "penny is the coin"),
    "JPY": Currency(
        "JPY", 0, Decimal("1"), Decimal("1"), "no minor unit; a 'cent' of JPY does not exist"
    ),
    "KWD": Currency(
        "KWD", 3, Decimal("0.001"), Decimal("0.005"), "1000 fils; smallest coin is 5 fils"
    ),
    "BHD": Currency("BHD", 3, Decimal("0.001"), Decimal("0.005"), "1000 fils"),
    "CHF": Currency(
        "CHF", 2, Decimal("0.01"), Decimal("0.05"), "books in rappen, pays in 5-rappen coins"
    ),
    "SEK": Currency(
        "SEK", 2, Decimal("0.01"), Decimal("1"), "ore coins withdrawn; cash rounds to the krona"
    ),
    "CAD": Currency(
        "CAD", 2, Decimal("0.01"), Decimal("0.05"), "penny withdrawn 2013; cash rounds to a nickel"
    ),
    "MRU": Currency(
        "MRU",
        2,
        Decimal("0.20"),
        Decimal("0.20"),
        "5 khoums to the ouguiya: the subdivision is a fifth, not a hundredth",
    ),
    "MGA": Currency(
        "MGA",
        2,
        Decimal("0.20"),
        Decimal("0.20"),
        "5 iraimbilanja to the ariary: also a fifth",
    ),
    "CLF": Currency(
        "CLF", 4, Decimal("0.0001"), None, "unidad de fomento, an index unit: 4 decimals"
    ),
}


def currency(code: str) -> Currency:
    try:
        return CURRENCIES[code]
    except KeyError:
        raise KeyError(
            f"{code!r} is not in the sample table. This module deliberately ships a "
            f"small table rather than guessing an exponent it has not verified."
        )


# --------------------------------------------------------------------------
# Rounding modes
# --------------------------------------------------------------------------

MODES: Dict[str, str] = {
    "half_even": ROUND_HALF_EVEN,  # Python / IEEE 754 default
    "half_up": ROUND_HALF_UP,  # ties away from zero (Excel ROUND, most tax law)
    "half_down": ROUND_HALF_DOWN,  # ties toward zero
    "ceiling": ROUND_CEILING,  # toward +inf
    "floor": ROUND_FLOOR,  # toward -inf
    "down": ROUND_DOWN,  # truncate toward zero
}

# Which modes treat +x and -x as mirror images. A mode that is not symmetric
# means a charge and its own refund do not cancel to zero.
SYMMETRIC_MODES = {"half_even", "half_up", "half_down", "down"}


def mode_is_symmetric(mode: str) -> bool:
    _check_mode(mode)
    return mode in SYMMETRIC_MODES


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; known: {sorted(MODES)}")


# --------------------------------------------------------------------------
# Quantising one amount
# --------------------------------------------------------------------------


def quantize(amount: Decimal, cur: Currency, mode: str = "half_even", cash: bool = False) -> Decimal:
    """Round one amount to a currency's increment.

    `step` may not be a power of ten (MRU, or CHF cash at 0.05), so this rounds
    to a *multiple of step* rather than to a number of decimal places. Those are
    the same operation only when step is 10**-exponent.
    """
    _check_mode(mode)
    step = cur.cash_step if cash else cur.step
    if step is None:
        raise ValueError(f"{cur.code} has no cash increment: it is not a physical currency")
    with localcontext() as ctx:
        ctx.prec = 60
        units = (amount / step).quantize(Decimal(1), rounding=MODES[mode])
        return (units * step).quantize(step)


def is_payable(amount: Decimal, cur: Currency, cash: bool = False) -> bool:
    """True when `amount` is an exact multiple of the relevant increment.

    An amount that is not payable is not a rounding preference. It is a number
    that cannot be transferred.
    """
    step = cur.cash_step if cash else cur.step
    if step is None:
        return False
    with localcontext() as ctx:
        ctx.prec = 60
        try:
            return (amount / step) % 1 == 0
        except InvalidOperation:
            return False


def residual(amount: Decimal, cur: Currency, mode: str = "half_even") -> Decimal:
    """What quantising threw away: rounded - original."""
    return quantize(amount, cur, mode) - amount


# --------------------------------------------------------------------------
# Allocation: split a total so the parts sum to it exactly
# --------------------------------------------------------------------------


@dataclass
class Allocation:
    """A split of `total` into parts that sum to `total` exactly."""

    total: Decimal
    parts: List[Decimal]
    labels: List[str]
    ideal: List[Fraction]  # the unrounded shares
    absorbed: List[int]  # indices that received a residual unit
    residual_units: int  # how many increments had to be moved
    step: Decimal
    tie_broken: bool  # True if a tie in remainders was settled by position
    order_sensitive: bool  # True if a different row order changes the parts

    @property
    def sums_exactly(self) -> bool:
        return sum(self.parts) == self.total

    def by_label(self) -> Dict[str, Decimal]:
        return dict(zip(self.labels, self.parts))


def allocate(
    total: Decimal,
    weights: Sequence[Decimal],
    cur: Currency,
    labels: Optional[Sequence[str]] = None,
) -> Allocation:
    """Largest-remainder allocation of `total` across `weights`.

    Guarantees `sum(parts) == total` when `total` is payable in `cur`. The
    guarantee is the point; the cost is that the residual increments land on
    specific rows, and *which* rows depends on the order the rows arrived in
    whenever two remainders tie.

    Negative weights are rejected rather than silently mishandled. Largest
    remainder assumes shares are non-negative; a mixed-sign weight vector can
    make the floor step overshoot the total and the method has no defined
    answer there.
    """
    n = len(weights)
    if n == 0:
        raise ValueError("cannot allocate across zero rows")
    labels = list(labels) if labels is not None else [f"row{i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels and weights differ in length")
    if any(w < 0 for w in weights):
        raise ValueError(
            "negative weight: largest-remainder allocation is defined for "
            "non-negative shares only. Split the credit lines out and allocate "
            "them separately, with their own sign."
        )
    wsum = sum(weights)
    if wsum == 0:
        raise ValueError("weights sum to zero: there is no share to compute")
    if not is_payable(total, cur):
        raise ValueError(
            f"{total} is not payable in {cur.code} (increment {cur.step}); "
            f"allocation cannot produce a sum that does not exist"
        )

    step = cur.step
    # Work in integer increments, exactly, via Fraction. No float, no early rounding.
    total_units = Fraction(total) / Fraction(step)
    assert total_units.denominator == 1
    total_units = int(total_units)

    ideal = [Fraction(total_units) * Fraction(w) / Fraction(wsum) for w in weights]
    floors = [int(x // 1) for x in ideal]
    remainders = [x - f for x, f in zip(ideal, floors)]
    short = total_units - sum(floors)

    # Rank by remainder descending; ties fall to the earlier index. That is a
    # choice, and it is the reason this allocation is order sensitive.
    order = sorted(range(n), key=lambda i: (-remainders[i], i))
    absorbed = sorted(order[:short]) if short > 0 else []

    units = list(floors)
    for i in absorbed:
        units[i] += 1

    # Did a tie actually decide anything? Only if the last row taken and the
    # first row not taken have equal remainders.
    tie_broken = False
    if 0 < short < n:
        tie_broken = remainders[order[short - 1]] == remainders[order[short]]
    # More generally: the parts depend on order whenever any residual was moved
    # and any two remainders are equal across the accept/reject boundary.
    order_sensitive = tie_broken

    parts = [Decimal(u) * step for u in units]
    return Allocation(
        total=total,
        parts=parts,
        labels=labels,
        ideal=ideal,
        absorbed=absorbed,
        residual_units=short,
        step=step,
        tie_broken=tie_broken,
        order_sensitive=order_sensitive,
    )


def allocate_evenly(total: Decimal, n: int, cur: Currency, labels=None) -> Allocation:
    return allocate(total, [Decimal(1)] * n, cur, labels)


# --------------------------------------------------------------------------
# Reconciling a ledger against a stated total
# --------------------------------------------------------------------------


@dataclass
class Reconciliation:
    verdict: str  # 'exact' | 'reconciled' | 'irreconcilable'
    stated_total: Decimal
    naive_parts: List[Decimal]  # each row rounded on its own
    naive_sum: Decimal
    gap: Decimal  # naive_sum - stated_total
    gap_units: Optional[int]
    allocation: Optional[Allocation]
    labels: List[str]
    reason: str
    untested: List[str] = field(default_factory=list)

    @property
    def decided(self) -> bool:
        return self.verdict in ("exact", "reconciled")


def reconcile(
    exact_rows: Sequence[Decimal],
    cur: Currency,
    stated_total: Optional[Decimal] = None,
    mode: str = "half_even",
    labels: Optional[Sequence[str]] = None,
) -> Reconciliation:
    """Round a set of rows that must add up, and say what that cost.

    `exact_rows` are the unrounded amounts (e.g. quantity * unit price, or a
    percentage of a base). `stated_total` is the figure the ledger must hit; if
    omitted, the rounded exact sum is used.

    Three verdicts:
      exact          - every row was already payable and they already summed right
      reconciled     - a residual existed; it was allocated, and the rows that
                       absorbed it are named
      irreconcilable - the stated total is not payable in this currency, so no
                       set of payable rows can sum to it
    """
    _check_mode(mode)
    n = len(exact_rows)
    labels = list(labels) if labels is not None else [f"line{i + 1}" for i in range(n)]

    naive = [quantize(x, cur, mode) for x in exact_rows]
    naive_sum = sum(naive, Decimal(0))
    exact_sum = sum(exact_rows, Decimal(0))
    target = stated_total if stated_total is not None else quantize(exact_sum, cur, mode)

    untested: List[str] = []
    if all(is_payable(x, cur) for x in exact_rows):
        untested.append("no row needed rounding: this ledger does not exercise the mode at all")
    if not any(_is_tie(x, cur) for x in exact_rows):
        untested.append("no row landed exactly on a tie: half_even and half_up agree here")
    if not cur.has_cash_gap:
        untested.append("this currency has no cash/book gap: the cash column is untested")

    if not is_payable(target, cur):
        return Reconciliation(
            verdict="irreconcilable",
            stated_total=target,
            naive_parts=naive,
            naive_sum=naive_sum,
            gap=naive_sum - target,
            gap_units=None,
            allocation=None,
            labels=labels,
            reason=(
                f"stated total {target} is not a multiple of {cur.step} in {cur.code}; "
                f"no set of payable rows sums to it"
            ),
            untested=untested,
        )

    gap = naive_sum - target
    gap_units = int(Fraction(gap) / Fraction(cur.step))

    if gap == 0 and all(is_payable(x, cur) for x in exact_rows):
        return Reconciliation(
            verdict="exact",
            stated_total=target,
            naive_parts=naive,
            naive_sum=naive_sum,
            gap=gap,
            gap_units=0,
            allocation=None,
            labels=labels,
            reason="every row was already payable and the rows already summed to the total",
            untested=untested,
        )

    weights = [abs(x) for x in exact_rows]
    if sum(weights) == 0:
        weights = [Decimal(1)] * n
    alloc = allocate(target, weights, cur, labels)
    if gap == 0:
        reason = (
            "rows summed to the total after independent rounding, but at least one "
            "row was rounded, so the split is one of several that also sum right"
        )
    else:
        reason = (
            f"independent rounding missed the total by {gap} "
            f"({gap_units:+d} increment(s)); reallocated so the rows sum exactly"
        )
    return Reconciliation(
        verdict="reconciled",
        stated_total=target,
        naive_parts=naive,
        naive_sum=naive_sum,
        gap=gap,
        gap_units=gap_units,
        allocation=alloc,
        labels=labels,
        reason=reason,
        untested=untested,
    )


def _is_tie(amount: Decimal, cur: Currency) -> bool:
    """True when `amount` sits exactly halfway between two payable amounts."""
    with localcontext() as ctx:
        ctx.prec = 60
        try:
            return (amount / cur.step) % 1 == Decimal("0.5")
        except InvalidOperation:
            return False


# --------------------------------------------------------------------------
# Order of operations
# --------------------------------------------------------------------------


def tax_line_level(
    nets: Sequence[Decimal], rate: Decimal, cur: Currency, mode: str = "half_even"
) -> Decimal:
    """Round the tax on every line, then add. What an invoice line-item shows."""
    return sum((quantize(n * rate, cur, mode) for n in nets), Decimal(0))


def tax_invoice_level(
    nets: Sequence[Decimal], rate: Decimal, cur: Currency, mode: str = "half_even"
) -> Decimal:
    """Add the lines, apply the rate once, round once. What a tax return shows."""
    return quantize(sum(nets, Decimal(0)) * rate, cur, mode)


def chain_round(amount: Decimal, places: Sequence[int], mode: str = "half_even") -> Decimal:
    """Round through a sequence of precisions, e.g. 4dp -> 3dp -> 2dp.

    Rounding twice is not rounding once. A value below a tie at the final
    precision can be pushed onto or over that tie by an earlier step.
    """
    _check_mode(mode)
    out = amount
    for p in places:
        out = out.quantize(Decimal(1).scaleb(-p), rounding=MODES[mode])
    return out


def discount_then_tax(
    gross: Decimal, discount: Decimal, rate: Decimal, cur: Currency, mode: str = "half_even"
) -> Decimal:
    net = quantize(gross * (Decimal(1) - discount), cur, mode)
    return net + quantize(net * rate, cur, mode)


def tax_then_discount(
    gross: Decimal, discount: Decimal, rate: Decimal, cur: Currency, mode: str = "half_even"
) -> Decimal:
    taxed = quantize(gross * (Decimal(1) + rate), cur, mode)
    return quantize(taxed * (Decimal(1) - discount), cur, mode)


# --------------------------------------------------------------------------
# Float vs Decimal
# --------------------------------------------------------------------------


def exact_value_of_float(x: float) -> Decimal:
    """The real number a float holds, in full. Not the literal that was typed."""
    return Decimal(x)


def float_round_disagrees(literal: str, places: int = 2) -> Tuple[Decimal, Decimal, bool]:
    """Compare rounding the float against rounding the decimal of the same text."""
    as_float = round(float(literal), places)
    as_dec = Decimal(literal).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    fl = Decimal(str(as_float))
    return fl, as_dec, fl != as_dec


# --------------------------------------------------------------------------
# Sample ledgers
# --------------------------------------------------------------------------


@dataclass
class Ledger:
    name: str
    currency: str
    rows: List[Tuple[str, Decimal]]  # (label, exact amount)
    stated_total: Optional[Decimal]
    story: str


def sample_ledgers() -> List[Ledger]:
    D = Decimal
    return [
        Ledger(
            "thirds",
            "USD",
            [("alice", D(100) / 3), ("bob", D(100) / 3), ("carol", D(100) / 3)],
            D("100.00"),
            "a $100 refund split three ways; 33.33 x 3 leaves a cent on the floor",
        ),
        Ledger(
            "vat_lines",
            "EUR",
            [
                ("mouse", D("12.99") * D("0.21")),
                ("cable", D("7.45") * D("0.21")),
                ("hub", D("31.20") * D("0.21")),
            ],
            None,
            "21% VAT computed per line; the invoice total disagrees with the return",
        ),
        Ledger(
            "ties",
            "USD",
            [("a", D("1.005")), ("b", D("1.015")), ("c", D("1.025")), ("d", D("1.035"))],
            None,
            "four exact ties: half_even splits them 2-2, half_up sends all four up",
        ),
        Ledger(
            "yen_split",
            "JPY",
            [("east", D(100) / 3), ("west", D(100) / 3), ("north", D(100) / 3)],
            D("100"),
            "the same three-way split in a currency with no minor unit",
        ),
        Ledger(
            "fils",
            "KWD",
            [("consulting", D("125.4567")), ("travel", D("18.2341"))],
            None,
            "3-decimal currency met by a system that rounds to 2",
        ),
        Ledger(
            "khoums",
            "MRU",
            [("grain", D("13.00")), ("oil", D("6.10"))],
            D("19.10"),
            "amounts in hundredths of an ouguiya, a unit that does not exist",
        ),
        Ledger(
            "swiss_cash",
            "CHF",
            [("coffee", D("4.20")), ("pastry", D("3.55")), ("water", D("2.18"))],
            D("9.93"),
            "a payable invoice that cannot be paid in coins",
        ),
        Ledger(
            "weighted",
            "USD",
            [("north", D("250.005")), ("south", D("250.005")), ("east", D("500.01"))],
            D("1000.02"),
            "cost allocation 25/25/50; the two equal rows tie and position decides",
        ),
    ]


def get_ledger(name: str) -> Ledger:
    for led in sample_ledgers():
        if led.name == name:
            return led
    raise KeyError(f"no sample ledger {name!r}")


# --------------------------------------------------------------------------
# Top-level audit
# --------------------------------------------------------------------------


@dataclass
class Audit:
    ledger: str
    currency: Currency
    reconciliation: Reconciliation
    mode: str
    cash_total: Optional[Decimal]
    cash_gap: Optional[Decimal]
    order_sensitive: bool

    @property
    def decided(self) -> bool:
        return self.reconciliation.decided


def audit(led: Ledger, mode: str = "half_even") -> Audit:
    cur = currency(led.currency)
    labels = [lab for lab, _ in led.rows]
    amounts = [amt for _, amt in led.rows]
    rec = reconcile(amounts, cur, led.stated_total, mode, labels)

    cash_total = cash_gap = None
    if cur.has_cash_gap and rec.decided:
        cash_total = quantize(rec.stated_total, cur, "half_even", cash=True)
        cash_gap = cash_total - rec.stated_total

    order_sensitive = bool(rec.allocation and rec.allocation.order_sensitive)
    return Audit(
        ledger=led.name,
        currency=cur,
        reconciliation=rec,
        mode=mode,
        cash_total=cash_total,
        cash_gap=cash_gap,
        order_sensitive=order_sensitive,
    )
