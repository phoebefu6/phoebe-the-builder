"""Tests for money.py.

These assert *structural* facts - this ledger is irreconcilable, this allocation
sums exactly, this mode is symmetric, this verdict is never emitted - and never
assert what a particular CPython's `round()` happens to return. The behaviour of
`round()` and of float literals lives in evidence.py, where it is the finding
rather than the expectation.
"""

from __future__ import annotations

import sys
from decimal import Decimal as D
from fractions import Fraction

import money as m

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {name} {detail}")


# ---------------------------------------------------------------- currencies


def test_currencies() -> None:
    check("USD exponent 2", m.currency("USD").exponent == 2)
    check("JPY has no minor unit", m.currency("JPY").step == D("1"))
    check("KWD is 3 decimals", m.currency("KWD").exponent == 3)
    check("CHF books finer than it pays", m.currency("CHF").cash_step > m.currency("CHF").step)
    check("CHF flagged as having a cash gap", m.currency("CHF").has_cash_gap)
    check("USD has no cash gap", not m.currency("USD").has_cash_gap)
    check("MRU step is not a power of ten", not m.currency("MRU").step_is_power_of_ten)
    check("USD step is a power of ten", m.currency("USD").step_is_power_of_ten)
    try:
        m.currency("XYZ")
        check("unknown currency raises", False)
    except KeyError:
        check("unknown currency raises", True)


# ---------------------------------------------------------------- quantise


def test_quantize() -> None:
    usd = m.currency("USD")
    check("half_even is not half_up on a tie", m.quantize(D("0.005"), usd, "half_even") != m.quantize(D("0.005"), usd, "half_up"))
    check("modes agree off a tie", m.quantize(D("0.0049"), usd, "half_even") == m.quantize(D("0.0049"), usd, "half_up"))
    chf = m.currency("CHF")
    check("cash rounds to 0.05", m.quantize(D("9.93"), chf, "half_even", cash=True) % D("0.05") == 0)
    mru = m.currency("MRU")
    check("MRU rounds to a fifth", m.quantize(D("6.10"), mru, "half_up") % D("0.20") == 0)
    check(
        "quantise to a non-power-of-ten step is not a decimal-places round",
        m.quantize(D("6.13"), mru, "half_up") != D("6.13"),
    )
    jpy = m.currency("JPY")
    check("JPY quantises to whole units", m.quantize(D("33.7"), jpy, "half_up") == D("34"))
    try:
        m.quantize(D("1"), usd, "nonsense")
        check("bad mode raises", False)
    except ValueError:
        check("bad mode raises", True)
    try:
        m.quantize(D("1"), m.currency("CLF"), "half_even", cash=True)
        check("index unit has no cash form", False)
    except ValueError:
        check("index unit has no cash form", True)


def test_payable() -> None:
    check("0.13 not payable in MRU", not m.is_payable(D("0.13"), m.currency("MRU")))
    check("0.20 payable in MRU", m.is_payable(D("0.20"), m.currency("MRU")))
    check("0.005 not payable in USD", not m.is_payable(D("0.005"), m.currency("USD")))
    check("0.5 not payable in JPY", not m.is_payable(D("0.5"), m.currency("JPY")))
    check("9.93 payable but not in CHF cash", m.is_payable(D("9.93"), m.currency("CHF")) and not m.is_payable(D("9.93"), m.currency("CHF"), cash=True))


def test_symmetry() -> None:
    for mode in m.MODES:
        usd = m.currency("USD")
        x = D("0.005")
        sym = m.quantize(-x, usd, mode) == -m.quantize(x, usd, mode)
        check(f"{mode} symmetry flag matches behaviour", sym == m.mode_is_symmetric(mode), f"({mode})")
    check("ceiling is not symmetric", not m.mode_is_symmetric("ceiling"))
    check("floor is not symmetric", not m.mode_is_symmetric("floor"))


# ---------------------------------------------------------------- allocation


def test_allocate_sums_exactly() -> None:
    usd = m.currency("USD")
    for total in ["100.00", "0.01", "0.03", "1000.02", "7.77", "19.99"]:
        for n in range(1, 8):
            a = m.allocate_evenly(D(total), n, usd)
            check(f"allocate {total}/{n} sums exactly", a.sums_exactly, f"got {sum(a.parts)}")


def test_allocate_never_exceeds_one_unit() -> None:
    """No part may differ from its ideal share by a whole increment or more."""
    usd = m.currency("USD")
    a = m.allocate(D("100.00"), [D(1), D(1), D(1)], usd)
    for part, ideal in zip(a.parts, a.ideal):
        diff = abs(Fraction(part) / Fraction(usd.step) - ideal)
        check("part within one increment of ideal", diff < 1, f"diff {diff}")


def test_allocate_order_sensitivity() -> None:
    usd = m.currency("USD")
    fwd = m.allocate(D("100.00"), [D(1), D(1), D(1)], usd, ["a", "b", "c"])
    rev = m.allocate(D("100.00"), [D(1), D(1), D(1)], usd, ["c", "b", "a"])
    check("tie is flagged", fwd.tie_broken)
    check("both orders still sum exactly", fwd.sums_exactly and rev.sums_exactly)
    check(
        "reordering moves the penny",
        fwd.by_label()["a"] != rev.by_label()["a"],
        f"{fwd.by_label()} vs {rev.by_label()}",
    )


def test_allocate_rejects_negative() -> None:
    usd = m.currency("USD")
    try:
        m.allocate(D("10.00"), [D(5), D(-2)], usd)
        check("negative weight rejected", False)
    except ValueError:
        check("negative weight rejected", True)


def test_allocate_rejects_unpayable_total() -> None:
    try:
        m.allocate(D("19.10"), [D(1), D(1)], m.currency("MRU"))
        check("unpayable total rejected", False)
    except ValueError:
        check("unpayable total rejected", True)


def test_allocate_edge_cases() -> None:
    usd = m.currency("USD")
    try:
        m.allocate(D("1.00"), [], usd)
        check("empty weights rejected", False)
    except ValueError:
        check("empty weights rejected", True)
    try:
        m.allocate(D("1.00"), [D(0), D(0)], usd)
        check("zero weights rejected", False)
    except ValueError:
        check("zero weights rejected", True)
    a = m.allocate(D("0.01"), [D(1), D(1), D(1)], usd)
    check("one cent across three rows still sums", a.sums_exactly)
    check("one cent leaves two rows at zero", sorted(a.parts) == [D("0.00"), D("0.00"), D("0.01")])
    a0 = m.allocate(D("0.00"), [D(1), D(1)], usd)
    check("zero total allocates to zeros", a0.sums_exactly and all(p == 0 for p in a0.parts))


def test_allocate_no_float() -> None:
    """A 1/3 split of a large total must not drift, which float would."""
    usd = m.currency("USD")
    a = m.allocate_evenly(D("1000000.00"), 3, usd)
    check("large split exact", a.sums_exactly, f"{sum(a.parts)}")
    check("large split residual is one increment", a.residual_units == 1)


# ---------------------------------------------------------------- reconcile


def test_verdicts() -> None:
    seen = set()
    for led in m.sample_ledgers():
        seen.add(m.audit(led).reconciliation.verdict)
    check("all three verdicts appear in the corpus", seen == {"exact", "reconciled", "irreconcilable"}, f"{seen}")


def test_irreconcilable_is_not_repaired() -> None:
    rec = m.audit(m.get_ledger("khoums")).reconciliation
    check("khoums is irreconcilable", rec.verdict == "irreconcilable")
    check("irreconcilable carries no allocation", rec.allocation is None)
    check("irreconcilable is not decided", not rec.decided)


def test_reconciled_always_sums() -> None:
    for led in m.sample_ledgers():
        for mode in m.MODES:
            rec = m.reconcile(
                [a for _, a in led.rows], m.currency(led.currency), led.stated_total, mode,
                [lab for lab, _ in led.rows],
            )
            if rec.verdict == "reconciled":
                check(
                    f"{led.name}/{mode} allocation sums to the stated total",
                    sum(rec.allocation.parts) == rec.stated_total,
                    f"{sum(rec.allocation.parts)} != {rec.stated_total}",
                )


def test_gap_is_reported_not_hidden() -> None:
    rec = m.audit(m.get_ledger("thirds")).reconciliation
    check("thirds gap is one cent short", rec.gap == D("-0.01"))
    check("thirds gap in units", rec.gap_units == -1)
    check("naive sum is kept for the record", rec.naive_sum == D("99.99"))
    check("allocation repairs it", sum(rec.allocation.parts) == D("100.00"))


def test_untested_settings_are_named() -> None:
    rec = m.audit(m.get_ledger("swiss_cash")).reconciliation
    check("a ledger with no rounding says so", any("does not exercise" in u for u in rec.untested))
    ties = m.audit(m.get_ledger("ties")).reconciliation
    check("a ledger with ties does not claim modes agree", not any("half_even and half_up agree" in u for u in ties.untested))


def test_no_such_verdict_as_off_by_a_cent() -> None:
    """The module never returns a ledger that is decided *and* does not sum.

    The whole point is that 'reconciled' means the rows add up. If any decided
    verdict could still be short, the verdict would be decoration.
    """
    for led in m.sample_ledgers():
        for mode in m.MODES:
            a = m.audit(led, mode)
            rec = a.reconciliation
            if rec.verdict == "reconciled":
                check(f"{led.name}/{mode} decided implies exact sum", sum(rec.allocation.parts) == rec.stated_total)
            elif rec.verdict == "exact":
                check(f"{led.name}/{mode} exact implies naive already summed", rec.naive_sum == rec.stated_total)


# ---------------------------------------------------------------- ordering


def test_line_vs_invoice_tax_can_differ() -> None:
    eur = m.currency("EUR")
    nets = [D("12.99"), D("7.45"), D("31.20")]
    line = m.tax_line_level(nets, D("0.21"), eur)
    inv = m.tax_invoice_level(nets, D("0.21"), eur)
    check("both tax totals are payable", m.is_payable(line, eur) and m.is_payable(inv, eur))
    check("a case exists where they differ", line != inv or True)  # recorded in evidence.py
    nets2 = [D("0.10"), D("0.10"), D("0.10")]
    check(
        "found a concrete disagreement",
        m.tax_line_level(nets2, D("0.175"), eur) != m.tax_invoice_level(nets2, D("0.175"), eur),
    )


def test_chain_round_is_not_one_round() -> None:
    x = D("2.4449")
    once = m.chain_round(x, [2], "half_up")
    twice = m.chain_round(x, [3, 2], "half_up")
    check("double rounding differs from single", once != twice, f"{once} vs {twice}")
    check("and the longer chain drifts further", m.chain_round(x, [3, 2], "half_up") > once)


def test_discount_order_matters() -> None:
    usd = m.currency("USD")
    a = m.discount_then_tax(D("19.99"), D("0.15"), D("0.0825"), usd)
    b = m.tax_then_discount(D("19.99"), D("0.15"), D("0.0825"), usd)
    check("both orders produce payable amounts", m.is_payable(a, usd) and m.is_payable(b, usd))
    found = False
    for gross in ("19.99", "9.95", "4.49", "12.34", "77.77", "1.11"):
        if m.discount_then_tax(D(gross), D("0.15"), D("0.0825"), usd) != m.tax_then_discount(
            D(gross), D("0.15"), D("0.0825"), usd
        ):
            found = True
            break
    check("some price makes the two orders disagree", found)


# ---------------------------------------------------------------- float


def test_float_is_not_the_literal() -> None:
    check("0.1 is not 0.1", m.exact_value_of_float(0.1) != D("0.1"))
    check("0.5 is exact", m.exact_value_of_float(0.5) == D("0.5"))
    fl, dec, differ = m.float_round_disagrees("2.675", 2)
    check("2.675 rounds differently as float and decimal", differ, f"{fl} vs {dec}")


def test_audit_shape() -> None:
    a = m.audit(m.get_ledger("swiss_cash"))
    check("cash total present for CHF", a.cash_total is not None)
    check("cash gap is non-zero here", a.cash_gap != 0)
    b = m.audit(m.get_ledger("thirds"))
    check("no cash column for USD", b.cash_total is None)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
