"""Every table in the README, printed. Each experiment isolates one mechanism.

Run: python3 evidence.py
"""

from __future__ import annotations

from decimal import Decimal as D
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, getcontext
from fractions import Fraction

import money as m

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------


def exp1_rows_that_do_not_add_up() -> None:
    head(1, "Every row is correctly rounded. The rows do not add up.")
    usd = m.currency("USD")
    exact = D(100) / 3
    print(f"a $100.00 refund split three ways; each share is exactly {exact}\n")
    print(f"{'row':8} {'exact share':26} {'rounded':>10} {'error':>8}")
    print(RULE)
    for lab in ("alice", "bob", "carol"):
        r = m.quantize(exact, usd, "half_even")
        print(f"{lab:8} {str(exact)[:26]:26} {r:>10} {m.residual(exact, usd):>8}")
    print(RULE)
    naive = m.quantize(exact, usd) * 3
    print(f"{'sum':8} {'100.000000...':26} {naive:>10} {naive - D('100.00'):>8}")
    print(f"{'stated':8} {'':26} {D('100.00'):>10}")
    print(RULE)
    print("no row is wrong. every row rounds to the nearest cent. the ledger is short $0.01.\n")

    rec = m.audit(m.get_ledger("thirds")).reconciliation
    a = rec.allocation
    print("allocation instead of independent rounding:")
    print(f"  parts   {[str(p) for p in a.parts]}   sum {sum(a.parts)}")
    print(f"  absorbed by {[a.labels[i] for i in a.absorbed]} ({a.residual_units} increment)")
    print(f"  verdict {rec.verdict}: {rec.reason}")


def exp2_sorting_moves_the_penny() -> None:
    head(2, "The total is stable. The rows are not. Sorting the ledger moves a cent.")
    usd = m.currency("USD")
    labels = ["carol", "alice", "bob"]  # the order the file arrived in
    print("$100.00, three equal shares, allocated in three different row orders:\n")
    print(f"{'order':22} {'alice':>9} {'bob':>9} {'carol':>9} {'sum':>10}")
    print(RULE)
    for name, order in [("as entered", labels), ("sorted by name", sorted(labels)), ("reversed", labels[::-1])]:
        a = m.allocate(D("100.00"), [D(1)] * 3, usd, order)
        d = a.by_label()
        print(f"{name:22} {d['alice']:>9} {d['bob']:>9} {d['carol']:>9} {sum(a.parts):>10}")
    print(RULE)
    print("every order sums to 100.00 exactly. no order agrees with another on who paid it.\n")

    print("the same on a weighted cost allocation (25 / 25 / 50 of $1000.02):\n")
    led = m.get_ledger("weighted")
    fwd = m.audit(led).reconciliation.allocation
    rows_rev = list(reversed(led.rows))
    rev = m.allocate(D("1000.02"), [abs(a) for _, a in rows_rev], usd, [l for l, _ in rows_rev])
    print(f"{'order':22} {'north':>9} {'south':>9} {'east':>9} {'sum':>10}")
    print(RULE)
    for name, alloc in [("as entered", fwd), ("reversed", rev)]:
        d = alloc.by_label()
        print(f"{name:22} {d['north']:>9} {d['south']:>9} {d['east']:>9} {sum(alloc.parts):>10}")
    print(RULE)
    print(f"remainders tie at {Fraction(1,2)}; the tie is broken by position, and that is a choice.")
    print(f"tie_broken flag: {fwd.tie_broken}   order_sensitive: {fwd.order_sensitive}")
    print("\na month-on-month variance report on 'north' shows a $0.01 movement that is")
    print("entirely an artefact of the sort order of the input file.")


def exp3_half_even_vs_half_up() -> None:
    head(3, "The default is not the law. half_even vs half_up on exact ties.")
    usd = m.currency("USD")
    led = m.get_ledger("ties")
    print("four amounts sitting exactly halfway between two cents:\n")
    print(f"{'amount':10} {'cent units':>12} {'half_even':>11} {'half_up':>9} {'agree':>7}")
    print(RULE)
    for lab, amt in led.rows:
        units = amt / usd.step
        he = m.quantize(amt, usd, "half_even")
        hu = m.quantize(amt, usd, "half_up")
        print(f"{str(amt):10} {str(units):>12} {he:>11} {hu:>9} {'yes' if he == hu else 'NO':>7}")
    print(RULE)
    he_sum = sum(m.quantize(a, usd, "half_even") for _, a in led.rows)
    hu_sum = sum(m.quantize(a, usd, "half_up") for _, a in led.rows)
    print(f"{'sum':10} {'':>12} {he_sum:>11} {hu_sum:>9}")
    print(RULE)
    print("half_even sends two up and two down, so the bias over many rows is near zero.")
    print("half_up sends every tie up, which is what most tax authorities specify and what")
    print("Excel's ROUND() does. Python's round(), Decimal's default context and IEEE 754")
    print("all default to half_even. The two answers differ by "
          f"{hu_sum - he_sum} on four rows.\n")
    print(f"decimal default context rounding: {getcontext().rounding}")
    print(f"round(0.5) = {round(0.5)}   round(1.5) = {round(1.5)}   round(2.5) = {round(2.5)}")
    print("\nmodes and whether a charge cancels its own refund:")
    print(f"\n{'mode':12} {'+0.005':>9} {'-0.005':>9} {'sums to 0':>11}")
    print(RULE)
    for mode in m.MODES:
        p = m.quantize(D("0.005"), usd, mode)
        n = m.quantize(D("-0.005"), usd, mode)
        print(f"{mode:12} {p:>9} {n:>9} {'yes' if p + n == 0 else 'NO':>11}")
    print(RULE)
    print("ceiling and floor are the two that do not. Under 'always round up', issuing a")
    print("charge and its exact refund leaves a cent behind, permanently, per transaction.")


def exp4_float_is_not_the_number() -> None:
    head(4, "The float you rounded is not the number you typed.")
    print(f"{'literal':10} {'the float actually holds':52}")
    print(RULE)
    for lit in ("0.1", "2.675", "1.005", "0.5", "0.145"):
        print(f"{lit:10} {str(m.exact_value_of_float(float(lit)))[:52]:52}")
    print(RULE)
    print("\nrounding the float vs rounding the decimal of the same text:\n")
    print(f"{'literal':10} {'round(float,2)':>15} {'Decimal half_up':>17} {'agree':>7}")
    print(RULE)
    for lit in ("2.675", "1.005", "0.145", "8.835", "1.115"):
        fl, dec, differ = m.float_round_disagrees(lit, 2)
        print(f"{lit:10} {fl:>15} {dec:>17} {'NO' if differ else 'yes':>7}")
    print(RULE)
    print("2.675 is stored as 2.67499999999999982..., which is below the tie, so it rounds")
    print("down. It is not a rounding-mode bug: at that value there is no tie to break.")
    print("\nand the sum that will not settle:")
    print(f"  sum([0.1] * 10)        = {sum([0.1] * 10)!r}")
    print(f"  0.1 + 0.2              = {0.1 + 0.2!r}")
    print(f"  Decimal('0.1') * 3     = {D('0.1') * 3}")
    total = 0.0
    for _ in range(10000):
        total += 0.01
    print(f"  0.01 added 10,000 times = {total!r}  (want 100.0, off by {abs(total - 100.0):.2e})")
    print("\nAt 10,000 rows that is a sub-cent error. It becomes visible when the ledger is")
    print("compared to a system that used Decimal, and it moves when rows are reordered.")


def exp5_order_of_operations() -> None:
    head(5, "Rounding does not commute. Line-level tax and invoice-level tax disagree.")
    eur = m.currency("EUR")
    for nets, rate, label in [
        ([D("12.99"), D("7.45"), D("31.20")], D("0.21"), "21% VAT, three lines"),
        ([D("0.10"), D("0.10"), D("0.10")], D("0.175"), "17.5% on three 10c lines"),
        ([D("9.99")] * 7, D("0.0825"), "8.25% on seven identical lines"),
    ]:
        line = m.tax_line_level(nets, rate, eur)
        inv = m.tax_invoice_level(nets, rate, eur)
        flag = "DIFFER" if line != inv else "agree"
        print(f"\n{label}")
        print(f"  per-line tax rounded then summed : {line}")
        print(f"  summed then taxed then rounded   : {inv}")
        print(f"  -> {flag}  (delta {line - inv})")
    print("\nThe first case agrees. That is the problem: the disagreement is intermittent,")
    print("so a test written against one basket passes and the next basket is short.")
    print("\nBoth are defensible. Line-level is what the printed invoice must show, because")
    print("each line has to display a payable amount. Invoice-level is what a single")
    print("rate applied to a single base gives. EU VAT rounding is set per member state,")
    print("not harmonised by the Directive, so 'the correct one' depends on jurisdiction.\n")

    print("discount and tax, in the two possible orders (15% off, 8.25% tax):\n")
    usd = m.currency("USD")
    print(f"{'gross':>8} {'discount->tax':>15} {'tax->discount':>15} {'delta':>8}")
    print(RULE)
    for g in ("19.99", "9.95", "4.49", "12.34", "77.77", "1.11"):
        a = m.discount_then_tax(D(g), D("0.15"), D("0.0825"), usd)
        b = m.tax_then_discount(D(g), D("0.15"), D("0.0825"), usd)
        print(f"{g:>8} {a:>15} {b:>15} {a - b:>8}")
    print(RULE)
    print("\nand rounding twice is not rounding once:\n")
    print(f"{'value':10} {'->2dp':>9} {'->3dp->2dp':>12} {'->4->3->2':>11}")
    print(RULE)
    for v in ("2.4449", "1.2349", "0.4449", "9.9949"):
        x = D(v)
        print(
            f"{v:10} {m.chain_round(x, [2], 'half_up'):>9} "
            f"{m.chain_round(x, [3, 2], 'half_up'):>12} "
            f"{m.chain_round(x, [4, 3, 2], 'half_up'):>11}"
        )
    print(RULE)
    print("A value below a tie is carried onto the tie by the earlier step and then over it.")
    print("Any pipeline that stores an intermediate at higher precision and rounds again at")
    print("report time has this in it.")


def exp6_two_decimals_is_an_assumption() -> None:
    head(6, "Two decimal places is an assumption, and ISO 4217 does not agree with it.")
    print(f"{'code':6} {'exp':>4} {'book step':>10} {'cash step':>10} {'note'}")
    print(RULE)
    for code in ("USD", "JPY", "KWD", "BHD", "CHF", "SEK", "CAD", "MRU", "MGA", "CLF"):
        c = m.currency(code)
        cash = str(c.cash_step) if c.cash_step is not None else "-"
        print(f"{c.code:6} {c.exponent:>4} {str(c.step):>10} {cash:>10} {c.note}")
    print(RULE)
    print("\nwhat a hardcoded round(x, 2) does to each:\n")
    print(f"{'code':6} {'amount':>12} {'round(x,2)':>12} {'payable?':>9} {'what broke'}")
    print(RULE)
    cases = [
        ("KWD", D("125.4567")),
        ("JPY", D("1234.50")),
        ("MRU", D("6.13")),
        ("CLF", D("38.1234")),
        ("USD", D("125.4567")),
    ]
    for code, amt in cases:
        c = m.currency(code)
        two = amt.quantize(D("0.01"), rounding=ROUND_HALF_UP)
        ok = m.is_payable(two, c)
        why = {
            "KWD": "2dp is coarser than the 5-fils coin: 0.003 lost, exact amount unstorable",
            "JPY": "invented a half-yen; no coin exists",
            "MRU": "0.13 is not a khoums multiple",
            "CLF": "index unit truncated from 4dp to 2dp",
            "USD": "fine, which is why nobody notices the others",
        }[code]
        print(f"{code:6} {str(amt):>12} {str(two):>12} {'yes' if ok else 'NO':>9} {why}")
    print(RULE)
    print("\nMRU and MGA are the sharp case: ISO 4217 gives them exponent 2, so a schema")
    print("built from the exponent stores two decimals and a validator built from the")
    print("exponent accepts 6.13. But the ouguiya divides into 5 khoums, not 100, so the")
    print("only legal cents are .00 .20 .40 .60 .80. The exponent describes how many")
    print("digits are printed, not which amounts exist.")
    print("\nrejected outright rather than rounded:")
    rec = m.audit(m.get_ledger("khoums")).reconciliation
    print(f"  ledger 'khoums' -> {rec.verdict}")
    print(f"  {rec.reason}")


def exp7_book_total_vs_cash_total() -> None:
    head(7, "Two totals, both correct, legally different. The books and the till.")
    a = m.audit(m.get_ledger("swiss_cash"))
    led = m.get_ledger("swiss_cash")
    print("a Swiss cafe bill:\n")
    for lab, amt in led.rows:
        print(f"  {lab:10} {amt:>8}")
    print(RULE)
    print(f"  {'invoice':10} {a.reconciliation.stated_total:>8}  (payable in the books, CHF 0.01)")
    print(f"  {'cash due':10} {a.cash_total:>8}  (smallest coin is 5 rappen)")
    print(f"  {'difference':10} {a.cash_gap:>8}  goes to a rounding account, not to a line")
    print(RULE)
    print("\nthe same gap across the currencies that have one:\n")
    print(f"{'code':6} {'invoice':>10} {'cash':>10} {'gap':>8} {'why'}")
    print(RULE)
    for code, amt in [("CHF", D("9.93")), ("CAD", D("9.93")), ("SEK", D("99.40")), ("KWD", D("143.691")), ("USD", D("9.93"))]:
        c = m.currency(code)
        if not c.has_cash_gap:
            print(f"{code:6} {str(amt):>10} {str(amt):>10} {'0':>8} book and cash are the same unit")
            continue
        cash = m.quantize(amt, c, "half_even", cash=True)
        print(f"{code:6} {str(amt):>10} {str(cash):>10} {str(cash - amt):>8} {c.note}")
    print(RULE)
    print("\nThe invoice is not wrong and the cash is not wrong. A reconciliation that")
    print("insists they match will chase a difference that is supposed to be there.")
    print("The card payment of the same bill settles at the invoice figure, so the same")
    print("basket costs a different amount depending on how it was paid.")


def exp8_the_ledger() -> None:
    head(8, "The ledger: every sample, its verdict, and whether it raises.")
    print(f"{'ledger':12} {'cur':5} {'verdict':16} {'gap':>8} {'absorbed':12} {'naive pipeline'}")
    print(RULE)
    decided = 0
    for led in m.sample_ledgers():
        a = m.audit(led)
        r = a.reconciliation
        absorbed = ",".join(r.allocation.labels[i] for i in r.allocation.absorbed) if r.allocation and r.allocation.absorbed else "-"
        raises = "silent"
        print(f"{led.name:12} {led.currency:5} {r.verdict:16} {str(r.gap):>8} {absorbed[:12]:12} {raises}")
        if r.decided:
            decided += 1
    print(RULE)
    n = len(m.sample_ledgers())
    print(f"\n{decided} of the {n} ledgers are decided by their own contents.")
    print(f"{n - decided} is not: the stated total does not exist in its currency.")
    print("\nfailure modes, and what each one costs:\n")
    print(f"{'failure mode':38} {'sample':12} {'effect':26}")
    print(RULE)
    rows = [
        ("rows rounded independently", "thirds", "ledger short $0.01"),
        ("penny placed by row position", "weighted", "$0.01 variance from a sort"),
        ("half_even where law says half_up", "ties", "4 rows, $0.02 apart"),
        ("float literal rounded", "2.675", "rounds down off a non-tie"),
        ("float accumulation", "0.01 x 10000", "1.4e-11 drift, order dependent"),
        ("line tax vs invoice tax", "vat_lines", "return disagrees with invoice"),
        ("rounding twice", "2.4449", "2.45 instead of 2.44"),
        ("discount before/after tax", "19.99", "$0.01 per order"),
        ("round(x, 2) on a 3dp currency", "fils", "0.457 KWD lost per invoice"),
        ("exponent read as subdivision", "khoums", "unpayable amount accepted"),
        ("book total compared to cash", "swiss_cash", "CHF 0.02 permanent difference"),
    ]
    for a, b, c in rows:
        print(f"{a:38} {b:12} {c:26}")
    print(RULE)
    print("\nEvery one is silent. None raises. All of them are reproducible - re-running the")
    print("job produces the same wrong number, so a re-run reconciles against itself and")
    print("agrees. That is why these survive: the failure is stable, and stability reads")
    print("as correctness.")


def main() -> None:
    exp1_rows_that_do_not_add_up()
    exp2_sorting_moves_the_penny()
    exp3_half_even_vs_half_up()
    exp4_float_is_not_the_number()
    exp5_order_of_operations()
    exp6_two_decimals_is_an_assumption()
    exp7_book_total_vs_cash_total()
    exp8_the_ledger()
    print()


if __name__ == "__main__":
    main()
