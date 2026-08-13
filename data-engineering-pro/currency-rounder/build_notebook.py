"""Generate demo.ipynb with money.py / evidence.py / make_chart.py embedded.

The notebook writes those three modules to disk from embedded source, so it runs
on Colab or Binder with no clone step, and there is no second copy of the logic
to drift out of sync with the repo.

Run: python3 build_notebook.py
"""

from __future__ import annotations

import json
import pathlib

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/currency-rounder"
HERE = pathlib.Path(__file__).parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def embed(fname: str) -> str:
    """A cell that writes `fname` to disk from its real source, base64-encoded.

    Base64 rather than a here-doc: the modules contain their own triple-quoted
    docstrings, so any quoting scheme that nests Python source inside Python
    source is one docstring away from breaking.
    """
    import base64

    src = (HERE / fname).read_text()
    blob = base64.b64encode(src.encode()).decode()
    var = "_" + fname[:-3] + "_b64"
    chunks = [blob[i : i + 88] for i in range(0, len(blob), 88)]
    literal = "\n    ".join(f'"{c}"' for c in chunks)
    return (
        f"import base64, pathlib\n"
        f"{var} = (\n    {literal}\n)\n"
        f'_src = base64.b64decode({var}).decode()\n'
        f'pathlib.Path("{fname}").write_text(_src)\n'
        f'print("wrote {fname}:", len(_src.splitlines()), "lines")'
    )


CELLS = [
    md(
        f"""
# currency-rounder - the cent a ledger loses, and where it goes

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 143 - Data Engineering Pro - [phoebe-the-builder](https://github.com/{REPO})**

A rounding function returns a number. It cannot return the fact that the rows no longer add up.

Round every line of a ledger correctly, to the nearest cent, using the mode your language
gives you by default, and the ledger can still be a cent short of its own total. No row is
wrong. Nothing raises. The report reconciles against itself on a re-run, because the error
is perfectly reproducible.

This notebook walks through eight mechanisms that produce a wrong money figure silently:

1. Independent rounding does not preserve a sum
2. Sorting the ledger moves the cent onto a different row
3. `half_even` is the language default; `half_up` is what most tax codes say
4. The float you rounded is not the number you typed
5. Rounding does not commute - line tax vs invoice tax, discount before vs after
6. Two decimal places is an assumption, and ISO 4217 does not share it
7. The books and the till settle at different totals, and both are correct
8. The ledger: what each failure costs, and whether anything raises

Then it builds a **three-verdict** audit - `exact`, `reconciled`, `irreconcilable` - and a
six-panel figure. Everything below runs on the standard library plus matplotlib.
"""
    ),
    md(
        """
## Step 0 - write the modules

The notebook carries the real source of `money.py`, `evidence.py` and `make_chart.py`
inline and writes them to disk. That keeps it self-contained on Colab and keeps the
notebook honest: it runs the same code the repo ships, not a simplified retelling.
"""
    ),
    code(embed("money.py")),
    code(embed("evidence.py")),
    code(embed("make_chart.py")),
    md(
        """
## Step 1 - the cent that goes missing

A $100.00 refund, split three ways. Each share is $33.33 recurring. Round each one to
the nearest cent, the way any correct rounding function would, and add them up.
"""
    ),
    code(
        """
from decimal import Decimal as D
import money as m

usd = m.currency("USD")
exact = D(100) / 3
per_row = m.quantize(exact, usd, "half_even")

print(f"exact share      : {exact}")
print(f"rounded to a cent: {per_row}")
print(f"three of those   : {per_row * 3}")
print(f"stated total     : {D('100.00')}")
print(f"gap              : {per_row * 3 - D('100.00')}")
"""
    ),
    md(
        """
Nothing here is a mistake. `33.3333...` really is nearer to `33.33` than to `33.34`, so
every individual answer is right and the set of them is wrong.

The fix is not a better rounding mode. It is to stop rounding the rows independently and
**allocate** the total instead: give each row its floor, then hand the leftover increments
to the rows with the largest remainders. The sum is then exact by construction.
"""
    ),
    code(
        """
alloc = m.allocate(D("100.00"), [D(1), D(1), D(1)], usd, ["alice", "bob", "carol"])

print("parts    :", {k: str(v) for k, v in alloc.by_label().items()})
print("sum      :", sum(alloc.parts))
print("absorbed :", [alloc.labels[i] for i in alloc.absorbed], f"({alloc.residual_units} increment)")
print("tie was broken by position:", alloc.tie_broken)
"""
    ),
    md(
        """
## Step 2 - the total is stable; the rows are not

Read that last line again. All three remainders are equal, so *nothing in the data* says
which row should absorb the cent. The tie is settled by position - whichever row happens
to come first in the input.

That means a sort is enough to move money between rows.
"""
    ),
    code(
        """
for label, order in [
    ("as the file arrived", ["carol", "alice", "bob"]),
    ("sorted by name     ", ["alice", "bob", "carol"]),
    ("reversed           ", ["bob", "carol", "alice"]),
]:
    a = m.allocate(D("100.00"), [D(1)] * 3, usd, order)
    d = a.by_label()
    print(f"{label}  alice {d['alice']}  bob {d['bob']}  carol {d['carol']}   sum {sum(a.parts)}")
"""
    ),
    md(
        """
Every order sums to exactly `100.00`. No two orders agree on who paid the extra cent.

A month-on-month variance report on alice will show a $0.01 movement that is entirely an
artefact of how the input file happened to be sorted. This module reports it as
`tie_broken` / `order_sensitive` rather than hiding it, because the tie-break is a
**preference**, not a finding.
"""
    ),
    md(
        """
## Step 3 - the default is not the law

Python's `round()`, `Decimal`'s default context and IEEE 754 all use **round-half-to-even**
(banker's rounding): a tie goes to whichever neighbour is even. Most tax authorities, and
Excel's `ROUND()`, use **round-half-up**: a tie always goes away from zero.

They disagree on exactly half of all ties.
"""
    ),
    code(
        """
from decimal import getcontext

print("Decimal default context :", getcontext().rounding)
print("round(0.5), round(1.5), round(2.5) =", round(0.5), round(1.5), round(2.5))
print()
print(f"{'amount':10} {'cent units':>11} {'half_even':>10} {'half_up':>9}  agree")
print("-" * 52)
for amt in ("1.005", "1.015", "1.025", "1.035"):
    a = D(amt)
    he, hu = m.quantize(a, usd, "half_even"), m.quantize(a, usd, "half_up")
    print(f"{amt:10} {str(a / usd.step):>11} {str(he):>10} {str(hu):>9}  {'yes' if he == hu else 'NO'}")
"""
    ),
    md(
        """
Over many rows `half_even` has almost no bias, which is exactly why it is the numerical
default. `half_up` has a deliberate upward bias, which is exactly why tax codes specify it.

There is a third property worth checking that neither name advertises: whether a charge
and its own refund cancel to zero.
"""
    ),
    code(
        """
print(f"{'mode':12} {'+0.005':>8} {'-0.005':>8}   charge + refund == 0 ?")
print("-" * 52)
for mode in m.MODES:
    p, n = m.quantize(D("0.005"), usd, mode), m.quantize(D("-0.005"), usd, mode)
    print(f"{mode:12} {str(p):>8} {str(n):>8}   {'yes' if p + n == 0 else 'NO'}")
"""
    ),
    md(
        """
`ceiling` and `floor` are the two that fail. Under an "always round up in our favour"
policy, issuing a charge and then its exact refund leaves a cent behind **per transaction,
permanently** - and the books still balance on both days.
"""
    ),
    md(
        """
## Step 4 - the float you rounded is not the number you typed

`round(2.675, 2)` returns `2.67`. That looks like a rounding-mode bug and is not one:
the float nearest to `2.675` is slightly *below* it, so there is no tie to break.
"""
    ),
    code(
        """
for lit in ("0.1", "2.675", "1.005", "0.5"):
    print(f"{lit:8} -> {m.exact_value_of_float(float(lit))}")

print()
print(f"{'literal':9} {'round(float,2)':>15} {'Decimal half_up':>17}  agree")
print("-" * 52)
for lit in ("2.675", "1.005", "0.145", "8.835", "1.115"):
    fl, dec, differ = m.float_round_disagrees(lit, 2)
    print(f"{lit:9} {str(fl):>15} {str(dec):>17}  {'NO' if differ else 'yes'}")

print()
total = 0.0
for _ in range(10000):
    total += 0.01
print(f"0.01 added 10,000 times as float : {total!r}")
print(f"the same in Decimal              : {D('0.01') * 10000}")
"""
    ),
    md(
        """
## Step 5 - rounding does not commute

Tax on each line, rounded, then summed is not the same as the lines summed, taxed, then
rounded once. Both are defensible: the printed invoice must show a payable amount per line,
while a tax return applies one rate to one base. EU VAT rounding is set per member state
rather than harmonised, so which one is "correct" depends on jurisdiction.
"""
    ),
    code(
        """
eur = m.currency("EUR")
for nets, rate, label in [
    ([D("12.99"), D("7.45"), D("31.20")], D("0.21"), "21% VAT, three lines"),
    ([D("0.10")] * 3, D("0.175"), "17.5% on three 10c lines"),
    ([D("9.99")] * 7, D("0.0825"), "8.25% on seven identical lines"),
]:
    line = m.tax_line_level(nets, rate, eur)
    inv = m.tax_invoice_level(nets, rate, eur)
    print(f"{label:34} per-line {line}   invoice-level {inv}   "
          f"{'DIFFER by ' + str(line - inv) if line != inv else 'agree'}")
"""
    ),
    md(
        """
The first basket agrees. That is what makes this dangerous: the disagreement is
intermittent, so a test written against one basket passes and the next basket is short.

The same non-commutativity shows up in rounding twice, which any pipeline that stores an
intermediate at higher precision and rounds again at report time is doing.
"""
    ),
    code(
        """
print(f"{'value':9} {'->2dp':>8} {'->3dp->2dp':>12} {'->4->3->2':>11}")
print("-" * 44)
for v in ("2.4449", "1.2349", "0.4449", "9.9949"):
    x = D(v)
    print(f"{v:9} {str(m.chain_round(x, [2], 'half_up')):>8} "
          f"{str(m.chain_round(x, [3, 2], 'half_up')):>12} "
          f"{str(m.chain_round(x, [4, 3, 2], 'half_up')):>11}")
"""
    ),
    md(
        """
## Step 6 - two decimal places is an assumption

`round(amount, 2)` is not a currency operation. ISO 4217 gives JPY an exponent of 0, KWD
and BHD an exponent of 3, and CLF an exponent of 4.

The sharpest case is MRU (and MGA). ISO assigns them exponent 2, so a schema built from
the exponent stores two decimals and a validator built from the exponent accepts `6.13` -
but the ouguiya divides into **5** khoums, not 100. The only legal cents are
`.00 .20 .40 .60 .80`. The exponent describes how many digits get printed, not which
amounts exist.
"""
    ),
    code(
        """
print(f"{'code':6} {'exp':>4} {'book':>8} {'cash':>8}  note")
print("-" * 78)
for c in ("USD", "JPY", "KWD", "CHF", "SEK", "CAD", "MRU", "CLF"):
    cur = m.currency(c)
    cash = str(cur.cash_step) if cur.cash_step is not None else "-"
    print(f"{cur.code:6} {cur.exponent:>4} {str(cur.step):>8} {cash:>8}  {cur.note}")

print()
print("is 6.13 a payable amount?")
for c in ("USD", "MRU"):
    print(f"  {c}: {m.is_payable(D('6.13'), m.currency(c))}")
"""
    ),
    md(
        """
So there is a third verdict, and it is not a rounding preference. When the stated total is
not a multiple of the currency's increment, **no** set of payable rows can sum to it. The
tool reports that instead of quietly rounding until the number agrees.
"""
    ),
    code(
        """
rec = m.audit(m.get_ledger("khoums")).reconciliation
print("verdict:", rec.verdict)
print("reason :", rec.reason)
print("allocation returned:", rec.allocation)
"""
    ),
    md(
        """
## Step 7 - the books and the till

Switzerland books in rappen (CHF 0.01) and pays in 5-rappen coins. Sweden withdrew its ore
coins and rounds cash to the whole krona. Canada withdrew the penny in 2013 and rounds cash
to a nickel. In each, the invoice total and the cash total are **different numbers, both
legally correct**, and the card payment of the same basket settles at the invoice figure.
"""
    ),
    code(
        """
a = m.audit(m.get_ledger("swiss_cash"))
led = m.get_ledger("swiss_cash")
for lab, amt in led.rows:
    print(f"  {lab:8} {amt:>7}")
print("-" * 26)
print(f"  {'invoice':8} {a.reconciliation.stated_total:>7}   (books, CHF 0.01)")
print(f"  {'cash due':8} {a.cash_total:>7}   (smallest coin, CHF 0.05)")
print(f"  {'gap':8} {a.cash_gap:>7}   -> a rounding account, not a line item")
"""
    ),
    md(
        """
A reconciliation that insists the two match will chase a difference that is supposed to be
there.

## Step 8 - the corpus, audited

Eight sample ledgers, each isolating one mechanism. The verdict column is the API:
`exact` (nothing was rounded away), `reconciled` (a residual existed and named rows
absorbed it), `irreconcilable` (the stated total does not exist in this currency).
"""
    ),
    code(
        """
print(f"{'ledger':12} {'cur':5} {'verdict':16} {'gap':>8} {'absorbed':12} raises?")
print("-" * 72)
for led in m.sample_ledgers():
    au = m.audit(led)
    r = au.reconciliation
    absorbed = ",".join(r.allocation.labels[i] for i in r.allocation.absorbed) if r.allocation and r.allocation.absorbed else "-"
    print(f"{led.name:12} {led.currency:5} {r.verdict:16} {str(r.gap):>8} {absorbed[:12]:12} silent")
print("-" * 72)
n = len(m.sample_ledgers())
dec = sum(1 for led in m.sample_ledgers() if m.audit(led).decided)
print(f"{dec} of {n} decided by their own contents; {n - dec} is not.")
"""
    ),
    md(
        """
Every failure mode in this notebook is **silent**. None of them raises. And all of them are
**reproducible** - re-running the job produces the same wrong number, so a reconciliation
against yesterday agrees.

That is why these survive in production: the failure is stable, and stability reads as
correctness.

## The figure

Six panels, every value computed from `money.py` rather than typed in.
"""
    ),
    code(
        """
import make_chart
import matplotlib.pyplot as plt
from IPython.display import Image, display

fig, axes = plt.subplots(3, 2, figsize=(14.5, 15.6))
fig.patch.set_facecolor("white")
make_chart.p1_shortfall(axes[0][0])
make_chart.p2_who_pays(axes[0][1])
make_chart.p3_mode_bias(axes[1][0])
make_chart.p4_float_drift(axes[1][1])
make_chart.p5_minor_units(axes[2][0])
make_chart.p6_verdicts(axes[2][1])
fig.suptitle("currency-rounder - the cent a ledger loses, and where it goes",
             fontsize=16, fontweight="bold", color=make_chart.INK, x=0.055, ha="left", y=0.985)
fig.text(0.055, 0.962, "every value computed from money.py, none typed in",
         fontsize=9.6, color=make_chart.MUTED, ha="left")
fig.tight_layout(rect=[0.012, 0.008, 0.988, 0.952])
fig.savefig("rounding_audit_nb.png", dpi=110, facecolor="white")
plt.close(fig)
display(Image("rounding_audit_nb.png"))
"""
    ),
    md(
        """
## Summary

| mechanism | sample | what it costs | raises? |
|---|---|---|---|
| rows rounded independently | `thirds` | ledger $0.01 short | no |
| penny placed by row position | `weighted` | $0.01 variance from a sort order | no |
| `half_even` where law says `half_up` | `ties` | four rows, $0.02 apart | no |
| float literal rounded | `2.675` | rounds down off a non-tie | no |
| float accumulation | `0.01` x 10,000 | sub-cent drift, order dependent | no |
| line tax vs invoice tax | `vat_lines` | the return disagrees with the invoice | no |
| rounding twice | `2.4449` | `2.45` instead of `2.44` | no |
| discount before vs after tax | `19.99` | $0.01 per order | no |
| `round(x, 2)` on a 3-decimal currency | `fils` | amount becomes unstorable | no |
| exponent read as the subdivision | `khoums` | an unpayable amount accepted | no |
| book total compared against cash | `swiss_cash` | CHF 0.02 permanent difference | no |

The design rule underneath all of it: **a function that returns one number cannot report
what it had to assume to get there.** So `classify_delimiter`-style enumeration applies to
money too - return the verdict, the residual, the rows that absorbed it, and the settings
this ledger never exercised.

## Try your own
"""
    ),
    code(
        """
# Paste your own rows below and re-run this cell.
#
# my_rows = [("north", D("1250.005")), ("south", D("1250.005")), ("east", D("2500.01"))]
# my_total = D("5000.02")
# my_currency = m.currency("USD")          # try "JPY", "KWD", "CHF", "MRU"
#
# rec = m.reconcile([a for _, a in my_rows], my_currency, my_total, "half_up",
#                   [l for l, _ in my_rows])
# print(rec.verdict, "-", rec.reason)
# if rec.allocation:
#     for lab, part in rec.allocation.by_label().items():
#         print(f"  {lab:10} {part}")
#     print("  sum:", sum(rec.allocation.parts))
# for note in rec.untested:
#     print("  untested:", note)

# Or run the full evidence script, which prints every table in the README:
# import evidence; evidence.main()
"""
    ),
    md(
        f"""
---

**Part of [phoebe-the-builder]({'https://github.com/' + REPO}) - Day 143, Data Engineering Pro.**

- [`README.md`](https://github.com/{REPO}/tree/main/{PATH}) - the write-up, with every table above
- `python3 test_money.py` - the suite (structural assertions only; it never asserts what a
  given CPython's `round()` returns)
- `python3 evidence.py` - reproduces every number in the README
- `streamlit run app.py` - the UI, verdict first and the table last
"""
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out = HERE / "demo.ipynb"
    out.write_text(json.dumps(NB, indent=1))
    print(f"wrote {out} - {len(CELLS)} cells")
