"""Streamlit UI for currency-rounder.

Layout is deliberate: the verdict comes first, the residual and who absorbed it
second, and the reconciled table last. A table rendered above the caveats reads
as the answer, so the caveats go above the table.
"""

from __future__ import annotations

from decimal import Decimal as D
from decimal import InvalidOperation

import pandas as pd
import streamlit as st

import money as m

st.set_page_config(page_title="currency-rounder", page_icon="¢", layout="wide")

st.title("currency-rounder")
st.caption(
    "A rounding function returns a number. It cannot return the fact that the rows "
    "no longer add up, that the cent landed on a row chosen by sort order, or that "
    "the amount is not payable in the currency it is denominated in."
)

# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Input")
    sample_names = [led.name for led in m.sample_ledgers()]
    source = st.radio("Source", ["sample ledger", "paste rows"], index=0)

    if source == "sample ledger":
        chosen = st.selectbox("Ledger", sample_names, index=0)
        led = m.get_ledger(chosen)
        st.caption(led.story)
        rows_text = "\n".join(f"{lab},{amt}" for lab, amt in led.rows)
        cur_code = led.currency
        stated_default = str(led.stated_total) if led.stated_total is not None else ""
    else:
        rows_text = st.text_area(
            "label,amount per line", "alice,33.333333\nbob,33.333333\ncarol,33.333333", height=150
        )
        cur_code = "USD"
        stated_default = "100.00"

    cur_code = st.selectbox(
        "Currency", sorted(m.CURRENCIES), index=sorted(m.CURRENCIES).index(cur_code)
    )
    mode = st.selectbox("Rounding mode", list(m.MODES), index=0)
    stated = st.text_input("Stated total (blank = use the rounded sum)", stated_default)

cur = m.currency(cur_code)

# ------------------------------------------------------------------ parse

labels, amounts, parse_errors = [], [], []
for i, line in enumerate(rows_text.strip().splitlines(), 1):
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2:
        parse_errors.append(f"line {i}: expected 'label,amount', got {line!r}")
        continue
    try:
        amounts.append(D(parts[1]))
        labels.append(parts[0])
    except InvalidOperation:
        parse_errors.append(f"line {i}: {parts[1]!r} is not a number")

if parse_errors:
    for e in parse_errors:
        st.error(e)
if not amounts:
    st.stop()

stated_total = None
if stated.strip():
    try:
        stated_total = D(stated.strip())
    except InvalidOperation:
        st.error(f"stated total {stated!r} is not a number")
        st.stop()

rec = m.reconcile(amounts, cur, stated_total, mode, labels)

# ------------------------------------------------------------------ verdict

BANNER = {
    "exact": (st.success, "EXACT", "nothing was rounded away and the rows already summed right"),
    "reconciled": (st.warning, "RECONCILED", "a residual existed; it was allocated to named rows"),
    "irreconcilable": (
        st.error,
        "IRRECONCILABLE",
        "no set of payable rows in this currency can sum to the stated total",
    ),
}
fn, word, gloss = BANNER[rec.verdict]
fn(f"**{word}** - {gloss}")
st.write(rec.reason)

c1, c2, c3, c4 = st.columns(4)
c1.metric("stated total", str(rec.stated_total))
c2.metric("sum of independently rounded rows", str(rec.naive_sum))
c3.metric("gap", str(rec.gap), delta=f"{rec.gap_units:+d} increments" if rec.gap_units is not None else None)
c4.metric("currency increment", f"{cur.step}  ({cur.code})")

if rec.verdict == "irreconcilable":
    st.info(
        f"{cur.code} moves in steps of {cur.step}. {cur.note}. "
        "This tool reports rather than repairs: the honest resolution is to ask the "
        "sender what amount they meant, not to round until the number agrees."
    )

# ------------------------------------------------------------------ caveats

with st.container():
    st.subheader("What this ledger does not settle")
    notes = list(rec.untested)
    if rec.allocation and rec.allocation.tie_broken:
        notes.append(
            "two rows tie on the remainder, so the cent was placed by row position. "
            "Re-sorting the input moves it to a different row, and the total stays right."
        )
    if not cur.step_is_power_of_ten:
        notes.append(
            f"{cur.code}'s increment ({cur.step}) is not a power of ten, so 'round to "
            f"{cur.exponent} decimal places' is not the same operation as rounding to this currency."
        )
    if cur.has_cash_gap:
        cash = m.quantize(rec.stated_total, cur, "half_even", cash=True) if rec.decided else None
        if cash is not None:
            notes.append(
                f"cash settles at {cash} ({cur.cash_step} increment), "
                f"a difference of {cash - rec.stated_total} from the invoice. Both figures are correct."
            )
    if not notes:
        notes = ["nothing flagged: this ledger exercises every setting the tool checks."]
    for nte in notes:
        st.markdown(f"- {nte}")

# ------------------------------------------------------------------ table

st.subheader("Rows")
alloc = rec.allocation
frame = {
    "row": labels,
    "exact amount": [str(a) for a in amounts],
    "rounded independently": [str(p) for p in rec.naive_parts],
}
if alloc is not None:
    frame["reconciled"] = [str(p) for p in alloc.parts]
    frame["absorbed a residual"] = ["yes" if i in alloc.absorbed else "" for i in range(len(labels))]
    frame["moved by"] = [str(alloc.parts[i] - rec.naive_parts[i]) for i in range(len(labels))]
df = pd.DataFrame(frame)
st.dataframe(df, width="stretch", hide_index=True)

if alloc is not None:
    st.caption(
        f"reconciled rows sum to {sum(alloc.parts)}, which equals the stated total exactly. "
        f"{rec.gap_units:+d} increment(s) were moved across "
        f"{len(alloc.absorbed)} row(s): {', '.join(alloc.labels[i] for i in alloc.absorbed) or 'none'}."
    )

# ------------------------------------------------------------------ modes

st.subheader("The same rows under every rounding mode")
mode_rows = []
for md in m.MODES:
    r = m.reconcile(amounts, cur, stated_total, md, labels)
    mode_rows.append(
        {
            "mode": md,
            "sum of rounded rows": str(r.naive_sum),
            "gap": str(r.gap),
            "verdict": r.verdict,
            "charge cancels its refund": "yes" if m.mode_is_symmetric(md) else "NO",
        }
    )
st.dataframe(pd.DataFrame(mode_rows), width="stretch", hide_index=True)
st.caption(
    "Python's round(), Decimal's default context and IEEE 754 all use half_even. "
    "Most tax codes and Excel's ROUND() use half_up. Neither is the default of the other."
)

st.divider()
st.caption(
    "Part of phoebe-the-builder, Day 143. Reproduce every claim with "
    "`python3 evidence.py`; run the suite with `python3 test_money.py`."
)
