"""Streamlit front end: split one warehouse invoice, and watch the answer move.

The point of the app is that there is no button that returns the right number. There is a
menu of defensible rules, a range of allocations nobody could object to, and a portion of
the bill that belongs to nobody at all.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import costs as C
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Warehouse Cost Attribution", layout="wide")

st.title("Who spent the forty thousand?")
st.caption(
    "A warehouse invoice is a JOINT cost: storage is paid once however many teams need the "
    "table, an upstream model is built once for everyone, the second read of the day is "
    "nearly free, and the reservation is owed the moment anybody shows up. A joint cost has "
    "no unique owner."
)

with st.sidebar:
    st.header("The month")
    st.metric("Invoice", f"${C.INVOICE:,.0f}")
    st.metric("Reservation floor", f"${C.RESERVED_FLOOR:,.0f}",
              delta=f"{C.RESERVED_FLOOR/C.INVOICE:.0%} of the bill", delta_color="off")
    st.divider()
    chosen = st.multiselect("Methods to compare", list(C.METHODS),
                            default=["direct_bytes", "equal_split", "shapley", "marginal"])
    show_core = st.checkbox("Show the core (what nobody could object to)", value=True)

allocs = {m: C.METHODS[m]() for m in chosen} if chosen else {}

st.subheader("The same month, billed several ways")
if allocs:
    df = pd.DataFrame({m: {t: allocs[m][t] for t in C.TEAM_NAMES} for m in allocs})
    df.index.name = "team"
    view = df.copy()
    view["min"] = df.min(axis=1)
    view["max"] = df.max(axis=1)
    view["ratio"] = (view["max"] / view["min"].clip(lower=1)).round(1)
    st.dataframe(view.style.format("{:,.0f}", subset=list(df.columns) + ["min", "max"])
                 .format("{:.1f}x", subset=["ratio"]), use_container_width=True)

    worst = view["ratio"].idxmax()
    st.warning(
        f"**{worst}** is billed between **${view.loc[worst,'min']:,.0f}** and "
        f"**${view.loc[worst,'max']:,.0f}** — a factor of {view.loc[worst,'ratio']:.0f} — "
        f"across rules that are all defensible. Every column adds up to the same invoice."
    )
    tops = {m: max(allocs[m], key=lambda k: allocs[m][k]) for m in allocs}
    st.info("**Most expensive team, by method:** " +
            "  ·  ".join(f"`{m}` → {t}" for m, t in tops.items()))
else:
    st.info("Pick at least one method in the sidebar.")

if show_core:
    st.subheader("Fair is a range, not a number")
    rows = []
    sh = C.shapley()
    for t in C.TEAM_NAMES:
        lo, hi = C.core_range(t)
        rows.append({"team": t, "core min": lo, "core max": hi, "width": hi - lo,
                     "width as % of invoice": (hi - lo) / C.INVOICE, "shapley": sh[t]})
    st.dataframe(pd.DataFrame(rows).style
                 .format("{:,.0f}", subset=["core min", "core max", "width", "shapley"])
                 .format("{:.1%}", subset=["width as % of invoice"]),
                 use_container_width=True, hide_index=True)
    st.caption(
        "The core is every allocation no group of teams would walk out of. It is non-empty "
        "here — and wide enough that it rules out the two rules people reach for first "
        "(`direct_bytes`, `equal_split`) and almost nothing else."
    )

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for i, t in enumerate(C.TEAM_NAMES):
        lo, hi = C.core_range(t)
        ax.barh(i, hi - lo, left=lo, height=0.5, color="#4a7c8c", alpha=0.25)
        ax.scatter(sh[t], i, s=60, marker="D", color="#4a7c8c", zorder=3)
        for m in allocs:
            ax.scatter(allocs[m][t], i, s=26, color="#c0392b", zorder=4)
    ax.set_yticks(range(len(C.TEAM_NAMES)))
    ax.set_yticklabels(C.TEAM_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$ billed  (shaded = the core, diamond = Shapley, red = your chosen methods)")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

st.subheader("What no method can fix")
a, b, c = st.columns(3)
raw_m = sum(C.raw_marginal().values())
raw_s = sum(C.raw_standalone().values())
a.metric("Marginal cost recovers", f"{raw_m/C.INVOICE:.1%}",
         delta=f"${C.INVOICE-raw_m:,.0f} unfunded", delta_color="inverse")
b.metric("Standalone cost recovers", f"{raw_s/C.INVOICE:.0%}",
         delta=f"${raw_s-C.INVOICE:,.0f} over", delta_color="inverse")
c.metric("Orphaned jobs: blame vs saving",
         f"{C.shapley()['scheduled_unowned']/C.unowned_cost():.0f}x apart",
         delta=f"${C.shapley()['scheduled_unowned']:,.0f} charged, ${C.unowned_cost():,.0f} saved",
         delta_color="off")
st.caption(
    "Attribution answers *who consumed it*. It does not answer *what would we save*. Budget "
    "decisions need the second, and the two differ by an order of magnitude on the jobs "
    "nobody claims."
)

st.divider()
st.caption("Day 161 of Phoebe's FDE portfolio. Run `python evidence.py` for the full "
           "ten-section study, or open demo.ipynb.")
