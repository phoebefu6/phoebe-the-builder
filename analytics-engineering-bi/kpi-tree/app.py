from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from kpitree import (
    REVENUE_TREE,
    SAMPLE_AFTER,
    SAMPLE_BEFORE,
    decomposition_summary,
    narrate,
)

st.set_page_config(page_title="KPI Tree / Driver Decomposition", layout="wide")
st.title("KPI Tree / Driver Decomposition")
st.caption('Answer "why did revenue move?" with an exact, no-residual split across its drivers.')

tree = REVENUE_TREE
st.markdown(f"**{tree.name} = " + " × ".join(tree.label(d) for d in tree.drivers) + "**")

st.subheader("Driver values: before → after")
before, after = {}, {}
cols = st.columns(len(tree.drivers))
for i, d in enumerate(tree.drivers):
    with cols[i]:
        st.markdown(f"**{tree.label(d)}**")
        before[d] = st.number_input(f"{d} before", value=float(SAMPLE_BEFORE[d]), key=f"b_{d}", format="%.4f")
        after[d] = st.number_input(f"{d} after", value=float(SAMPLE_AFTER[d]), key=f"a_{d}", format="%.4f")

summary = decomposition_summary(tree, before, after)

c1, c2, c3 = st.columns(3)
c1.metric(f"{tree.name} before", f"{summary['before']:,.0f}")
c2.metric(f"{tree.name} after", f"{summary['after']:,.0f}", f"{summary['pct_change']:+.1%}")
c3.metric("Total change", f"{summary['total_change']:,.0f}")

st.info(narrate(summary))

# waterfall chart
contribs = summary["contributions"]
labels = ["Before"] + [c.label for c in contribs] + ["After"]
running = summary["before"]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(0, summary["before"], color="#6b7280")
ax.text(0, summary["before"], f"{summary['before']:,.0f}", ha="center", va="bottom", fontsize=8)
for i, c in enumerate(contribs, start=1):
    color = "#2e7d32" if c.contribution >= 0 else "#c0553b"
    bottom = running if c.contribution >= 0 else running + c.contribution
    ax.bar(i, abs(c.contribution), bottom=bottom, color=color)
    ax.text(i, bottom + abs(c.contribution), f"{c.contribution:+,.0f}", ha="center", va="bottom", fontsize=8)
    running += c.contribution
ax.bar(len(contribs) + 1, summary["after"], color="#3b6fd6")
ax.text(len(contribs) + 1, summary["after"], f"{summary['after']:,.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_title(f"Why {tree.name} moved — driver waterfall", fontsize=13, weight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Contribution table")
st.dataframe(
    pd.DataFrame([{
        "driver": c.label,
        "before": c.before,
        "after": c.after,
        "% change": f"{c.pct_change:+.1%}",
        "contribution": round(c.contribution, 1),
        "share of change": f"{c.share:+.1%}",
    } for c in contribs]),
    use_container_width=True,
)
st.caption(f"Residual (unexplained): {summary['residual']:.4f} — LMDI is exact, so this is ~0.")
