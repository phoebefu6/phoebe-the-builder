from __future__ import annotations

# Streamlit UI for the Pivot Narrator. Shows the pivot, then the paragraph that
# should have been under it. Deterministic - no LLM, every sentence is
# arithmetic with a threshold.
import pandas as pd
import streamlit as st
from narrate import (
    CONCENTRATION_ALERT,
    LIFT_THRESHOLD,
    MIN_CELL_SHARE,
    expected_matrix,
    lift_matrix,
    narrate,
    notable_cells,
    sample_pivots,
)

st.set_page_config(page_title="Pivot Narrator", page_icon="🗣️", layout="wide")

st.title("🗣️ Pivot Narrator")
st.caption(
    "A crosstab is a wall of numbers. The three facts a reader needs - what dominates, "
    "what moved, and where the interaction is - are all in there and none are legible. "
    "This writes the paragraph. Deterministically: no model call, every sentence is "
    "arithmetic you can trace to a cell."
)

cur_default, prev_default = sample_pivots()

with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Bundled sample", "Upload CSV"], index=0)
    st.header("Labels")
    metric = st.text_input("Metric name", "revenue")
    unit = st.text_input("Unit prefix", "$")
    st.header("Thresholds")
    lift_t = st.slider("Lift threshold (interaction)", 0.05, 1.0, LIFT_THRESHOLD, 0.05)
    min_share = st.slider("Min cell share to report", 0.0, 0.10, MIN_CELL_SHARE, 0.005)
    compare = st.checkbox("Compare against prior period", value=True)

if src == "Upload CSV":
    up = st.file_uploader("Pivot CSV (first column = row labels)", type=["csv"])
    if up is None:
        st.info("Upload a pivoted CSV, or switch back to the bundled sample.")
        st.stop()
    cur = pd.read_csv(up, index_col=0)
    cur = cur.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    prev = None
    st.caption(f"Uploaded: {cur.shape[0]} rows x {cur.shape[1]} columns")
else:
    cur, prev = cur_default, prev_default
    st.caption(
        "Bundled sample: revenue by region x product, two quarters. The row and column "
        "totals are multiplicative by construction - except for one planted interaction "
        "and one brand-new cell."
    )

if not compare:
    prev = None

n = narrate(cur, metric=metric, unit=unit, previous=prev,
            row_label=cur.index.name or "row", col_label=cur.columns.name or "column")

st.subheader(n.headline)

left, right = st.columns([1.1, 1])
with left:
    st.markdown("**The pivot**")
    st.dataframe(
        cur.style.format("{:,.0f}").background_gradient(cmap="Blues"),
        width="stretch",
    )
with right:
    st.markdown("**The narration**")
    for p in n.paragraphs:
        st.write(p)
    st.markdown("**At a glance**")
    for b in n.bullets:
        st.markdown(f"- {b}")

tab_lift, tab_notable, tab_movers, tab_margins = st.tabs(
    ["🎯 Interaction (lift)", "📌 Notable cells", "📈 Movers", "➕ Margins"]
)

with tab_lift:
    st.write(
        "**Expected** is `row_total x col_total / grand_total` - what each cell would be if "
        "the two dimensions were independent (the same expectation a chi-square test uses). "
        "**Lift** is how far the actual value deviates from that. This is the interaction, and "
        "it is the one thing eyeballing a grid reliably misses, because the margins are "
        "large and the deviation is not."
    )
    lift = lift_matrix(cur)
    st.dataframe(
        lift.style.format("{:+.1%}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
        width="stretch",
    )
    st.caption(
        "Red is above expectation, blue below. A row of uniform colour means that row is "
        "simply big or small - no interaction. A single hot cell in an otherwise flat row "
        "is the finding."
    )

with tab_notable:
    nc = notable_cells(cur, lift_threshold=float(lift_t), min_share=float(min_share), top=10)
    if len(nc):
        st.dataframe(nc, width="stretch", hide_index=True)
    else:
        st.success(
            f"No cell deviates more than {lift_t:.0%} from expectation while holding at "
            f"least {min_share:.1%} of the total - the margins tell the whole story."
        )
    st.info(
        f"**Two guards, both necessary.** The lift threshold ({lift_t:.0%}) finds the "
        f"interaction. The share floor ({min_share:.1%}) stops a tiny cell with a 500% lift "
        "from being reported as the headline. Drop the share floor to 0 and watch this table "
        "fill with statistically loud, practically irrelevant cells."
    )

with tab_movers:
    movers = n.facts["movers"]
    if prev is None:
        st.info("Enable the prior-period comparison in the sidebar.")
    elif len(movers):
        st.dataframe(movers, width="stretch", hide_index=True)
        st.caption(
            "For a cell that did not exist before, `pct_change` is written as `None` and shows "
            "as null here - never `inf`, which is what a naive pct_change would put on a slide. "
            "Growth from zero is undefined, so the explicit `is_new` flag drives the sentence "
            "and the narration says 'new'. The pivots are also reindexed to the union of both "
            "periods, so an appearing or disappearing segment is reported, not silently dropped."
        )
    else:
        st.success("No cell changed between periods.")

with tab_margins:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{cur.index.name or 'Row'} totals**")
        rt = n.facts["row_totals"]
        st.dataframe(
            rt.rename(metric).to_frame().assign(share=(rt / rt.sum()).round(4)),
            width="stretch",
        )
        st.bar_chart(rt)
    with c2:
        st.markdown(f"**{cur.columns.name or 'Column'} totals**")
        ct = n.facts["col_totals"]
        st.dataframe(
            ct.rename(metric).to_frame().assign(share=(ct / ct.sum()).round(4)),
            width="stretch",
        )
        st.bar_chart(ct)
    rc = n.facts["row_concentration"]
    if rc["top1_share"] >= CONCENTRATION_ALERT:
        st.warning(
            f"{rc['top1']} is {rc['top1_share']:.0%} of {metric} on its own - the mean "
            "across rows describes almost nothing here."
        )
    st.markdown("**Expected values** (independence model)")
    st.dataframe(expected_matrix(cur).style.format("{:,.0f}"), width="stretch")

st.divider()
st.caption(
    "Day 130 of Phoebe's daily FDE build - Analytics Engineering & BI line. "
    "Pairs with Day 107 (KPI tree), Day 121 (metric diff), Day 120 (crosstab & chi-square)."
)
