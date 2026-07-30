from __future__ import annotations

# Streamlit UI for the Null Heatmap. Shows what df.isna().sum() cannot: whether
# nulls are independent or correlated, which columns fail together, and who
# dropna() actually deletes. Fully offline.
import pandas as pd
import streamlit as st
from missingness import (
    SEGMENT_SKEW_THRESHOLD,
    co_missing_matrix,
    completeness_by_segment,
    row_patterns,
    sample_frame,
    summary,
)

st.set_page_config(page_title="Null Heatmap", page_icon="🕳️", layout="wide")

st.title("🕳️ Null Heatmap")
st.caption(
    "`df.isna().sum()` gives you a per-column count and hides the only thing that "
    "matters: whether the nulls are independent or correlated. 8% missing at random "
    "is a nuisance you impute. 8% missing on the same rows is a broken join - and "
    "`dropna()` will silently delete that entire population."
)

with st.sidebar:
    st.header("Data")
    n_rows = st.slider("Sample rows", 200, 5000, 800, 100)
    st.header("Segment")
    segment = st.selectbox("Bias check against", ["channel", "country", "(none)"], index=0)
    seg = None if segment == "(none)" else segment
    st.header("Upload")
    up = st.file_uploader("Or use your own CSV", type=["csv"])

if up is not None:
    df = pd.read_csv(up)
    st.caption(f"Using uploaded CSV: {len(df)} rows x {len(df.columns)} columns")
    if seg and seg not in df.columns:
        obj_cols = [c for c in df.columns if df[c].dtype == object and df[c].nunique() <= 20]
        seg = obj_cols[0] if obj_cols else None
        st.caption(f"Segment column auto-picked: {seg or 'none suitable found'}")
else:
    df = sample_frame(int(n_rows))
    st.caption(
        f"Bundled sample: {len(df)} customers. Three different mechanisms hide behind "
        "similar null rates - a genuinely optional field, one failed payment join, and "
        "an activity source that barely covers the `partner` channel."
    )

s = summary(df, segment=seg)
reports = s["reports"]
d = s["dropna"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{s['rows']:,}")
c2.metric("Cell completeness", f"{s['cell_completeness']:.2%}")
c3.metric("Columns with nulls", s["columns_with_nulls"])
c4.metric("dropna() deletes", f"{d['share_dropped']:.1%}", f"{d['rows_dropped']:,} rows")
skewed = [r for r in reports if r.segment_note]
c5.metric("Segment-skewed cols", len(skewed))

if d.get("bias_warning"):
    st.error(f"**Bias:** {d['bias_warning']}")
elif seg:
    st.success(f"dropna() removes rows roughly evenly across `{seg}` - no population loss.")

tab_cols, tab_co, tab_rows_, tab_drop, tab_map = st.tabs(
    ["📊 Columns & mechanism", "🔗 Co-missingness", "🧩 Row patterns",
     "✂️ dropna() cost", "🗺️ Null map"]
)

with tab_cols:
    rep = pd.DataFrame([{
        "column": r.column, "n_missing": r.n_missing, "completeness": r.completeness,
        "mechanism": r.mechanism, "partner": r.partner or "-",
        "jaccard": r.partner_jaccard, "segment_spread": r.segment_spread,
    } for r in reports])
    st.dataframe(rep, width="stretch", hide_index=True)
    st.bar_chart(rep.set_index("column")["completeness"])
    st.markdown(
        f"""
**Two independent axes, and the table shows both.**

`jaccard` is column-pair lockstep - how much two columns' nulls land on the *same rows*.
1.00 means one root cause, not two. `segment_spread` is the completeness gap across
`{seg or 'the segment'}` - above **{SEGMENT_SKEW_THRESHOLD:.0%}** the nulls are not
population-neutral.

A column can score 1.00 on the first and ~0 on the second (a failed join that hit
everyone equally) or high on both (a source that only covers part of your customers).
The first is an engineering bug. The second is a governance problem.
        """
    )
    for r in skewed:
        st.warning(f"**{r.column}** — {r.segment_note}")

with tab_co:
    nulled = [c for c in df.columns if df[c].isna().any()]
    if len(nulled) < 2:
        st.info("Fewer than two columns have nulls - nothing to correlate.")
    else:
        jac = co_missing_matrix(df).loc[nulled, nulled]
        st.dataframe(
            jac.style.background_gradient(cmap="Reds", vmin=0, vmax=1).format("{:.2f}"),
            width="stretch",
        )
        st.caption(
            "Jaccard overlap of null positions, not phi correlation. Both agree on perfect "
            "lockstep (1.0), but phi is base-rate dependent: two columns sharing half their "
            "nulls score phi=0.49 at a 2% null rate and phi=0.29 at a 30% rate - same "
            "overlap, different numbers. Every pair here has a different base rate, which is "
            "exactly when you need to compare them. Jaccard returns 0.33 for both: of the "
            "rows missing in either column, what share are missing in both?"
        )
        pairs = [
            (a, b, jac.loc[a, b])
            for i, a in enumerate(nulled) for b in nulled[i + 1:]
            if jac.loc[a, b] >= 0.9
        ]
        if pairs:
            st.error(
                "**Lockstep pairs (one root cause):** "
                + "; ".join(f"`{a}` + `{b}` ({v:.0%})" for a, b, v in pairs)
            )

with tab_rows_:
    st.write(
        "Distinct null **signatures** across rows. A handful of repeated shapes means a "
        "systematic cause; thousands of unique shapes means random noise."
    )
    st.dataframe(row_patterns(df, top=12), width="stretch", hide_index=True)

with tab_drop:
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows in", f"{d['rows_total']:,}")
    k2.metric("Rows surviving dropna()", f"{d['rows_kept']:,}")
    k3.metric("Deleted", f"{d['rows_dropped']:,}", f"{d['share_dropped']:.1%}")
    if "segment_impact" in d:
        si = d["segment_impact"]
        st.dataframe(si, width="stretch", hide_index=True)
        st.bar_chart(si.set_index("segment")["retained"])
        st.caption(
            "`retained` is the share of each segment's rows that survive. When these differ, "
            "`dropna()` is not cleaning - it is sampling, and `share_shift_pp` is how much "
            "the population composition moved."
        )
    if seg:
        st.write(f"**Per-column completeness by `{seg}`** - where the gap actually lives:")
        st.dataframe(completeness_by_segment(df, seg), width="stretch", hide_index=True)

with tab_map:
    st.write(
        "Null positions for a sample of rows - black is missing. Vertical stripes that "
        "line up across two columns are the lockstep pattern the Jaccard matrix scores."
    )
    show_n = min(200, len(df))
    mask = df.isna().head(show_n).astype(int)
    st.dataframe(
        mask.T.style.background_gradient(cmap="Greys", vmin=0, vmax=1).format(""),
        width="stretch",
    )
    st.caption(f"First {show_n} rows, columns as rows of the grid.")

st.divider()
st.caption(
    "Day 129 of Phoebe's daily FDE build - Data Quality & Governance line. "
    "Pairs with Day 91 (anomaly detector), Day 92 (DQ rules engine), Day 28 (DQ scorecard)."
)
