from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from explorer import ExploreQuery, profile_columns, run_query, sample_dataframe

st.set_page_config(page_title="Self-Serve Data Explorer", layout="wide")
st.title("Self-Serve Data Explorer")
st.caption("Let anyone slice the data themselves — pivot, aggregate, filter — so analysts stop getting pinged for every number.")

with st.sidebar:
    st.subheader("Data")
    uploaded = st.file_uploader("Upload a CSV (or use sample)", type=["csv"])

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
prof = profile_columns(df)
dims, measures = prof["dimensions"], prof["measures"]

st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns · {len(dims)} dimensions · {len(measures)} measures")

col_a, col_b, col_c = st.columns(3)
with col_a:
    rows = st.multiselect("Group by (rows)", dims, default=dims[:1])
    pivot_col = st.selectbox("Pivot (columns, optional)", ["(none)"] + [d for d in dims if d not in rows])
with col_b:
    measure = st.selectbox("Measure", measures or df.columns.tolist())
    agg = st.selectbox("Aggregation", ["sum", "mean", "count", "count_distinct", "min", "max"])
with col_c:
    top_n = st.number_input("Top N (0 = all)", min_value=0, value=10, step=1)
    sort_desc = st.checkbox("Sort descending", value=True)

st.markdown("**Filters**")
filters: dict = {}
fcols = st.columns(min(4, len(dims)) or 1)
for i, d in enumerate(dims[:4]):
    with fcols[i % len(fcols)]:
        opts = sorted(df[d].dropna().unique().tolist())
        chosen = st.multiselect(d, opts, default=[])
        if chosen:
            filters[d] = chosen

query = ExploreQuery(
    rows=rows,
    measure=measure,
    agg=agg,
    columns=None if pivot_col == "(none)" else pivot_col,
    filters=filters,
    top_n=int(top_n) or None,
    sort_desc=sort_desc,
)

try:
    result = run_query(df, query)
except Exception as e:
    st.error(f"Query error: {e}")
    st.stop()

st.subheader("Result")
st.dataframe(result, use_container_width=True)
st.download_button("Download CSV", data=result.to_csv(index=False), file_name="explore.csv", mime="text/csv")

# quick chart when there's a single measure column and dimension rows
measure_col = f"{agg}_{measure}"
if query.columns is None and rows and measure_col in result.columns and len(result) <= 30:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = result[rows[0]].astype(str) if len(rows) == 1 else result[rows].astype(str).agg(" · ".join, axis=1)
    ax.bar(labels, result[measure_col], color="#3b6fd6")
    ax.set_title(f"{agg}({measure}) by {', '.join(rows)}", fontsize=12, weight="bold")
    ax.set_ylabel(measure_col)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

with st.expander("Sample data"):
    st.dataframe(df.head(20), use_container_width=True)
