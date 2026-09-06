from __future__ import annotations

# Streamlit front end for the source-to-target reconciliation checker.
# Upload the SOURCE csv and the TARGET csv (or use the built-in sample),
# pick the key column, and get a pass/fail verdict with the missing / extra /
# mismatch tables laid out for a steward to act on. Logic lives in
# reconciler.py so the notebook, CLI, and app all agree on the same rules.
import pandas as pd
import streamlit as st
from reconciler import (
    DEFAULT_TOLERANCE,
    make_sample_data,
    mismatches_frame,
    reconcile,
    summarize,
)

st.set_page_config(page_title="Source-to-Target Reconciliation", layout="wide")

st.title("Source-to-Target Reconciliation")
st.caption(
    "Prove a table copied from a source system into the warehouse matched - "
    "row counts, missing / extra keys, and silent value drift, at one instant."
)


def _load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)


with st.sidebar:
    st.header("1. Data")
    use_sample = st.checkbox("Use built-in sample (with planted drift)", value=True)

    source_df = None
    target_df = None
    if use_sample:
        data = make_sample_data()
        source_df, target_df = data["source"], data["target"]
        st.success("Loaded sample: 50 source rows vs 48 target rows.")
    else:
        src_up = st.file_uploader("Source CSV (system of record)", type="csv")
        tgt_up = st.file_uploader("Target CSV (the copy)", type="csv")
        if src_up is not None:
            source_df = _load_csv(src_up)
        if tgt_up is not None:
            target_df = _load_csv(tgt_up)

    st.header("2. Settings")
    tolerance = st.number_input(
        "Numeric tolerance (absolute)",
        min_value=0.0, value=float(DEFAULT_TOLERANCE), format="%.8f",
        help="Numeric cells within this absolute difference count as a match.",
    )

if source_df is None or target_df is None:
    st.info("Upload both a source and a target CSV, or tick the sample box in the sidebar.")
    st.stop()

# Key must exist in BOTH frames - offer only the columns they share.
common = [c for c in source_df.columns if c in target_df.columns]
if not common:
    st.error("Source and target share no columns - cannot pick a key.")
    st.stop()

key = st.sidebar.selectbox("3. Key column", common,
                           index=0, help="The column that uniquely identifies a row in both systems.")

run = st.sidebar.button("Run reconciliation", type="primary")

col_s, col_t = st.columns(2)
with col_s:
    st.subheader("Source (preview)")
    st.dataframe(source_df.head(20), use_container_width=True)
with col_t:
    st.subheader("Target (preview)")
    st.dataframe(target_df.head(20), use_container_width=True)

if not run:
    st.stop()

result = reconcile(source_df, target_df, key=key, tolerance=tolerance)

if not result.ok:
    st.error(f"Reconciliation could not run: {result.error}")
    st.stop()

# --- verdict banner ---
if result.passed:
    st.success(f"PASS - the copy reconciles. Match rate {result.match_rate:.2%}.")
else:
    st.error(f"FAIL - the copy does not fully reconcile. Match rate {result.match_rate:.2%}.")

# --- headline metrics ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Row delta", result.row_delta, help="target rows minus source rows")
m2.metric("Missing in target", len(result.missing_keys))
m3.metric("Extra in target", len(result.extra_keys))
m4.metric("Cell mismatches", len(result.cell_mismatches))
m5.metric("Match rate", f"{result.match_rate:.1%}")

st.subheader("Summary")
st.dataframe(summarize(result), use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Keys missing in target")
    st.caption("Rows in the source that never landed in the target.")
    if result.missing_keys:
        st.dataframe(pd.DataFrame({key: result.missing_keys}),
                     use_container_width=True, hide_index=True)
    else:
        st.success("None - every source key is present in the target.")
with c2:
    st.subheader("Keys extra in target")
    st.caption("Rows in the target with no matching source key.")
    if result.extra_keys:
        st.dataframe(pd.DataFrame({key: result.extra_keys}),
                     use_container_width=True, hide_index=True)
    else:
        st.success("None - the target has no stray rows.")

st.subheader("Cell-level value mismatches")
st.caption("Shared keys where a column's value differs between the two systems.")
mm = mismatches_frame(result)
if not mm.empty:
    st.dataframe(mm, use_container_width=True, hide_index=True)
else:
    st.success("None - every shared key matched on every compared column.")

st.subheader("Aggregate checks")
st.caption("Independent scale-level cross-check: a numeric sum must agree within tolerance.")
if result.agg_checks:
    agg_rows = [
        {"column": a.column, "agg": a.agg, "source": a.source_value,
         "target": a.target_value, "delta": a.delta, "tolerance": a.tolerance,
         "result": "PASS" if a.passed else "FAIL"}
        for a in result.agg_checks
    ]
    st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)
else:
    st.info("No numeric columns shared between source and target to aggregate.")

st.caption(
    "A clean reconciliation is evidence, not proof - tolerances are a judgment "
    "call, and a match on these checks does not certify every value is correct."
)
