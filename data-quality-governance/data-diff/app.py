from __future__ import annotations

# Streamlit UI for the Dataset Snapshot Diff. Upload two snapshots of the SAME
# dataset (yesterday vs today) or load the built-in sample, pick the key column,
# and see exactly what changed since last run: added / removed / modified rows,
# schema drift, and change velocity. Every modification names the column and the
# old -> new value. Fully offline, no API keys.

import pandas as pd
import streamlit as st

from differ import (
    diff_snapshots,
    make_sample_snapshots,
    modified_frame,
    summary_frame,
)

st.set_page_config(page_title="Dataset Snapshot Diff", page_icon="🕒", layout="wide")

st.title("🕒 Dataset Snapshot Diff")
st.caption(
    "Between yesterday's snapshot and today's, what actually changed? This diffs "
    "two snapshots of the SAME dataset on a key column and reports the added, "
    "removed, and modified rows - each modification naming the column and the "
    "old -> new value - plus schema drift and change velocity. Temporal change "
    "tracking, not cross-system reconciliation. Rule-based, offline."
)

with st.sidebar:
    st.header("Snapshots")
    before_up = st.file_uploader("BEFORE snapshot (CSV)", type=["csv"], key="before")
    after_up = st.file_uploader("AFTER snapshot (CSV)", type=["csv"], key="after")
    st.markdown("or")
    use_sample = st.button("Load sample products snapshots", use_container_width=True)
    st.divider()
    st.markdown(
        "**What you get**\n\n"
        "- **Added** - keys new in the after snapshot\n"
        "- **Removed** - keys gone from the after snapshot\n"
        "- **Modified** - shared key, >=1 column changed\n"
        "- **Schema drift** - columns added / removed\n"
        "- **Velocity** - % of prior snapshot that moved"
    )

# Decide the data source: uploaded pair takes priority, else sample.
if before_up is not None and after_up is not None:
    before_df = pd.read_csv(before_up)
    after_df = pd.read_csv(after_up)
    source = f"{before_up.name} -> {after_up.name}"
else:
    snaps = make_sample_snapshots()
    before_df, after_df = snaps["before"], snaps["after"]
    source = "sample products snapshots (planted changes)"
    if before_up is not None or after_up is not None:
        st.info("Upload BOTH before and after CSVs to diff your own - showing sample for now.")

st.write(
    f"**Source:** {source} - before {len(before_df):,}x{before_df.shape[1]}, "
    f"after {len(after_df):,}x{after_df.shape[1]}"
)

# Key picker - only columns that exist in BOTH snapshots can be a valid key.
shared = [c for c in before_df.columns if c in after_df.columns]
if not shared:
    st.error("The two snapshots share no columns - cannot pick a key to diff on.")
    st.stop()
key = st.selectbox("Key column (unique row identifier)", shared)

# Run the diff, surfacing the graceful errors from differ.py as UI messages.
try:
    result = diff_snapshots(before_df, after_df, key=key)
except ValueError as e:
    st.error(f"Cannot diff: {e}")
    st.stop()

v = result.velocity
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Added", f"{v['added']:,}")
c2.metric("Removed", f"{v['removed']:,}")
c3.metric("Modified", f"{v['modified']:,}")
c4.metric("Unchanged", f"{v['unchanged']:,}")
c5.metric("% changed", f"{v['pct_changed']}%")

st.subheader("Change summary")
st.dataframe(summary_frame(result), use_container_width=True, hide_index=True)

# Schema drift note - the easiest thing to miss by eyeballing two files.
drift = result.schema_drift
if drift.added_columns or drift.removed_columns:
    parts = []
    if drift.added_columns:
        parts.append(f"columns **added**: {', '.join(drift.added_columns)}")
    if drift.removed_columns:
        parts.append(f"columns **removed**: {', '.join(drift.removed_columns)}")
    st.warning("Schema drift detected - " + "; ".join(parts))
else:
    st.success("No schema drift - the column set is identical across snapshots.")

st.subheader(f"Added rows ({len(result.added)})")
if len(result.added):
    st.dataframe(result.added, use_container_width=True, hide_index=True)
else:
    st.caption("No new keys.")

st.subheader(f"Removed rows ({len(result.removed)})")
if len(result.removed):
    st.dataframe(result.removed, use_container_width=True, hide_index=True)
else:
    st.caption("No removed keys.")

st.subheader(f"Modified rows ({len(result.modified)}) - one line per changed cell")
mf = modified_frame(result)
if not mf.empty:
    st.dataframe(mf, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download modified cells CSV",
        mf.to_csv(index=False).encode(),
        file_name="modified_cells.csv",
        mime="text/csv",
    )
else:
    st.caption("No shared rows changed.")

st.caption(
    "A diff shows WHAT changed, not WHY. Large diffs need a human to judge intent - "
    "treat this as an audit trail, not an approval."
)
