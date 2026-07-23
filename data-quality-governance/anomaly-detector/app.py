from __future__ import annotations

# Streamlit UI for the Column Anomaly Detector. Upload a CSV (or use the built-in
# sample), scan every column for numeric outliers, null spikes, and rare
# categories, then triage findings by severity - each with a plain-English reason.
# Fully offline, no API keys.

import pandas as pd
import streamlit as st

from detector import (
    anomalies_frame,
    make_sample_data,
    scan_dataframe,
    summarize,
)

st.set_page_config(page_title="Column Anomaly Detector", page_icon="🔎", layout="wide")

st.title("🔎 Column Anomaly Detector")
st.caption(
    "Bad values slip into prod - a fat-finger amount, an impossible age, a typo'd "
    "category, a sudden null spike. This scans each column with complementary "
    "methods (z-score, IQR, MAD, null-rate, rare-category) and tells you which "
    "method fired and why, so you can trust the alert. Rule-based, offline."
)

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload a CSV", type=["csv"])
    st.markdown("or")
    use_sample = st.button("Load sample orders data", use_container_width=True)
    st.divider()
    st.markdown(
        "**Methods**\n\n"
        "- **z-score** > 3 (Gaussian outliers)\n"
        "- **IQR** Tukey fences (distribution-free)\n"
        "- **MAD** modified z > 3.5 (robust)\n"
        "- **null-rate** > 20% (column-level)\n"
        "- **rare-category** < 1% of rows"
    )

if up is not None:
    df = pd.read_csv(up)
    source = up.name
elif use_sample or "df" not in st.session_state:
    df = make_sample_data()
    source = "sample orders (planted anomalies)"
else:
    df = st.session_state["df"]
    source = st.session_state.get("source", "current")

st.session_state["df"] = df
st.session_state["source"] = source

st.write(f"**Source:** {source} — {len(df):,} rows × {df.shape[1]} columns")

reports = scan_dataframe(df)
summary = summarize(reports)
flat = anomalies_frame(reports)

total = len(flat)
highs = int((flat["severity"] == "high").sum()) if total else 0
cols_flagged = int((summary["anomalies"] > 0).sum()) if not summary.empty else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total anomalies", f"{total:,}")
c2.metric("High severity", f"{highs:,}")
c3.metric("Columns flagged", f"{cols_flagged} / {df.shape[1]}")

st.subheader("Per-column triage")
st.dataframe(summary, use_container_width=True, hide_index=True)

if not summary.empty:
    st.bar_chart(summary.set_index("column")[["high", "medium", "low"]])

st.subheader("Findings")
if total == 0:
    st.success("No anomalies found under the current thresholds. ✅")
else:
    sev_order = {"high": 0, "medium": 1, "low": 2}
    pick = st.multiselect(
        "Filter by severity", ["high", "medium", "low"], default=["high", "medium"]
    )
    view = flat[flat["severity"].isin(pick)] if pick else flat
    view = view.sort_values("severity", key=lambda s: s.map(sev_order))
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download findings CSV",
        view.to_csv(index=False).encode(),
        file_name="anomalies.csv",
        mime="text/csv",
    )

st.caption(
    "Thresholds live in detector.py (Z_THRESH, IQR_MULT, MAD_THRESH, "
    "NULL_RATE_WARN, RARE_CATEGORY_FRAC) - tune them to your own quality bar."
)
