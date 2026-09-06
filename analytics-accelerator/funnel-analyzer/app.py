"""Funnel Analyzer — Streamlit UI.

Upload an event log (or use the built-in sample), define the funnel steps, and
see exactly where users drop off — an interactive Plotly funnel, the biggest
leak called out, a step-by-step table, and an optional segment comparison.
"""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from funnel import (
    DEFAULT_STEPS,
    compute_funnel,
    compute_funnel_by_segment,
    sample_events,
    steps_to_frame,
)

st.set_page_config(page_title="Funnel Analyzer", page_icon="🪜", layout="wide")

st.title("🪜 Funnel Analyzer")
st.caption(
    "See exactly where users drop off. Define your steps, and get an interactive "
    "funnel, the biggest leak called out, and a segment comparison."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Event log CSV (user + event columns)", type=["csv"])
    use_sample = st.button("Use sample event log")

if uploaded is not None:
    events = pd.read_csv(uploaded)
elif use_sample:
    events = sample_events()
else:
    st.info(
        "Upload an event log (one row per user-event), or click **Use sample "
        "event log** to try a 5-step e-commerce funnel."
    )
    st.stop()

st.subheader("Raw events")
st.dataframe(events.head(20), use_container_width=True)

cols = list(events.columns)
with st.sidebar:
    st.header("Funnel")
    user_col = st.selectbox(
        "User column", cols, index=cols.index("user_id") if "user_id" in cols else 0
    )
    event_col = st.selectbox(
        "Event column", cols, index=cols.index("event") if "event" in cols else 0
    )
    all_events = sorted(events[event_col].dropna().unique().tolist())
    default_steps = [s for s in DEFAULT_STEPS if s in all_events] or all_events
    steps = st.multiselect("Funnel steps (in order)", all_events, default=default_steps)
    seg_options = ["(none)"] + [c for c in cols if c not in (user_col, event_col)]
    segment_col = st.selectbox("Compare by segment", seg_options, index=0)

if len(steps) < 2:
    st.warning("Pick at least 2 funnel steps.")
    st.stop()

if segment_col != "(none)":
    result = compute_funnel_by_segment(events, steps, segment_col, user_col, event_col)
else:
    result = compute_funnel(events, steps, user_col, event_col)

# --- Headline --------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Users entering", f"{result.steps[0].users:,}")
c2.metric("Overall conversion", f"{result.overall_conversion:.1f}%")
c3.metric(
    "Biggest leak",
    result.biggest_drop or "—",
    delta=f"-{result.biggest_drop_pct:.1f}%" if result.biggest_drop else None,
    delta_color="inverse",
)

if result.biggest_drop:
    st.warning(
        f"🚨 Largest drop-off is at **{result.biggest_drop}** — "
        f"{result.biggest_drop_pct:.1f}% of users are lost here. Fix this step first."
    )

# --- Plotly funnel ---------------------------------------------------------
st.subheader("Conversion funnel")
labels = [s.step for s in result.steps]
values = [s.users for s in result.steps]
fig = go.Figure(
    go.Funnel(
        y=labels,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker={"color": "#4F46E5"},
        connector={"line": {"color": "#C7D2FE"}},
    )
)
fig.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=380)
st.plotly_chart(fig, use_container_width=True)

# --- Step table ------------------------------------------------------------
st.subheader("Step-by-step breakdown")
table = steps_to_frame(result)
st.dataframe(table, use_container_width=True)

# --- Segment comparison ----------------------------------------------------
if result.by_segment:
    st.subheader(f"Overall conversion by {result.segment_col}")
    seg_df = (
        pd.DataFrame(
            {"segment": list(result.by_segment.keys()),
             "overall conversion %": list(result.by_segment.values())}
        )
    )
    bar = go.Figure(
        go.Bar(
            x=seg_df["overall conversion %"],
            y=seg_df["segment"],
            orientation="h",
            marker={"color": "#16A34A"},
            text=seg_df["overall conversion %"],
            textposition="auto",
        )
    )
    bar.update_layout(margin={"l": 10, "r": 10, "t": 10, "b": 10}, height=300)
    st.plotly_chart(bar, use_container_width=True)

buf = io.StringIO()
table.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download funnel report CSV",
    buf.getvalue(),
    file_name="funnel_report.csv",
    mime="text/csv",
)
