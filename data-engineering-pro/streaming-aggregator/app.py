from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streaming import WindowAggregator, make_event_stream

st.set_page_config(page_title="Streaming Window Aggregator", page_icon="🌊", layout="wide")
st.title("🌊 Streaming Window Aggregator")
st.caption("Tumbling/sliding event-time windows with watermarks — rolling metrics from an out-of-order stream.")

with st.sidebar:
    st.header("Stream")
    n_events = st.slider("Events", 500, 10_000, 2000, step=500)
    duration = st.slider("Stream duration (s)", 120, 1800, 600, step=60)
    late_frac = st.slider("Very-late event fraction", 0.0, 0.10, 0.02, 0.01)
    st.header("Window")
    window_s = st.slider("Window size (s)", 10, 300, 60, step=10)
    mode = st.radio("Mode", ["tumbling", "sliding"])
    slide_s = st.slider("Slide (s)", 5, window_s, max(5, window_s // 3), step=5) if mode == "sliding" else None
    lateness = st.slider("Allowed lateness (s)", 0, 120, 10, step=5)

if st.button("Run stream", type="primary"):
    stream = make_event_stream(n_events, duration_s=duration, late_fraction=late_frac)
    agg = WindowAggregator(window_s, slide_s, allowed_lateness_seconds=lateness)
    results = []
    for e in stream:
        results.extend(agg.feed(e))
    results.extend(agg.flush())

    df = pd.DataFrame([{"window_start": w.window_start, "window_end": w.window_end,
                        "key": w.key, "count": w.count, "mean": w.mean,
                        "min": w.vmin, "max": w.vmax} for w in results])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", f"{len(stream):,}")
    c2.metric("Windows finalized", len(results))
    c3.metric("Late dropped", agg.late_dropped)
    c4.metric("Accounted", f"{sum(w.count for w in results) + agg.late_dropped:,}")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for key, g in df.groupby("key"):
            g = g.sort_values("window_start")
            ax.plot(g["window_start"], g["mean"], marker="o", markersize=3, label=key)
        ax.set_xlabel("Window start (s)")
        ax.set_ylabel("Mean value")
        ax.set_title(f"{mode} windows: rolling mean per key")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
    with col2:
        st.dataframe(df.sort_values(["window_start", "key"]), use_container_width=True,
                     hide_index=True, height=300)
else:
    st.info("Configure the stream and window, then click **Run stream**.")
