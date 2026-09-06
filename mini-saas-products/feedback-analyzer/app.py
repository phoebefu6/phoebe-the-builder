from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from feedback_analyzer import SAMPLE_REVIEWS, analyze_feedback

st.set_page_config(page_title="User Feedback Analyzer", page_icon="🗣️")
st.title("🗣️ User Feedback Analyzer")
st.caption(
    "10K reviews and no insights? Paste them in and get sentiment split, top complaints, "
    "top praises, and the one thing to fix first."
)

with st.sidebar:
    st.subheader("Settings")
    use_claude = st.checkbox("Claude summary (needs ANTHROPIC_API_KEY)", value=False)
    st.info("Paste one review per line, or use the sample set.")

default = "\n".join(SAMPLE_REVIEWS)
raw = st.text_area("Reviews (one per line)", default, height=220)

if st.button("Analyze", type="primary"):
    reviews = [ln for ln in raw.splitlines() if ln.strip()]
    report = analyze_feedback(reviews, use_claude=use_claude)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", len(report.scores))
    c2.metric("Positive", report.distribution["positive"])
    c3.metric("Negative", report.distribution["negative"])
    c4.metric("NPS-like", f"{report.nps_like:+.0f}")

    st.info(f"**Insight:** {report.insight}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Sentiment split")
        dist = report.distribution
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar(dist.keys(), dist.values(), color=["#2e9e5b", "#b0b0b0", "#d64545"])
        ax.set_ylabel("Reviews")
        plt.tight_layout()
        st.pyplot(fig)
    with col_r:
        st.subheader("🔴 Top complaints")
        st.dataframe(pd.DataFrame(report.top_complaints, columns=["theme", "count"]),
                     hide_index=True, use_container_width=True)
        st.subheader("🟢 Top praises")
        st.dataframe(pd.DataFrame(report.top_praises, columns=["theme", "count"]),
                     hide_index=True, use_container_width=True)

    st.subheader("Per-review sentiment")
    st.dataframe(
        pd.DataFrame([{"label": s.label, "score": s.score, "review": s.text} for s in report.scores]),
        hide_index=True, use_container_width=True,
    )
