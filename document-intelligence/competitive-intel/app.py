from __future__ import annotations

import pandas as pd
import streamlit as st
from intel import SAMPLE_COMPETITORS, summarize_competitors

st.set_page_config(page_title="Competitive Intel Summarizer", layout="wide")
st.title("Competitive Intel Summarizer")
st.caption("Turn scattered competitor notes into one comparison matrix + strategic takeaways.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using deterministic extraction (regex feature/pricing scan).")
    st.caption("Paste each competitor's site copy or your notes below.")

st.subheader("Competitors")
competitors: dict = {}
default = list(SAMPLE_COMPETITORS.items())
n = st.number_input("How many competitors?", 1, 8, len(default))

for i in range(int(n)):
    d_name, d_text = default[i] if i < len(default) else (f"Competitor {i + 1}", "")
    c1, c2 = st.columns([1, 3])
    name = c1.text_input(f"Name {i + 1}", value=d_name, key=f"n{i}")
    text = c2.text_area(f"Notes {i + 1}", value=d_text, key=f"t{i}", height=90)
    if name.strip() and text.strip():
        competitors[name] = text

if st.button("Analyze", type="primary"):
    if not competitors:
        st.warning("Add at least one competitor with notes.")
        st.stop()

    with st.spinner("Analyzing..."):
        try:
            report = summarize_competitors(competitors, api_key=api_key or None)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    st.subheader("Strategic takeaways")
    for t in report.takeaways:
        st.markdown(f"- {t}")

    st.subheader("Feature matrix")
    if report.feature_matrix:
        df = pd.DataFrame(report.feature_matrix).T
        df = df.replace({True: "✅", False: "—"})
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No comparable features detected.")

    st.subheader("Profiles")
    for p in report.competitors:
        with st.expander(f"{p.name} — {p.target_market or 'market n/a'}"):
            st.markdown(f"**Positioning:** {p.positioning}")
            st.markdown(f"**Pricing:** {', '.join(p.pricing) or 'not disclosed'}")
            st.markdown(f"**Features:** {', '.join(p.features) or 'none detected'}")
