"""Survey Results Analyzer — Streamlit UI.

Upload a survey CSV (or use the built-in sample) and get an instant,
type-aware breakdown: NPS gauge, Likert/numeric summaries, choice
distributions, and open-text sentiment + themes.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from survey import (
    LIKERT,
    NPS,
    NUMERIC,
    OPEN_TEXT,
    SINGLE_CHOICE,
    analyze_survey,
    sample_survey,
)

st.set_page_config(page_title="Survey Results Analyzer", page_icon="📊", layout="wide")

st.title("📊 Survey Results Analyzer")
st.caption(
    "Upload a survey export and get a type-aware summary in seconds — "
    "NPS, Likert scores, choice breakdowns, and open-text sentiment."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Survey CSV", type=["csv"])
    use_sample = st.button("Use sample survey")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = sample_survey()
else:
    st.info("Upload a CSV in the sidebar, or click **Use sample survey** to try it.")
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)

report = analyze_survey(df)
st.success(
    f"Analyzed {report.n_respondents} respondents "
    f"across {len(df.columns)} questions."
)

# --- NPS headline ----------------------------------------------------------
if report.nps is not None and report.nps_breakdown is not None:
    st.subheader("Net Promoter Score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NPS", f"{report.nps:.0f}")
    c2.metric("Promoters", report.nps_breakdown["promoters"])
    c3.metric("Passives", report.nps_breakdown["passives"])
    c4.metric("Detractors", report.nps_breakdown["detractors"])

# --- Per-question breakdown ------------------------------------------------
st.subheader("Question breakdown")

for q in report.questions:
    with st.expander(f"{q.column}  —  `{q.qtype}`  (n={q.n})", expanded=False):
        if q.qtype in (LIKERT, SINGLE_CHOICE):
            dist = q.summary.get("distribution", {})
            if dist:
                dist_series = pd.Series(dist).sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, 0.4 * len(dist_series) + 1))
                ax.barh(
                    dist_series.index.astype(str),
                    dist_series.values,
                    color="#4F46E5",
                )
                ax.set_xlabel("Responses")
                st.pyplot(fig)
                plt.close(fig)
            if q.summary.get("mean_score") is not None:
                st.metric("Mean Likert score (1-5)", q.summary["mean_score"])

        elif q.qtype in (NUMERIC, NPS):
            cols = st.columns(len(q.summary))
            for col, (k, v) in zip(cols, q.summary.items()):
                col.metric(k, v)

        elif q.qtype == OPEN_TEXT:
            pct = q.summary.get("sentiment_pct", {})
            if pct:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Positive", f"{pct.get('positive', 0)}%")
                sc2.metric("Neutral", f"{pct.get('neutral', 0)}%")
                sc3.metric("Negative", f"{pct.get('negative', 0)}%")
            themes = q.summary.get("themes", [])
            if themes:
                theme_str = ", ".join(f"`{w}` ({c})" for w, c in themes)
                st.markdown("**Top themes:** " + theme_str)
            sample = q.summary.get("sample", [])
            if sample:
                st.markdown("**Sample responses:**")
                for s in sample:
                    st.markdown(f"> {s}")

# --- Download summary ------------------------------------------------------
rows = [
    {"question": q.column, "type": q.qtype, "n": q.n, "summary": str(q.summary)}
    for q in report.questions
]
summary_df = pd.DataFrame(rows)
buf = io.StringIO()
summary_df.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download summary CSV",
    buf.getvalue(),
    file_name="survey_summary.csv",
    mime="text/csv",
)
