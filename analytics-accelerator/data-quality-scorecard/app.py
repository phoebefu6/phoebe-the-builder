"""Data Quality Scorecard — Streamlit UI.

Upload any CSV (or use the built-in messy sample) and get a 0-100 data-quality
score with a letter grade, a per-dimension breakdown, a ranked issues list, and
a downloadable check report.
"""
from __future__ import annotations

import io
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quality import checks_to_frame, sample_dirty_data, score_dataframe

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Quality Scorecard", page_icon="🧪", layout="wide")

st.title("🧪 Data Quality Scorecard")
st.caption(
    "Stop guessing how bad your data is. Upload a CSV and get a 0-100 score, a "
    "letter grade, and a ranked list of what to fix first."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("CSV to score", type=["csv"])
    use_sample = st.button("Use messy sample data")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = sample_dirty_data()
else:
    st.info("Upload a CSV, or click **Use messy sample data** to see a scorecard.")
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)

card = score_dataframe(df)

# --- Headline score --------------------------------------------------------
grade_color = {
    "A": "#16A34A", "B": "#65A30D", "C": "#F59E0B", "D": "#EA580C", "F": "#DC2626",
}
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Overall score", f"{card.overall:.1f} / 100")
c2.markdown(
    f"<div style='font-size:64px;font-weight:800;line-height:1;"
    f"color:{grade_color[card.grade]}'>{card.grade}</div>",
    unsafe_allow_html=True,
)
c3.metric("Rows × Columns", f"{card.n_rows:,} × {card.n_cols}")

# --- Dimension breakdown ---------------------------------------------------
st.subheader("Quality by dimension")
dims = list(card.dimension_scores.keys())
scores = [card.dimension_scores[d] for d in dims]
fig, ax = plt.subplots(figsize=(9, 3.5))
colors = ["#16A34A" if s >= 90 else "#F59E0B" if s >= 70 else "#DC2626" for s in scores]
bars = ax.barh(dims[::-1], scores[::-1], color=colors[::-1])
ax.set_xlim(0, 100)
ax.set_xlabel("score (0-100)")
ax.bar_label(bars, fmt="%.0f", padding=3)
st.pyplot(fig)
plt.close(fig)

# --- Issues to fix ---------------------------------------------------------
st.subheader("🔧 Fix these first")
if card.issues:
    for issue in card.issues:
        st.markdown(f"- {issue}")
else:
    st.success("No significant data quality issues found. 🎉")

# --- Full check report -----------------------------------------------------
st.subheader("Full check report")
report = checks_to_frame(card).sort_values("score")
st.dataframe(report, use_container_width=True)

buf = io.StringIO()
report.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download quality report CSV",
    buf.getvalue(),
    file_name="data_quality_report.csv",
    mime="text/csv",
)

st.caption(
    "Dimensions: completeness (missing), uniqueness (duplicates), validity "
    "(type/range/format), consistency (casing/whitespace), timeliness (stale/future "
    "dates), distribution (outliers). Weighted into the overall score."
)
