from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from outliers import detect_outliers, explain_text, sample_dataframe, summary

st.set_page_config(page_title="Outlier Explainer", layout="wide")
st.title("Outlier Explainer")
st.caption('"Which rows are weird, and why?" — Isolation Forest finds them, z-scores explain them.')

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (or use sample)", type=["csv"])
    contamination = st.slider("Expected outlier fraction", 0.01, 0.20, 0.05)

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
outliers, raw, flags = detect_outliers(df, contamination=contamination)
summ = summary(outliers, len(df))

c1, c2, c3 = st.columns(3)
c1.metric("Rows", summ["total"])
c2.metric("Outliers", summ["outliers"])
c3.metric("Outlier rate", f"{summ['pct']}%")

st.subheader("Flagged rows — with reasons")
rows = []
for o in outliers:
    rec = {"row": o.index, "anomaly_score": o.score, "why": explain_text(o, df)}
    for col in df.select_dtypes("number").columns:
        rec[col] = df.iloc[o.index][col]
    rows.append(rec)
st.dataframe(pd.DataFrame(rows), use_container_width=True)

if summ["top_drivers"]:
    st.caption("Most common driver features: " + ", ".join(f"{k} ({v})" for k, v in summ["top_drivers"].items()))

st.subheader("Anomaly score distribution")
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.hist(raw[~flags], bins=30, color="#3b6fd6", alpha=0.8, label="normal")
ax.hist(raw[flags], bins=15, color="#c0553b", alpha=0.9, label="outlier")
ax.set_xlabel("Anomaly score (higher = weirder)")
ax.set_ylabel("Rows")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig)

with st.expander("Data preview"):
    st.dataframe(df.head(), use_container_width=True)
