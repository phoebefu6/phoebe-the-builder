from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from cohort import build_retention_matrix, generate_sample_events

st.set_page_config(page_title="Cohort Analysis Tool", layout="wide")
st.title("Cohort Analysis Tool")
st.caption("Retention heatmap by signup cohort, built in minutes instead of 2 analyst-days.")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload event log (CSV with user_id, event_date)", type="csv")
    use_sample = st.checkbox("Use sample data", value=uploaded is None)
    user_col = st.text_input("User ID column", value="user_id")
    date_col = st.text_input("Event date column", value="event_date")

if uploaded is not None and not use_sample:
    events = pd.read_csv(uploaded)
elif use_sample:
    events = generate_sample_events()
else:
    st.info("Upload a CSV or check 'Use sample data'.")
    st.stop()

try:
    retention = build_retention_matrix(events, user_col=user_col, date_col=date_col)
except KeyError:
    st.error(f"Couldn't find columns '{user_col}' / '{date_col}' in the uploaded file.")
    st.stop()

st.subheader("Retention Heatmap (% of cohort active per month since signup)")
fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(retention))))
sns.heatmap(retention, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100, ax=ax)
ax.set_xlabel("Months Since Signup")
ax.set_ylabel("Signup Cohort")
st.pyplot(fig)

st.subheader("Retention Matrix")
st.dataframe(retention, use_container_width=True)

month1 = retention[1].dropna() if 1 in retention.columns else pd.Series(dtype=float)
if not month1.empty:
    st.metric("Avg Month-1 Retention", f"{month1.mean():.1f}%")
