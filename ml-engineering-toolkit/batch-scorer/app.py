from __future__ import annotations

"""Batch Scoring Service - Streamlit UI.

Turns the manual "score new data by hand" ritual into an upload-and-download
workflow: train/refresh a demo model, feed it a CSV (or built-in sample), and
get back predictions plus probabilities with a one-click download.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from scorer import (
    demo_train,
    load_bundle,
    score_frame,
    sample_new_data,
    FEATURE_NAMES,
)

st.set_page_config(page_title="Batch Scoring Service", page_icon="📦", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

st.title("📦 Batch Scoring Service")
st.caption(
    "Scoring new data shouldn't be a manual copy-paste ritual - "
    "load a trained model, drop in a CSV, and get predictions + probabilities back."
)


# --- Step 1: model ---------------------------------------------------------
st.header("1. Model")

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("🔄 Train / refresh demo model", use_container_width=True):
        with st.spinner("Training RandomForest on synthetic churn data..."):
            path = demo_train()
        st.success("Model trained and persisted.")

with col_b:
    if os.path.exists(MODEL_PATH):
        st.info(f"Model bundle ready at `{os.path.basename(MODEL_PATH)}`.")
    else:
        st.warning("No model yet - click **Train / refresh demo model** to create one.")

if not os.path.exists(MODEL_PATH):
    st.stop()

bundle = load_bundle(MODEL_PATH)
st.write("**Expected feature columns:**", ", ".join(FEATURE_NAMES))


# --- Step 2: data ----------------------------------------------------------
st.header("2. Data to score")

source = st.radio(
    "Choose a data source",
    ["Use built-in sample data", "Upload a CSV"],
    horizontal=True,
)

df: Optional[pd.DataFrame] = None
if source == "Upload a CSV":
    uploaded = st.file_uploader("Upload a CSV with the feature columns above", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    n = st.slider("Number of sample rows", min_value=50, max_value=1000, value=200, step=50)
    df = sample_new_data(n=n)

if df is None:
    st.info("Upload a CSV or switch to built-in sample data to continue.")
    st.stop()


# --- Step 3: score ---------------------------------------------------------
st.header("3. Scored results")

try:
    scored = score_frame(bundle, df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

n_rows = len(scored)
n_flagged = int(scored["prediction"].sum()) if n_rows else 0
positive_rate = (n_flagged / n_rows) if n_rows else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Rows scored", f"{n_rows:,}")
m2.metric("Rows flagged", f"{n_flagged:,}")
m3.metric("Positive rate", f"{positive_rate:.1%}")

st.subheader("Scored table")
st.dataframe(scored, use_container_width=True)

st.subheader("Score distribution")
if n_rows:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(scored["score"], bins=25, color="#4C72B0", edgecolor="white")
    ax.axvline(
        float(bundle.get("threshold", 0.5)),
        color="#C44E52",
        linestyle="--",
        label="threshold",
    )
    ax.set_xlabel("Predicted probability of positive class")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.write("No rows to plot.")

# --- Step 4: download ------------------------------------------------------
st.header("4. Download")
csv_bytes = scored.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download scored CSV",
    data=csv_bytes,
    file_name="scored_output.csv",
    mime="text/csv",
    use_container_width=True,
)
