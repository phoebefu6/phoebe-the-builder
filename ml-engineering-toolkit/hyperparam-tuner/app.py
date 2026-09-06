from __future__ import annotations

"""Streamlit UI for the Hyperparameter Tuner.

Upload a CSV (or use the sample), pick a binary target and a trial budget, and
watch an Optuna TPE search beat the model's defaults - with the optimization
history and the winning params laid out, not hand-edited in a cell.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from tuner import make_sample_data, summary_frame, tune

st.set_page_config(page_title="Hyperparameter Tuner", page_icon="🎛️", layout="wide")

st.title("🎛️ Hyperparameter Tuner")
st.caption(
    "Optuna TPE search over a RandomForest, cross-validated on ROC AUC - "
    "a budgeted, reproducible search that reports its lift over the defaults."
)

with st.sidebar:
    st.header("Search")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    n_trials = st.slider("Trial budget", 10, 100, 40, step=10)
    st.divider()
    st.markdown("No file? A synthetic churn dataset loads so you can try it now.")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
else:
    df = make_sample_data()
    source = "sample churn data"

st.subheader(f"Dataset — {source}")
st.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

target = st.selectbox("Target column (binary)", df.columns, index=len(df.columns) - 1)

if st.button("Tune", type="primary"):
    try:
        with st.spinner(f"Running {n_trials} Optuna trials..."):
            result = tune(df, target=target, n_trials=n_trials)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Default ROC AUC", f"{result['default_score']:.4f}")
    c2.metric("Tuned ROC AUC", f"{result['best_score']:.4f}", f"+{result['lift']:.4f}")
    c3.metric("Trials", result["n_trials"])

    st.subheader("Default vs. tuned")
    st.dataframe(summary_frame(result), use_container_width=True, hide_index=True)

    st.subheader("Best hyperparameters")
    st.json(result["best_params"])

    st.subheader("Optimization history")
    st.caption("Each dot is a trial; the line is the best score found so far.")
    hist = result["history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(hist["trial"], hist["value"], s=22, alpha=0.5, label="trial")
    ax.plot(hist["trial"], hist["best_so_far"], color="#C44E52", lw=2, label="best so far")
    ax.axhline(result["default_score"], color="#555", ls="--", lw=1, label="defaults")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV ROC AUC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
