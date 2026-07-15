from __future__ import annotations

"""Streamlit UI for the Train/Eval Leaderboard.

Upload a CSV (or use the built-in sample), pick the target column, and get a
cross-validated leaderboard plus a per-fold spread chart - so model choice is a
ranked, reproducible decision instead of one lucky notebook split.
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from harness import (
    DEFAULT_METRICS,
    format_leaderboard,
    make_sample_data,
    run_leaderboard,
)

st.set_page_config(page_title="Train/Eval Leaderboard", page_icon="🏁", layout="wide")

st.title("🏁 Train/Eval Leaderboard")
st.caption(
    "Cross-validate a roster of models on one dataset and rank them honestly - "
    "mean ± std across folds, with a majority-class baseline for a sanity floor."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    n_splits = st.slider("CV folds", 2, 10, 5)
    st.divider()
    st.markdown("No file? The app loads a synthetic churn dataset so you can try it now.")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
else:
    df = make_sample_data()
    source = "sample churn data"

st.subheader(f"Dataset — {source}")
st.write(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# Default the target to the last column - the common convention.
target = st.selectbox("Target column (binary)", df.columns, index=len(df.columns) - 1)

if st.button("Run leaderboard", type="primary"):
    try:
        with st.spinner("Cross-validating models..."):
            board, folds = run_leaderboard(df, target=target, n_splits=n_splits)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Leaderboard")
    st.dataframe(format_leaderboard(board), use_container_width=True, hide_index=True)

    winner = board.iloc[0]["model"]
    top_metric = list(DEFAULT_METRICS.values())[0]
    st.success(
        f"🥇 **{winner}** leads on {top_metric} "
        f"({board.iloc[0][f'{top_metric} mean']:.3f})."
    )

    st.subheader("Per-fold spread")
    st.caption(
        "Wide boxes = unstable across folds. A high mean with a wide spread is "
        "riskier than a slightly lower but tight one."
    )
    metric_choice = st.selectbox("Metric to plot", list(DEFAULT_METRICS.values()))
    sub = folds[folds["metric"] == metric_choice]
    order = board["model"].tolist()
    data = [sub[sub["model"] == m]["score"].values for m in order]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=False, tick_labels=order)
    ax.set_xlabel(metric_choice)
    ax.set_title(f"{metric_choice} across {n_splits} folds")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

    csv = board.to_csv(index=False)
    st.download_button(
        "Download leaderboard (CSV)",
        data=io.BytesIO(csv.encode()),
        file_name="leaderboard.csv",
        mime="text/csv",
    )
