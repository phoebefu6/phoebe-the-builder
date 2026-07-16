"""Class Imbalance Toolkit — Streamlit UI.

Upload a CSV (or use the built-in fraud sample), pick the target column, and
compare rebalancing strategies. Ranks by minority-class recall so the rare
class you actually care about drives the decision, not accuracy.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from imbalance import (
    compare_strategies,
    imbalance_report,
    make_sample_data,
    recommend,
)

st.set_page_config(page_title="Class Imbalance Toolkit", page_icon="⚖️", layout="wide")
st.title("⚖️ Class Imbalance Toolkit")
st.caption(
    "Accuracy lies on skewed data. Compare SMOTE / class-weights / undersampling "
    "and rank by the metric that matters: minority-class recall."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info("No file — using synthetic fraud sample (3% positive).")
        df = make_sample_data()
    target = st.selectbox("Target column", options=list(df.columns), index=len(df.columns) - 1)

rep = imbalance_report(df, target)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{rep['total_rows']:,}")
c2.metric("Minority class", f"{rep['minority_count']:,}")
c3.metric("Minority %", f"{rep['minority_pct']}%")
c4.metric("Imbalance ratio", f"{rep['imbalance_ratio']}:1")

if st.button("Compare strategies", type="primary"):
    with st.spinner("Training 4 strategies on the same split…"):
        board, probs = compare_strategies(df, target=target)

    st.success(recommend(board))
    st.subheader("Leaderboard (sorted by recall)")
    st.dataframe(
        board.style.background_gradient(subset=["recall", "pr_auc"], cmap="Greens"),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    board_sorted = board.sort_values("recall")
    ax.barh(board_sorted.index, board_sorted["recall"], color="#4C72B0", label="Recall")
    ax.barh(
        board_sorted.index,
        board_sorted["precision"],
        color="#DD8452",
        alpha=0.6,
        label="Precision",
    )
    ax.set_xlabel("Score")
    ax.set_title("Minority-class recall vs precision by strategy")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.caption(
        "Note: rebalancing trades precision for recall. If false positives are "
        "cheap (a review queue) that's a win; if they're expensive, tune the "
        "decision threshold on PR-AUC."
    )
