from __future__ import annotations

"""Streamlit UI for the Data Leakage Detector."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from leakage import demo_clean_data, demo_leaky_data, run_all

SEVERITY_COLORS = {"high": "#e5484d", "medium": "#f5a623", "low": "#8b8d98"}

st.set_page_config(page_title="Data Leakage Detector", page_icon="🕵️", layout="wide")

st.title("🕵️ Data Leakage Detector")
st.caption(
    "Great CV, terrible in prod. When a model scores 0.99 in cross-validation "
    "and collapses in production, the culprit is usually data leakage. "
    "This tool runs target-leak and train/test-leak checks so you catch it early."
)


# ---------------------------------------------------------------------------
# Data selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Choose data")
    source = st.radio(
        "Data source",
        ["Leaky demo", "Clean demo", "Upload CSVs"],
        index=0,
    )

    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    target: Optional[str] = None

    if source == "Leaky demo":
        train_df, test_df, target = demo_leaky_data()
    elif source == "Clean demo":
        train_df, test_df, target = demo_clean_data()
    else:
        train_file = st.file_uploader("Train CSV", type="csv", key="train_csv")
        test_file = st.file_uploader("Test CSV", type="csv", key="test_csv")
        if train_file is not None and test_file is not None:
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            target = st.selectbox("Target column", list(train_df.columns))


# ---------------------------------------------------------------------------
# What each check does
# ---------------------------------------------------------------------------
with st.expander("What does each check look for?"):
    st.info(
        "**Target correlation** - a numeric feature whose absolute correlation "
        "with the target is extreme (>= 0.95). The answer likely leaked into a column.\n\n"
        "**Single-feature AUC** - one feature that on its own scores AUC >= 0.90 "
        "against a binary target. A lone feature that nearly solves the task is a red flag.\n\n"
        "**Duplicate rows** - rows in the test set that also appear (exactly) in "
        "train. The model is validated on rows it was trained on.\n\n"
        "**ID-like columns** - near-unique, monotonic, or id/uuid/index-named "
        "columns that can leak row identity or ordering.\n\n"
        "**Train/test distribution** - a large gap in the positive rate between "
        "train and test, a sign of a bad (non-random) split."
    )


# ---------------------------------------------------------------------------
# Run + render
# ---------------------------------------------------------------------------
if train_df is None or test_df is None or target is None:
    st.warning("Upload both a train and a test CSV and pick a target column to run.")
    st.stop()

with st.expander("Preview train data"):
    st.dataframe(train_df.head(20), use_container_width=True)

result = run_all(train_df, test_df, target)
summary = result["summary"]
findings: List[Dict] = result["findings"]

# Verdict banner
if result["verdict"] == "leaky":
    st.error("### 🚨 LEAKY - this dataset shows signs of leakage. Do not trust the CV score.")
else:
    st.success("### ✅ CLEAN - no high-severity leakage signals found.")

# Metric cards
c1, c2, c3 = st.columns(3)
c1.metric("High severity", summary["n_high"])
c2.metric("Medium severity", summary["n_medium"])
c3.metric("Low severity", summary["n_low"])

# Findings table (color-coded)
st.subheader("Findings")
if findings:
    table = pd.DataFrame(findings)[["severity", "check", "feature", "detail"]]

    def _row_style(row: pd.Series) -> List[str]:
        color = SEVERITY_COLORS.get(row["severity"], "#8b8d98")
        return [f"background-color: {color}22; color: inherit"] * len(row)

    styled = table.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)
else:
    st.write("No findings - nothing tripped the checks.")

# Bar chart of findings by severity
st.subheader("Findings by severity")
order = ["high", "medium", "low"]
counts = [summary["n_high"], summary["n_medium"], summary["n_low"]]
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(order, counts, color=[SEVERITY_COLORS[s] for s in order])
ax.set_ylabel("count")
ax.set_title("Leak findings by severity")
for i, v in enumerate(counts):
    ax.text(i, v + 0.05, str(v), ha="center", va="bottom", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
st.pyplot(fig)

st.caption(
    "Heuristics can false-positive on legitimately strong features. Always "
    "review each finding before dropping a column."
)
