from __future__ import annotations

"""Streamlit UI for the Probability Calibration Checker."""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from calibration import (
    demo_data,
    evaluate,
    fit_models,
    reliability_points,
)

st.set_page_config(page_title="Probability Calibration Checker", page_icon="🎯", layout="wide")

st.title("🎯 Probability Calibration Checker")
st.caption(
    'Pain point: "Our probabilities are meaningless." A model that says 0.9 '
    "should be right ~90% of the time. This tool measures whether yours is, and "
    "whether Platt or isotonic recalibration fixes it."
)


@st.cache_data(show_spinner=False)
def _train_and_score(n_bins: int):
    X_train, X_test, y_train, y_test = demo_data()
    models = fit_models(X_train, y_train)
    table = evaluate(models, X_test, y_test, n_bins=n_bins).sort_values("ece")
    probs: Dict[str, np.ndarray] = {
        name: m.predict_proba(X_test)[:, 1] for name, m in models.items()
    }
    return table, probs, y_test


with st.sidebar:
    st.header("Settings")
    n_bins = st.slider("Reliability bins", min_value=5, max_value=20, value=10, step=1)
    st.markdown(
        "**Methods compared**\n\n"
        "- **Uncalibrated** - raw GaussianNB scores\n"
        "- **Platt (sigmoid)** - fits a logistic map\n"
        "- **Isotonic** - non-parametric monotone map"
    )

table, probs, y_test = _train_and_score(n_bins)

best = table.index[0]
best_ece = float(table.loc[best, "ece"])
best_brier = float(table.loc[best, "brier"])
uncal_ece = float(table.loc["uncalibrated", "ece"]) if "uncalibrated" in table.index else float("nan")
improvement = (uncal_ece - best_ece) / uncal_ece * 100 if uncal_ece else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Best model (by ECE)", best.title())
c2.metric("Best ECE", f"{best_ece:.4f}", help="Expected Calibration Error - lower is better")
c3.metric(
    "Best Brier",
    f"{best_brier:.4f}",
    delta=f"-{improvement:.0f}% ECE vs uncalibrated" if improvement > 0 else None,
    delta_color="normal",
    help="Brier score - lower is better",
)

st.subheader("Reliability diagram")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfectly calibrated")

colors = {"uncalibrated": "#d1495b", "platt": "#2e86ab", "isotonic": "#2a9d8f"}
for name in ["uncalibrated", "platt", "isotonic"]:
    if name not in probs:
        continue
    mean_pred, frac_pos, _ = reliability_points(y_test, probs[name], n_bins=n_bins)
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        markersize=5,
        color=colors.get(name, None),
        label=name.title(),
    )

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives (observed)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Reliability curve: predicted vs observed")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.info(
    "**How to read this:** the dashed diagonal is perfect calibration. "
    "Points **below** the diagonal mean the model is **over-confident** "
    "(it predicts higher probabilities than reality delivers). Points "
    "**above** mean it is **under-confident**. The curve that hugs the "
    "diagonal most tightly is best calibrated."
)

st.subheader("Metrics (sorted by ECE, lower = better calibrated)")


def _highlight_best(row: pd.Series):
    return ["background-color: #d8f3dc" if row.name == best else "" for _ in row]


styled = (
    table.style.apply(_highlight_best, axis=1).format(
        {"brier": "{:.4f}", "log_loss": "{:.4f}", "roc_auc": "{:.4f}", "ece": "{:.4f}"}
    )
)
st.dataframe(styled, use_container_width=True)

st.caption(
    "Note: `roc_auc` measures ranking, not calibration - it barely changes "
    "after recalibration because the score *order* is preserved. Brier and "
    "ECE are what move."
)
