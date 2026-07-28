from __future__ import annotations

import io
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from threshold import (
    Point,
    best_by,
    min_cost,
    roc_auc,
    sample_scores,
    sweep,
    under_constraint,
)

st.set_page_config(page_title="Threshold Explorer", layout="wide")
st.title("Threshold Explorer — what cutoff should I actually use?")
st.caption(
    "0.5 is a default, not a decision. Sweep every cutoff and pick the one that "
    "matches what your errors really cost."
)


@st.cache_data
def _sample() -> pd.DataFrame:
    y, s = sample_scores()
    return pd.DataFrame({"y_true": y, "y_score": s})


with st.sidebar:
    st.header("Data")
    up = st.file_uploader("CSV with y_true (0/1) and y_score (0-1)", type="csv")
    st.caption("No file? The imbalanced fraud-style sample loads by default.")
    st.header("Cost of being wrong")
    cost_fp = st.number_input("Cost of a false positive", 0.0, value=1.0, step=1.0)
    cost_fn = st.number_input("Cost of a false negative", 0.0, value=20.0, step=1.0)
    st.caption("Miss a fraud vs annoy a good customer — rarely a 1:1 trade.")
    st.header("Hard floor (SLA)")
    floor_metric = st.selectbox("Metric that must hold", ["precision", "recall"])
    floor_value = st.slider("Minimum value", 0.0, 1.0, 0.80, 0.01)
    min_flags = st.number_input("Minimum rows flagged", 1, value=30, step=5)
    st.caption(
        "Ignores cutoffs that flag too few rows — 3-for-3 is not precision 1.00."
    )

if up is not None:
    df = pd.read_csv(up)
    missing = {"y_true", "y_score"} - set(df.columns)
    if missing:
        st.error(f"CSV is missing column(s): {', '.join(sorted(missing))}")
        st.stop()
else:
    df = _sample()

try:
    points: List[Point] = sweep(df["y_true"], df["y_score"])
    auc = roc_auc(df["y_true"], df["y_score"])
except ValueError as e:
    st.error(str(e))
    st.stop()

prevalence = float(np.mean(df["y_true"]))
default = min(points, key=lambda p: abs(p.threshold - 0.5))
f1_best = best_by(points, "f1")
cheapest = min_cost(points, cost_fp, cost_fn)
other = "recall" if floor_metric == "precision" else "precision"
constrained: Optional[Point] = under_constraint(
    points, floor_metric, floor_value, other, min_flags=int(min_flags)
)

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Positive rate", f"{prevalence:.2%}")
c3.metric("ROC AUC", f"{auc:.4f}")

st.subheader("Four defensible cutoffs — they rarely agree")
cand = [
    ("Default 0.5", default, "What you ship if you never think about it"),
    ("Best F1", f1_best, "Balanced, but assumes FP and FN hurt equally"),
    (
        "Cheapest",
        cheapest,
        f"Minimises cost at FP={cost_fp:g} / FN={cost_fn:g}",
    ),
]
if constrained is not None:
    cand.append(
        (
            f"{floor_metric} >= {floor_value:.2f}",
            constrained,
            f"Best {other} that still clears the floor",
        )
    )

cols = st.columns(len(cand))
for col, (label, pt, why) in zip(cols, cand):
    col.metric(label, f"{pt.threshold:.2f}", delta=f"F1 {pt.f1:.3f}", delta_color="off")
    col.caption(why)
    col.write(
        f"P **{pt.precision:.2f}** · R **{pt.recall:.2f}** · "
        f"flags **{pt.flag_rate:.1%}** · cost **{pt.cost(cost_fp, cost_fn):,.0f}**"
    )

if constrained is None:
    st.warning(
        f"No cutoff reaches {floor_metric} >= {floor_value:.2f} while flagging at least "
        f"{int(min_flags)} rows. That SLA is unreachable with this model — "
        "retrain, don't re-threshold."
    )

st.divider()
tab_curves, tab_table = st.tabs(["Curves", "Full sweep table"])

with tab_curves:
    thr = [p.threshold for p in points]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, vals in [
        ("precision", [p.precision for p in points]),
        ("recall", [p.recall for p in points]),
        ("f1", [p.f1 for p in points]),
    ]:
        ax1.plot(thr, vals, label=name, lw=2)
    ax1.axvline(cheapest.threshold, color="crimson", ls="--", lw=1.4, label="cheapest")
    ax1.axvline(0.5, color="grey", ls=":", lw=1.4, label="default 0.5")
    ax1.set(xlabel="threshold", ylabel="score", title="Metrics vs threshold", ylim=(0, 1.02))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    costs = [p.cost(cost_fp, cost_fn) for p in points]
    ax2.plot(thr, costs, color="crimson", lw=2)
    ax2.axvline(cheapest.threshold, color="crimson", ls="--", lw=1.4)
    ax2.axvline(0.5, color="grey", ls=":", lw=1.4)
    ax2.set(xlabel="threshold", ylabel="total cost", title="Cost curve")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)

    saved = default.cost(cost_fp, cost_fn) - cheapest.cost(cost_fp, cost_fn)
    st.info(
        f"Moving from 0.50 to {cheapest.threshold:.2f} saves "
        f"**{saved:,.0f}** cost units on this dataset "
        f"({saved / max(default.cost(cost_fp, cost_fn), 1):.1%} of the default's bill)."
    )

with tab_table:
    table = pd.DataFrame([p.as_row(cost_fp, cost_fn) for p in points])
    st.dataframe(table, use_container_width=True, height=420)
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    st.download_button("Download sweep as CSV", buf.getvalue(), "threshold_sweep.csv")
