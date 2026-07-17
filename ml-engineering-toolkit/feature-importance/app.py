"""Feature Importance Explainer — Streamlit UI.

Train the demo model, run three independent importance methods, and see where
they agree (trust it) and where they disagree (investigate) — the evidence a
skeptical stakeholder needs before they'll sign off on a model.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from importance import demo_data, explain, train_model

st.set_page_config(page_title="Feature Importance Explainer", page_icon="🔍", layout="wide")
st.title("🔍 Feature Importance Explainer")
st.caption(
    "\"Stakeholders don't trust the model.\" One importance number is easy to "
    "doubt. Three independent methods that agree is evidence. This scores every "
    "feature by impurity, permutation, and drop-column, then flags consensus."
)

with st.sidebar:
    st.header("Settings")
    seed = st.number_input("Random seed", value=42, step=1)
    st.markdown(
        "**Methods**\n\n"
        "- **Impurity** — tree's built-in split gain (fast, biased to "
        "high-cardinality)\n"
        "- **Permutation** — shuffle a column, measure AUC drop (model-agnostic)\n"
        "- **Drop-column** — retrain without it (honest, expensive)"
    )

X, y = demo_data(seed=int(seed))
model, X_tr, X_te, y_tr, y_te = train_model(X, y, seed=int(seed))
res = explain(model, X_tr, y_tr, X_te, y_te, seed=int(seed))
tbl = res["table"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test AUC", f"{res['baseline_auc']:.3f}")
c2.metric("Trusted features", res["n_trusted"])
c3.metric("Noise (drop-safe)", res["n_noise"])
c4.metric("Needs review", res["n_review"])

st.subheader("Consensus ranking")
plot_df = tbl[["impurity_norm", "permutation_norm", "drop_column_norm"]].iloc[::-1]
plot_df.columns = ["impurity", "permutation", "drop-column"]
fig, ax = plt.subplots(figsize=(9, 0.5 * len(plot_df) + 1))
plot_df.plot.barh(ax=ax, width=0.8)
ax.set_xlabel("normalized importance (0-1)")
ax.set_title("Three methods, side by side")
ax.legend(loc="lower right")
fig.tight_layout()
st.pyplot(fig)

st.subheader("Verdict table")
show = tbl[["impurity", "permutation", "drop_column", "consensus", "verdict"]].round(4)


def _color(v: str) -> str:
    return {
        "trusted": "background-color: #d1e7dd",
        "noise": "background-color: #f8d7da",
        "review": "background-color: #fff3cd",
    }.get(v, "")


st.dataframe(show.style.map(_color, subset=["verdict"]), use_container_width=True)

st.info(
    "**How to read this:** *trusted* = top-half by all three methods, safe to "
    "put in front of a stakeholder. *noise* = bottom-half by all three, safe to "
    "drop. *review* = the methods disagree — usually a redundant/correlated "
    "feature; a human should decide."
)
