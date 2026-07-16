"""Model Card Generator — Streamlit UI.

Fill in the human-authored fields (intended use, training data, ethics),
train/score on the built-in demo model (or your own via the notebook), and
download a ready-to-commit Model Card in Markdown.
"""
from __future__ import annotations

import streamlit as st

from model_card import compute_metrics, demo_train, generate_model_card

st.set_page_config(page_title="Model Card Generator", page_icon="📇", layout="wide")
st.title("📇 Model Card Generator")
st.caption(
    "Models ship with no docs. Introspect the estimator, score it, add the "
    "human context, and export a Google-style Model Card in one click."
)

model, X_te, y_te, region_te = demo_train()

with st.sidebar:
    st.header("Human-authored fields")
    model_name = st.text_input("Model name", "Customer Churn Classifier")
    version = st.text_input("Version", "1.0.0")
    owners = st.text_input("Owners", "Data Science Team")
    intended_use = st.text_area(
        "Intended use",
        "Flag customers at risk of churning for proactive retention outreach.",
    )
    training_data = st.text_area(
        "Training data",
        "1,500 synthetic customer records (tenure, spend, support tickets, region).",
    )
    slice_on = st.checkbox("Break down performance by region", value=True)

res = compute_metrics(model, X_te, y_te, slices=region_te if slice_on else None, slice_name="region")

c1, c2, c3 = st.columns(3)
c1.metric("Task", res["task"])
c2.metric("Test rows", f"{res['n_test']:,}")
c3.metric("F1 (weighted)", f"{res['overall'].get('f1', 0):.3f}")

card = generate_model_card(
    model,
    X_te,
    y_te,
    model_name=model_name,
    version=version,
    owners=owners,
    intended_use=intended_use,
    training_data=training_data,
    slices=region_te if slice_on else None,
    slice_name="region",
)

st.subheader("Generated Model Card")
st.markdown(card)

st.download_button(
    "⬇️ Download MODEL_CARD.md",
    data=card,
    file_name="MODEL_CARD.md",
    mime="text/markdown",
    type="primary",
)

with st.expander("Raw markdown"):
    st.code(card, language="markdown")
