"""Churn Predictor — Streamlit UI.

Upload historical customer data with a churn label (or use the built-in
sample), train a classifier, and get a ranked at-risk list, churn drivers,
and held-out model quality — so you can act before customers leave.
"""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from churn import (
    find_label_column,
    sample_customers,
    score_customers,
    select_features,
    train_churn_model,
)

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Churn Predictor")
st.caption(
    "Spot at-risk customers *before* they leave. Train on your history, get a "
    "ranked risk list, the drivers behind churn, and an honest quality score."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Customer history CSV (with a churn label)", type=["csv"]
    )
    use_sample = st.button("Use sample customer base")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = sample_customers()
else:
    st.info(
        "Upload a CSV (one row per customer, numeric features, a binary churn "
        "column), or click **Use sample customer base** to try it."
    )
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)

label_guess = find_label_column(df)
if label_guess is None:
    st.error("No churn label found — need a binary column (e.g. churned 0/1, yes/no).")
    st.stop()

with st.sidebar:
    st.header("Model")
    label_col = st.selectbox(
        "Churn label column",
        df.columns,
        index=list(df.columns).index(label_guess),
    )
    all_feats = select_features(df, label_col)
    features = st.multiselect("Features", all_feats, default=all_feats)
    threshold = st.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)

if len(features) < 2:
    st.warning("Pick at least 2 features.")
    st.stop()

cm = train_churn_model(
    df, label_col=label_col, feature_cols=features, threshold=threshold
)

# --- Model quality ---------------------------------------------------------
st.subheader("Model quality (held-out test set)")
m = cm.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("ROC AUC", f"{m['auc']:.3f}")
c2.metric("Avg precision", f"{m['avg_precision']:.3f}")
c3.metric("Precision", f"{m['precision']:.3f}")
c4.metric("Recall", f"{m['recall']:.3f}")
st.caption(
    f"Base churn rate {m['churn_rate']:.1%} · trained on {m['n_train']} · "
    f"tested on {m['n_test']}. AUC 0.5 = coin flip, 1.0 = perfect. "
    "Lower the threshold to catch more churners (higher recall, lower precision)."
)

# --- Churn drivers ---------------------------------------------------------
st.subheader("What drives churn")
imp = pd.DataFrame(cm.importances, columns=["feature", "importance"])
fig, ax = plt.subplots(figsize=(7, 0.45 * len(imp) + 1))
ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#DC2626")
ax.set_xlabel("Feature importance")
st.pyplot(fig)
plt.close(fig)

# --- At-risk customers -----------------------------------------------------
st.subheader("At-risk customers")
scored = score_customers(cm, df)
band_counts = (
    scored["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
)
b1, b2, b3 = st.columns(3)
b1.metric("🔴 High risk", int(band_counts["High"]))
b2.metric("🟡 Medium risk", int(band_counts["Medium"]))
b3.metric("🟢 Low risk", int(band_counts["Low"]))

show_cols = ["churn_risk", "risk_band"] + [c for c in df.columns if c != label_col]
st.dataframe(scored[show_cols].head(25), use_container_width=True)

buf = io.StringIO()
scored.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download scored customers CSV",
    buf.getvalue(),
    file_name="customers_churn_scored.csv",
    mime="text/csv",
)
