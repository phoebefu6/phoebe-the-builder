from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from drift_detector import (
    PSI_MODERATE,
    PSI_SIGNIFICANT,
    detect_drift,
    emit_alert,
    make_sample_data,
)

st.set_page_config(page_title="Model Drift Detector", page_icon="📉")
st.title("📉 Model Drift Detector")
st.caption(
    "Models degrade silently in production. This compares a live window to the training "
    "reference via Population Stability Index (PSI) and alerts before accuracy quietly rots."
)

with st.sidebar:
    st.subheader("Simulate production")
    drift_strength = st.slider("Drift strength", 0.0, 3.0, 1.5, 0.1,
                               help="Shifts the 'income' feature and degrades the model score.")
    bins = st.slider("PSI bins", 5, 20, 10)
    model_name = st.text_input("Model name", "churn-model")
    st.info("Set ALERT_WEBHOOK_URL to post real Slack alerts. Otherwise alerts dry-run.")
    st.caption(f"Bands: moderate ≥ {PSI_MODERATE}, significant ≥ {PSI_SIGNIFICANT}")

ref, prod = make_sample_data(drift_strength=drift_strength)
report = detect_drift(ref, prod, prediction_col="score", bins=bins)

if report.alert:
    st.error("🚨 DRIFT ALERT")
    for r in report.reasons:
        st.write(f"- {r}")
else:
    st.success("✅ No significant drift — model inputs and outputs are stable.")

st.subheader("Per-feature PSI")
df = report.to_frame()
color_map = {"none": "#2e9e5b", "moderate": "#e0a800", "significant": "#d64545"}
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(df["feature"], df["psi"], color=[color_map[s] for s in df["severity"]])
ax.axhline(PSI_MODERATE, color="#e0a800", linestyle="--", linewidth=1, label=f"moderate ({PSI_MODERATE})")
ax.axhline(PSI_SIGNIFICANT, color="#d64545", linestyle="--", linewidth=1, label=f"significant ({PSI_SIGNIFICANT})")
ax.set_ylabel("PSI")
ax.set_title(f"{model_name} — drift by feature")
ax.legend()
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Detail")
st.dataframe(df, hide_index=True, use_container_width=True)

st.subheader("Alerting")
st.code(emit_alert(report, model_name))
