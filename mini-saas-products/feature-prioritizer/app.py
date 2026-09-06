from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from feature_prioritizer import IMPACT_SCALE, Feature, prioritize, to_frame

st.set_page_config(page_title="Feature Prioritization Tool", page_icon="📊", layout="wide")
st.title("📊 Feature Prioritization Tool")
st.caption(
    "Stop arguing about priorities without data. Score features with RICE = "
    "(Reach × Impact × Confidence) ÷ Effort, then rank and place them on a value/effort map."
)

with st.sidebar:
    st.subheader("RICE reference")
    st.write("**Impact scale:** " + ", ".join(f"{k} = {v}" for k, v in IMPACT_SCALE.items()))
    st.caption("Reach = users/quarter · Confidence = % (0-100) · Effort = person-months")

default_df = pd.DataFrame([
    {"feature": "Dark mode", "reach": 8000, "impact": 1.0, "confidence": 90, "effort": 2},
    {"feature": "SSO / SAML login", "reach": 1500, "impact": 2.0, "confidence": 80, "effort": 3},
    {"feature": "Mobile app", "reach": 12000, "impact": 3.0, "confidence": 60, "effort": 12},
    {"feature": "CSV export", "reach": 3000, "impact": 1.0, "confidence": 100, "effort": 1},
    {"feature": "AI recommendations", "reach": 10000, "impact": 2.0, "confidence": 50, "effort": 8},
    {"feature": "Two-factor auth", "reach": 5000, "impact": 2.0, "confidence": 90, "effort": 2},
    {"feature": "Custom themes", "reach": 2000, "impact": 0.5, "confidence": 80, "effort": 4},
    {"feature": "Slack integration", "reach": 4000, "impact": 1.0, "confidence": 85, "effort": 2},
])

st.subheader("Features")
edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, hide_index=True)

if st.button("Prioritize", type="primary"):
    feats = []
    for _, r in edited.iterrows():
        if not str(r["feature"]).strip():
            continue
        feats.append(Feature(str(r["feature"]), float(r["reach"]), float(r["impact"]),
                             float(r["confidence"]), float(r["effort"])))
    if not feats:
        st.warning("Add at least one feature.")
        st.stop()

    scored = prioritize(feats)
    df = to_frame(scored)

    top = scored[0]
    st.success(f"🏆 Top priority: **{top.name}** (RICE {top.rice}) — a {top.quadrant.lower()}.")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Ranked by RICE")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        d = df.sort_values("RICE")
        ax.barh(d["feature"], d["RICE"], color="#4361ee")
        ax.set_xlabel("RICE score")
        plt.tight_layout()
        st.pyplot(fig)
    with col_r:
        st.subheader("Value vs Effort")
        q_color = {"Quick win": "#2e9e5b", "Big bet": "#4361ee", "Fill-in": "#e0a800", "Time sink": "#d64545"}
        fig2, ax2 = plt.subplots(figsize=(6, 4.2))
        for s in scored:
            value = s.reach * s.impact * (s.confidence / 100.0)
            ax2.scatter(s.effort, value, s=90, color=q_color[s.quadrant])
            ax2.annotate(s.name, (s.effort, value), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax2.set_xlabel("Effort (person-months)")
        ax2.set_ylabel("Value (Reach × Impact × Conf)")
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Full ranking")
    st.dataframe(df, hide_index=True, use_container_width=True)
