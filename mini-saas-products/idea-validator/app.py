from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from idea_validator import (
    DIMENSIONS,
    SAMPLE_IDEA,
    SAMPLE_SCORES,
    validate_idea,
)

st.set_page_config(page_title="Startup Idea Validator", page_icon="🚀")
st.title("🚀 Startup Idea Validator")
st.caption(
    "We build before we validate. Score your idea across five dimensions, get a Lean Canvas, "
    "and — most importantly — the cheapest experiment to test your riskiest assumption first."
)

with st.sidebar:
    st.subheader("Settings")
    use_claude = st.checkbox("Use Claude to fill canvas + score (needs ANTHROPIC_API_KEY)", value=False)
    st.info("Without Claude, you self-score each dimension below and the tool builds the canvas scaffold, verdict, and experiments.")

idea = st.text_area("Your startup idea", SAMPLE_IDEA, height=100)

st.subheader("Self-score each dimension (1 = weak, 5 = strong)")
scores = {}
cols = st.columns(len(DIMENSIONS))
for col, dim in zip(cols, DIMENSIONS):
    scores[dim] = col.slider(dim, 1, 5, SAMPLE_SCORES.get(dim, 3), key=dim)

if st.button("Validate idea", type="primary"):
    result = validate_idea(idea, scores=None if use_claude else scores, use_claude=use_claude)

    st.metric("Overall score", f"{result.overall} / 5")
    st.info(f"**Verdict:** {result.verdict}")
    if result.rationale:
        st.caption(f"Investor take: {result.rationale}")

    # radar chart of the five dimensions
    labels = list(result.scores.keys())
    vals = list(result.scores.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals_c, angles_c = vals + vals[:1], angles + angles[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    ax.plot(angles_c, vals_c, color="#4361ee", linewidth=2)
    ax.fill(angles_c, vals_c, color="#4361ee", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_title("Validation scorecard")
    st.pyplot(fig)

    st.subheader("🎯 Validate these first (riskiest assumptions)")
    for e in result.experiments:
        st.markdown(f"**{e['assumption']}** (score {result.scores[e['assumption']]}/5)")
        st.write(e["experiment"])

    st.subheader("Lean Canvas")
    for name, text in result.lean_canvas.items():
        with st.expander(name):
            st.write(text)
