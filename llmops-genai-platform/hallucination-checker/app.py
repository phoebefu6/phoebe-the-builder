from __future__ import annotations

# Streamlit UI for the Hallucination Detector. Paste a source context and an
# LLM answer; the app scores groundedness per claim and flags sentences the
# context does not support. Fully offline lexical support signal - swap in an
# NLI model or LLM judge for production.

import pandas as pd
import streamlit as st

from detector import SAMPLES, check_answer, run_samples

st.set_page_config(page_title="Hallucination Detector", page_icon="🔎",
                   layout="wide")

st.title("🔎 Hallucination Detector")
st.caption(
    "Your RAG system retrieves context, the LLM writes an answer - and "
    "sometimes states things the context never supported. This scores "
    "*groundedness* per claim and flags unsupported sentences before they "
    "reach a user. Offline lexical signal; swap in an NLI/LLM judge for prod."
)

threshold = st.slider("Support threshold (higher = stricter)", 0.1, 0.9, 0.35, 0.05)

rows = run_samples(threshold)
flagged = sum(1 for r in rows if not r["grounded"])
c1, c2, c3 = st.columns(3)
c1.metric("Sample answers", len(rows))
c2.metric("Flagged", flagged)
c3.metric("Clean", len(rows) - flagged)

st.subheader("Sample RAG answers")
df = pd.DataFrame(rows)[["label", "groundedness", "grounded", "flagged"]]
st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Try your own")
preset = st.selectbox("Load a sample", ["(custom)"] + [s["label"] for s in SAMPLES])
d = next((s for s in SAMPLES if s["label"] == preset), None)

context = st.text_area("Source context (what was retrieved)",
                       value=d["context"] if d else "", height=140)
answer = st.text_area("LLM answer to check",
                      value=d["answer"] if d else "", height=120)

if st.button("Check", type="primary") and context.strip() and answer.strip():
    res = check_answer(answer, context, threshold)
    if res.grounded:
        st.success(f"Grounded · groundedness {res.groundedness:.2f}")
    else:
        st.error(f"{res.n_flagged} unsupported claim(s) · groundedness {res.groundedness:.2f}")
    for c in res.claims:
        icon = "✅" if c.grounded else "🚩"
        st.markdown(f"{icon} **{c.support:.2f}** — {c.claim}")
        if not c.grounded:
            st.caption(f"   ↳ {c.reason}")

st.divider()
st.caption("Day 88 of Phoebe's FDE build sprint · LLMOps & GenAI Platform · "
           "lexical support is a screen, not a verdict - pair with a judge model.")
