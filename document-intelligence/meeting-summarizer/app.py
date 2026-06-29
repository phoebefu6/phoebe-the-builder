from __future__ import annotations

import streamlit as st

from summarizer import SAMPLE_TRANSCRIPT, summarize

st.set_page_config(page_title="Meeting Notes Summarizer", layout="wide")
st.title("Meeting Notes Summarizer")
st.caption("Turn a raw transcript into a TL;DR, decisions, action items, and open questions.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using heuristic extraction (regex-based cue matching).")
    use_sample = st.checkbox("Use sample transcript", value=True)

if use_sample:
    transcript = st.text_area("Transcript", value=SAMPLE_TRANSCRIPT, height=250)
else:
    transcript = st.text_area("Paste your meeting transcript", height=250)

if st.button("Summarize", type="primary"):
    if not transcript.strip():
        st.warning("Paste a transcript first.")
        st.stop()

    with st.spinner("Summarizing..."):
        try:
            result = summarize(transcript, api_key=api_key or None)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            st.stop()

    st.subheader("TL;DR")
    st.write(result.tldr)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Decisions")
        st.write("\n".join(f"- {d}" for d in result.decisions) or "_None found_")
    with col2:
        st.subheader("Action Items")
        st.write("\n".join(f"- {a}" for a in result.action_items) or "_None found_")
    with col3:
        st.subheader("Open Questions")
        st.write("\n".join(f"- {q}" for q in result.open_questions) or "_None found_")
