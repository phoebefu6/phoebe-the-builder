from __future__ import annotations

import streamlit as st
from faqgen import SAMPLE_DOC, chunk_document, generate_faq

st.set_page_config(page_title="FAQ Generator from Docs", layout="wide")
st.title("FAQ Generator from Docs")
st.caption("Stop answering the same question daily — turn your docs into a ready FAQ.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using heuristic generation (heading → grounded answer).")
    max_items = st.slider("Max FAQ entries", 3, 12, 8)
    use_sample = st.checkbox("Use sample doc", value=True)

if use_sample:
    doc = st.text_area("Documentation", value=SAMPLE_DOC, height=320)
else:
    doc = st.text_area("Paste your documentation (markdown or plain text)", height=320)

if st.button("Generate FAQ", type="primary"):
    if not doc.strip():
        st.warning("Add some documentation first.")
        st.stop()

    with st.spinner("Generating..."):
        try:
            faq = generate_faq(doc, api_key=api_key or None, max_items=max_items)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    if not faq.items:
        st.warning("No FAQ entries could be generated. Try a doc with clear sections.")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("FAQ entries", len(faq.items))
    c2.metric("Source chunks", len(chunk_document(doc)))

    st.subheader("Generated FAQ")
    for it in faq.items:
        with st.expander(it.question):
            st.write(it.answer)
            if it.source_chunk:
                st.caption("Grounded in:")
                st.code(it.source_chunk, language="text")

    st.download_button(
        "Download as Markdown",
        data=faq.to_markdown(),
        file_name="FAQ.md",
        mime="text/markdown",
    )
