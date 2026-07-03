from __future__ import annotations

import streamlit as st

from kb import SAMPLE_DOCS, KnowledgeBase

st.set_page_config(page_title="Knowledge Base Builder", layout="wide")
st.title("Knowledge Base Builder")
st.caption("Turn scattered docs into a searchable, cited knowledge base — before the expert leaves.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — extractive answers (best cited passage). Add a key for synthesized answers.")
    top_k = st.slider("Passages to retrieve", 1, 5, 3)

# Build / cache the KB in session state
if "kb" not in st.session_state:
    kb = KnowledgeBase()
    for name, text in SAMPLE_DOCS.items():
        kb.add_document(name, text)
    st.session_state.kb = kb
    st.session_state.docs = dict(SAMPLE_DOCS)

kb: KnowledgeBase = st.session_state.kb

st.subheader("Add to the knowledge base")
c1, c2 = st.columns([1, 3])
new_name = c1.text_input("Doc name", value="", placeholder="runbook.md")
new_text = c2.text_area("Doc content", value="", height=100, placeholder="Paste any doc, wiki page, or notes...")
if st.button("Ingest document"):
    if new_name.strip() and new_text.strip():
        n = kb.add_document(new_name, new_text)
        st.session_state.docs[new_name] = new_text
        st.success(f"Ingested {new_name}: {n} chunks. KB now has {len(kb.chunks)} chunks.")
    else:
        st.warning("Give the doc a name and content.")

st.caption("Docs in KB: " + ", ".join(st.session_state.docs.keys()))

st.subheader("Ask the knowledge base")
query = st.text_input("Question", value="how do I roll back a bad deploy?")
if st.button("Ask", type="primary"):
    if not query.strip():
        st.stop()
    with st.spinner("Retrieving..."):
        answer = kb.ask(query, api_key=api_key or None, top_k=top_k)

    st.markdown("### Answer")
    st.write(answer.text)
    if answer.sources:
        st.markdown("**Sources:** " + ", ".join(f"`{s}`" for s in answer.sources))
        with st.expander("Retrieved passages"):
            for p in answer.passages:
                st.markdown(f"**`{p.doc}`**")
                st.write(p.text)
    else:
        st.info("No relevant passages found. Try ingesting more docs.")
