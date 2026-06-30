from __future__ import annotations

import streamlit as st

from searcher import (
    SAMPLE_DOCS,
    SemanticIndex,
    keyword_baseline,
    make_voyage_embedder,
)

st.set_page_config(page_title="Semantic Search Engine", layout="wide")
st.title("Semantic Search Engine")
st.caption("Search by meaning, not just exact words — what Ctrl+F can't do.")

with st.sidebar:
    st.header("Corpus")
    raw = st.text_area(
        "Documents (one per line)",
        value="\n".join(SAMPLE_DOCS),
        height=320,
    )
    use_voyage = st.checkbox("Use Voyage dense embeddings (needs VOYAGE_API_KEY)", value=False)
    top_k = st.slider("Results", 1, 8, 3)

docs = [line.strip() for line in raw.splitlines() if line.strip()]

embed_fn = make_voyage_embedder() if use_voyage else None
if use_voyage and embed_fn is None:
    st.warning("Voyage unavailable (no VOYAGE_API_KEY or voyageai not installed) — using local TF-IDF.")

index = SemanticIndex(embed_fn=embed_fn).build(docs)

query = st.text_input("Search query", value="how do I get my money back")

if st.button("Search", type="primary") or query:
    if not docs:
        st.warning("Add some documents first.")
        st.stop()
    if not query.strip():
        st.stop()

    backend = "Voyage dense embeddings" if embed_fn is not None else "local TF-IDF"
    st.caption(f"Backend: {backend}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Semantic search")
        for h in index.search(query, top_k=top_k):
            st.markdown(f"**{h.score:.3f}** — {h.text}")
    with col2:
        st.subheader("Keyword (Ctrl+F baseline)")
        for h in keyword_baseline(query, docs, top_k=top_k):
            st.markdown(f"**{int(h.score)} hits** — {h.text}")
