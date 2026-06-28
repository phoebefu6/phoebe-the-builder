from __future__ import annotations

import streamlit as st

from qa import answer_question, chunk_text, extract_pages

st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("PDF Q&A Bot")
st.caption("Ask questions about a policy manual instead of reading all 200 pages.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — answers will fall back to extractive mode (best matching excerpt).")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

with open("_uploaded.pdf", "wb") as f:
    f.write(uploaded.read())

try:
    pages = extract_pages("_uploaded.pdf")
except Exception as e:
    st.error(f"Couldn't read this PDF: {e}")
    st.stop()

if not any(p.strip() for p in pages):
    st.error("No extractable text found (this may be a scanned/image-only PDF).")
    st.stop()

chunks = chunk_text(pages)
st.success(f"Indexed {len(pages)} pages into {len(chunks)} searchable chunks.")

question = st.text_input("Ask a question about this document")
if question:
    with st.spinner("Searching document..."):
        answer = answer_question(chunks, question, api_key=api_key or None)
    st.subheader("Answer")
    st.write(answer)
