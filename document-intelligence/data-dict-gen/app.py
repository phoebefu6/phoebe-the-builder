from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from dictgen import (
    dictionary_to_dataframe,
    dictionary_to_markdown,
    generate_dictionary,
    sample_dataframe,
)

st.set_page_config(page_title="Data Dictionary Generator", layout="wide")
st.title("Data Dictionary Generator")
st.caption("Point it at a table, get a documented data dictionary — types, PII flags, descriptions.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using heuristic descriptions (profile-based).")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    table_name = st.text_input("Table name", value="customers")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.caption(f"Loaded {uploaded.name}: {df.shape[0]} rows × {df.shape[1]} columns.")
else:
    df = sample_dataframe()
    st.caption("Using the built-in sample table. Upload a CSV to document your own.")

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

if st.button("Generate dictionary", type="primary"):
    with st.spinner("Profiling columns..."):
        try:
            docs = generate_dictionary(df, api_key=api_key or None, table_name=table_name)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    pii_cols = [c.name for c in docs if c.is_pii]
    c1, c2, c3 = st.columns(3)
    c1.metric("Columns", len(docs))
    c2.metric("PII columns", len(pii_cols))
    c3.metric("Cols with nulls", sum(1 for c in docs if c.null_pct > 0))

    if pii_cols:
        st.warning("⚠️ PII detected in: " + ", ".join(pii_cols) + " — handle per your data policy.")

    st.subheader("Data dictionary")
    st.dataframe(dictionary_to_dataframe(docs), use_container_width=True)

    md = dictionary_to_markdown(docs, table_name)
    st.download_button("Download as Markdown", data=md, file_name=f"{table_name}_dictionary.md", mime="text/markdown")

    csv_buf = io.StringIO()
    dictionary_to_dataframe(docs).to_csv(csv_buf, index=False)
    st.download_button("Download as CSV", data=csv_buf.getvalue(), file_name=f"{table_name}_dictionary.csv", mime="text/csv")
