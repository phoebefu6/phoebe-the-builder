from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from ner import SAMPLE_TEXT, extract_entities, group_by_label

COLOR = {
    "PERSON": "#c9e7ff", "ORG": "#ffe0b3", "MONEY": "#c9f2d0",
    "DATE": "#e6d4ff", "EMAIL": "#ffd6e0", "PERCENT": "#fff3b0", "GPE": "#d0f0f0",
}

st.set_page_config(page_title="Named Entity Extractor", layout="wide")
st.title("Named Entity Extractor")
st.caption('"Pull names, orgs, money, and dates from text" — highlighted and grouped, no model download.')

with st.sidebar:
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using the heuristic extractor (regex + capitalization).")

text = st.text_area("Text", value=SAMPLE_TEXT, height=200)

if st.button("Extract entities", type="primary") or text:
    if not text.strip():
        st.stop()
    ents = extract_entities(text, api_key=api_key or None)

    # highlighted text
    out, cursor = [], 0
    for e in sorted(ents, key=lambda x: x.start):
        if e.start < cursor or e.start < 0:
            continue
        out.append(html.escape(text[cursor:e.start]))
        color = COLOR.get(e.label, "#eeeeee")
        out.append(
            f'<span style="background:{color};padding:1px 4px;border-radius:3px;">'
            f'{html.escape(e.text)}<sub style="font-size:0.7em;color:#555;"> {e.label}</sub></span>'
        )
        cursor = e.end
    out.append(html.escape(text[cursor:]))
    st.markdown("### Annotated text")
    st.markdown("".join(out), unsafe_allow_html=True)

    grouped = group_by_label(ents)
    c1, c2 = st.columns(2)
    c1.metric("Entities found", len(ents))
    c2.metric("Entity types", len(grouped))

    st.subheader("By type")
    st.dataframe(
        pd.DataFrame([{"type": k, "count": len(v), "entities": ", ".join(v)} for k, v in grouped.items()]),
        use_container_width=True,
    )
