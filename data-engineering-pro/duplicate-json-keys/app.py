"""Streamlit front end: paste one JSON object, see which value each reader hands back for its repeated key.

Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import json

import explore as E
import streamlit as st

st.set_page_config(page_title="duplicate json keys", layout="wide")
st.title("Which value wins? One JSON object with a repeated key, through every reader")
tab_try, tab_study = st.tabs(["Try a document", "The study"])

with tab_try:
    doc = st.text_area("One JSON object", '{"user":"ana","role":"viewer","role":"admin"}', key="doc")
    try:
        parsed = json.loads(doc)
    except ValueError as e:
        st.warning(f"Not valid JSON: {e}")
        parsed = None
    if parsed is not None and not isinstance(parsed, dict):
        st.warning("Paste a JSON object ({...}); duplicate keys only exist inside objects.")
    elif parsed is not None:
        res = E.check(doc)
        st.table(res["rows"])
        answers = {r[next(k for k in r if k != "reader")] for r in res["rows"]}
        answers = {a for a in answers if not a.startswith("refused")}
        if res["dups"] and len(answers) > 1:
            st.error(f"Key {res['key']!r} appears twice and the readers disagree: {', '.join(sorted(answers))}. "
                     f"A validator and a consumer built on different readers will act on different values.")
        elif res["dups"]:
            st.info(f"Key {res['key']!r} appears twice; every reader that accepted it returned the same value.")
        else:
            st.success("No top-level key repeats. (Nested duplicates are checked in the study tab.)")

with tab_study:
    st.image("dupkeys.png")
    st.code(open("evidence.txt").read())
