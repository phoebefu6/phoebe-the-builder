"""Streamlit front end: type an ID, pick the odd row that shares its column, see what each reader hands back.

Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="json number precision", layout="wide")
st.title("The ID lost its last digits: which JSON readers give a 64-bit ID back exactly")
tab_try, tab_study = st.tabs(["Try an ID", "The study"])

with tab_try:
    raw = st.text_input("An integer ID", "1801234567890123457", key="value").strip()
    context = st.selectbox("One more row in the same column", list(E.CONTEXTS), index=1, key="context")
    if not raw.lstrip("-").isdigit():
        st.warning(f"{raw!r} is not an integer.")
    else:
        x = int(raw)
        rows, bad = [], []
        for name in list(E.READERS) + [k for k in E.FIXES if k not in E.READERS]:
            fn = E.READERS.get(name) or E.FIXES[name]
            try:
                got, dtype = fn(E.ndjson([x], E.CONTEXTS[context]))
                y = got[0]
                verdict = "exact" if y is not None and y == x else "WRONG"
            except Exception as e:  # noqa: BLE001 - a refusal is shown, not raised
                y, dtype, verdict = None, "-", f"refused ({type(e).__name__})"
            if verdict == "WRONG":
                bad.append(name)
            rows.append({"reader": name, "came back": "-" if y is None else str(y), "column type": dtype,
                         "verdict": verdict})
        st.table(rows)
        if bad:
            st.error(f"{len(bad)} readers changed {x} with no error: {', '.join(bad)}. A join on this ID will "
                     f"miss, or match a different record.")
        else:
            st.success(f"Every reader that accepted the file gave {x} back exactly.")

with tab_study:
    st.image("precision.png")
    st.code(open("evidence.txt").read())
