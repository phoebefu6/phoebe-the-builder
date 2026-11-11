"""Streamlit front end: type a number, pick a writer, see whether it comes back bit for bit.

Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="float round-trip", layout="wide")
st.title("The number changed after a save: which writers give a float64 back bit for bit")
tab_try, tab_study = st.tabs(["Try a value", "The study"])

with tab_try:
    raw = st.text_input("A value (anything float() accepts, or an expression like 0.1+0.2 / 3)", "0.1+0.2",
                        key="value")
    writer = st.selectbox("Writer", list(E.WRITERS), index=list(E.WRITERS).index("%.15g"), key="writer")
    try:
        x = float(eval(raw, {"__builtins__": {}}, {"nan": float("nan"), "inf": float("inf")}))
    except Exception:  # noqa: BLE001 - any bad input becomes a message, not a traceback
        x = None
    if x is None:
        st.warning(f"Could not read {raw!r} as a number.")
    else:
        y = E.WRITERS[writer]([x])[0]
        c = E.compare([x], [y])
        st.table([{"": "sent", "value": repr(x), "bits": f"{E.bits(x):#x}"},
                  {"": "read back", "value": "lost" if y is None else repr(y),
                   "bits": "-" if y is None else f"{E.bits(y):#x}"}])
        st.caption(f"{writer}: {E.WHAT[writer]}")
        if c["exact"]:
            st.success(f"{writer} gave {x!r} back bit for bit.")
        elif c["lost"]:
            st.error(f"{writer} lost {x!r}: it came back empty or as NaN.")
        else:
            st.error(f"{writer} changed {x!r} to {y!r} ({c['max_ulp']:,} ULP away). A join or dedupe key on "
                     f"this column will no longer match the source.")

with tab_study:
    st.image("roundtrip.png")
    st.code(open("evidence.txt").read())
