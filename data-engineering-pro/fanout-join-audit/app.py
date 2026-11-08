"""Streamlit front end: pick a join, see what it does to the report and which checks would have caught it.

Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="fan-out join audit", layout="wide")
st.title("Revenue went up after the join: which checks catch it before the aggregate")
tab_try, tab_study = st.tabs(["Break the join", "The study"])


@st.cache_data
def study() -> dict:
    return E.study()


with tab_try:
    k = st.selectbox("Join version", E.SCENARIOS, index=1, key="scenario")
    v = study()["scenarios"][k]
    st.table([{"measure": "rows vs correct join", "change": f"{100 * v['rows']:+.1f}%"},
              {"measure": "revenue vs correct join", "change": f"{100 * v['revenue']:+.1f}%"},
              {"measure": f"worst report cell ({v['worst_cell']})", "change": f"{100 * v['worst']:+.1f}%"}])
    st.table([{"check": c, "verdict": "ALARM" if v["fires"][c] else "pass"} for c in E.CHECKS])
    broken = k not in ("clean", "orphan_dups")
    missed = [c for c in E.CHECKS if broken and not v["fires"][c]]
    false = [c for c in E.CHECKS if not broken and v["fires"][c]]
    if broken and missed:
        st.error(f"{k}: the report is wrong (worst cell {100 * v['worst']:+.1f}%) and these checks pass: "
                 f"{', '.join(missed)}.")
    elif false:
        st.warning(f"{k}: the report is correct and these checks raise a false alarm: {', '.join(false)}.")
    elif broken:
        st.error(f"{k}: the report is wrong and every check fires.")
    else:
        st.success(f"{k}: the report is correct and every check is quiet.")
    st.caption("pk_preserved = every source order appears exactly once in the joined rows. Run it before you sum.")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
