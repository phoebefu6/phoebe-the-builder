"""Streamlit front end: pick a roll-up method, see its region totals next to the truth and where the dollars went.

Tab one rolls the fictional hierarchy up with one method. Tab two shows the study. Verified headless with
`streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="hierarchy reconciliation", layout="wide")
st.title("The regions do not sum to the total: roll a defective hierarchy up five ways")
tab_try, tab_study = st.tabs(["Roll it up", "The study"])


@st.cache_data
def study() -> dict:
    return E.study()


@st.cache_data
def truth() -> dict:
    return dict(E.truth(E.make_world()))


with tab_try:
    m = st.selectbox("Roll-up method", E.METHODS, index=0, key="method")
    r = study()
    a = r["methods"][m]
    tru = truth()
    rows = [{"bucket": k, "reported": f"{v:,.0f}", "true": f"{tru.get(k, 0):,.0f}" if k in E.REGIONS else "-"}
            for k, v in a["buckets"].items()]
    st.table(rows)
    st.info(f"Regions sum to the reported total: {'yes' if a['sums_to_total'] else 'NO'}. "
            "This check passes for every method; it only proves the report agrees with itself.")
    if not a["total_is_ledger"]:
        st.error(f"{m}: the total is {100 * a['total']:.1f}% of the ledger. {100 * a['dropped']:.1f}% of dollars "
                 f"reach no region and {100 * a['double']:.1f}% are counted twice.")
    elif a["double"] > 0:
        st.warning(f"{m}: ties to the ledger with a plug of {100 * r['plug_net']:.1f}%, while "
                   f"{100 * a['wrong']:.1f}% of dollars are in the wrong place underneath it.")
    else:
        st.success(f"{m}: ties to the ledger and names what it could not place: {100 * r['quarantined']:.1f}% in "
                   "Unassigned, listed by defect. Only a fix in the source hierarchy places those.")
    st.caption(f"Defects found in the extract: {r['defects']}")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
