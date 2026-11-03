"""Streamlit front end: pick a month, see what each rebuild strategy says the published report was.

Tab one rebuilds one month four ways next to what was actually published, and the as-is restatement next to it.
Tab two shows the study. Verified headless with `streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="reference data versioning", layout="wide")
st.title("The country list changed: rebuild last year's regional report")
tab_try, tab_study = st.tabs(["Rebuild a month", "The study"])


@st.cache_data
def reports() -> dict:
    w = E.make_world()
    out = {k: E.report(w, k)[0] for k in E.STRATEGIES}
    out["as-is (50/50 split)"] = E.restate(w, "even")
    return {k: {m: dict(v[m]) for m in range(E.MONTHS)} for k, v in out.items()}


with tab_try:
    m = st.slider("Report month", 0, E.MONTHS - 1, 19, key="month")
    reps = reports()
    pub = reps["bitemporal"][m]
    rows = [{"strategy": "published", **{r: f"{pub.get(r, 0):,.0f}" for r in E.REGIONS}}]
    wrong = []
    for k, rep in reps.items():
        rows.append({"strategy": k, **{r: f"{rep[m].get(r, 0):,.0f}" for r in E.REGIONS}})
        if k in E.STRATEGIES and any(abs(rep[m].get(r, 0) - pub.get(r, 0)) > 0.5 for r in E.REGIONS):
            wrong.append(k)
    st.table(rows)
    st.warning(f"Month {m}: {', '.join(wrong)} do not reproduce what was published.")
    if "scd2_asof" in wrong:
        st.error(f"Month {m}: even the as-of join misses - a region move effective by month {m} was recorded after "
                 "the report went out. Only the bitemporal table (or a stored snapshot) reproduces it.")
    else:
        st.success(f"Month {m}: the as-of join reproduces this month. It fails on "
                   f"{36 - E.study()['asw']['scd2_asof']['months']} others.")
    st.caption("as-is moves every past sale to the region its territory sits in today: right for a trend, wrong "
               "for 'what did we report'. Both are valid questions; the failure is not knowing which one you answered.")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
