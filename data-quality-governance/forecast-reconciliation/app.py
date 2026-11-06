"""Streamlit front end: pick a reconciliation method, see its region forecasts next to what happened.

Tab one reconciles the fictional retailer with one method. Tab two shows the study. Verified headless with
`streamlit.testing.v1.AppTest`, widgets BY KEY.
"""

from __future__ import annotations

import explore as E
import streamlit as st

st.set_page_config(page_title="forecast reconciliation", layout="wide")
st.title("The regions do not sum to the forecast: seven ways to make them add up, scored")
tab_try, tab_study = st.tabs(["Reconcile it", "The study"])


@st.cache_data
def study() -> dict:
    return E.study()


with tab_try:
    m = st.selectbox("Method", E.METHODS, index=0, key="method")
    r = study()
    a = r["methods"][m]
    fc = r["regions_fc"][m]
    rows = [{"region": g, "forecast": f"{f:,.0f}", "actual": f"{x:,.0f}", "error": f"{100 * (f / x - 1):+.1f}%"}
            for g, f, x in zip(E.REGIONS, fc, r["regions_actual"])]
    rows.append({"region": "sum of regions", "forecast": f"{sum(fc):,.0f}", "actual": f"{r['total_actual']:,.0f}",
                 "error": f"{100 * (sum(fc) / r['total_actual'] - 1):+.1f}%"})
    st.table(rows)
    vs = a["vs_base"]
    if a["gap"] > 1e-9:
        st.error(f"{m}: the levels do not add up - the largest gap is {100 * a['gap']:.2f}% of the national "
                 "forecast. Each series was forecast on its own.")
    elif vs["region"] > 0.10:
        st.warning(f"{m}: adds up exactly, and region error is {100 * vs['region']:+.1f}% vs the unreconciled "
                   "forecasts. The 'regions sum to the forecast' check passes; it tests arithmetic, not accuracy.")
    else:
        st.success(f"{m}: adds up. RMSE vs unreconciled - total {100 * vs['total']:+.1f}%, region "
                   f"{100 * vs['region']:+.1f}%, store {100 * vs['store']:+.1f}%.")
    st.caption("Reconciliation redistributes the forecasts it is given. A cut that no history contains stays a miss "
               "under every method - see the study tab, section 5.")

with tab_study:
    st.image("audit.png")
    st.code(open("evidence.txt").read())
