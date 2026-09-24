"""Streamlit front end: type in your conversion counts, get all four tests and what each one costs.

The stored study comes from results.json. What the app computes live is the user's own design - its
four p-values, and each test's EXACT worst-case false-alarm rate at those arm sizes - which is the
thing a reader needs to decide which answer to believe.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import numpy as np
import proportion as P
import streamlit as st

st.set_page_config(page_title="Did conversion actually move?", layout="wide")

R = json.load(open("results.json"))
MAX_N = 300   # edge case: exact size needs every table; beyond this the live grid gets slow

st.title("Did conversion actually move?")
st.markdown(
    "Four tests get run on a 2x2 conversion table as though they were interchangeable. "
    "**Two of them are the same test** (the z-test and the uncorrected chi-square), and the rest "
    "can split on the same counts. Enter yours."
)

tab1, tab2, tab3 = st.tabs(["Your table", "The study", "The rule of five"])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        n1 = st.number_input("control visitors", 2, MAX_N, 20, key="n1")
        x1 = st.number_input("control conversions", 0, MAX_N, 0, key="x1")
        n2 = st.number_input("variant visitors", 2, MAX_N, 20, key="n2")
        x2 = st.number_input("variant conversions", 0, MAX_N, 4, key="x2")
    n1, n2 = int(n1), int(n2)
    x1, x2 = min(int(x1), n1), min(int(x2), n2)
    if x1 == 0 and x2 == 0 or (x1 == n1 and x2 == n2):
        st.warning("Nobody (or everybody) converted in both arms: every test returns p = 1.")

    pv = P.all_pvalues(n1, n2)
    ps = np.linspace(0.01, 0.5, 25)
    rows = []
    for t in P.TESTS:
        size = float(P.size_curve(pv[t], n1, n2, ps).max())
        rows.append({"test": P.LABEL[t], "p-value": round(float(pv[t][x1, x2]), 4),
                     "verdict": "moved" if pv[t][x1, x2] < P.ALPHA else "not shown",
                     "worst false-alarm rate at these arm sizes": round(size, 4),
                     "calibration": P.verdict(size)})
    with right:
        st.metric("control rate vs variant rate", f"{x1 / n1:.1%} vs {x2 / n2:.1%}")
        st.dataframe(rows, use_container_width=True)
        calls = {r["verdict"] for r in rows}
        if len(calls) > 1:
            st.error("The tests DISAGREE on this table. Believe the ones whose calibration "
                     "column says ok - an INFLATED test says 'moved' too often by construction, a "
                     "CONSERVATIVE one says 'not shown' too often.")
        else:
            st.success(f"All four tests agree: {calls.pop()}.")

with tab2:
    st.subheader("Exact worst-case false-alarm rate, nominal 5%")
    st.dataframe([{"n1/n2": f"{r['n1']}/{r['n2']}",
                   **{P.LABEL[t]: f"{r[t]:.4f} {r[t + '_verdict']}" for t in P.TESTS}}
                  for r in R["size"]], use_container_width=True)
    st.subheader("Exact power")
    st.dataframe([{"rates": f"{r['p1']} vs {r['p2']}", "n1/n2": f"{r['n1']}/{r['n2']}",
                   **{P.LABEL[t]: round(r[t], 4) for t in P.TESTS}} for r in R["power"]],
                 use_container_width=True)

with tab3:
    ec = R["expected_count_rule"]
    st.markdown(
        f"The rule *use Fisher when any expected count is below 5*, scored against the z-test's "
        f"exact false-alarm rate on {sum(ec.values())} design x rate cells: it calls "
        f"**{ec['safe_but_broken']} broken cells safe**, and "
        f"**{ec['unsafe_but_fine']} of the {ec['unsafe_but_fine'] + ec['unsafe_broken']}** "
        f"cells it sends to Fisher were already fine."
    )
    only_broken = st.checkbox("show only cells the rule calls safe but are inflated",
                              key="only_broken")
    cells = R["expected_count_cells"]["safe_but_broken"] if only_broken else (
        R["expected_count_cells"]["safe_but_broken"] + R["expected_count_cells"]["unsafe_but_fine"])
    st.dataframe(cells, use_container_width=True)
