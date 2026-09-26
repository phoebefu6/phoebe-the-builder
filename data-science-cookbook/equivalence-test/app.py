"""Streamlit front end: paste two groups and a margin, see what the data can and cannot claim.

What the app computes live is the user's own data - the t-test, TOST, the 90% CI - plus the exact
probability that a study of THEIR size could ever have declared equivalence. The stored study comes
from results.json.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import equiv as E
import numpy as np
import streamlit as st

st.set_page_config(page_title="No significant difference is not no difference", layout="wide")
R = json.load(open("results.json"))
EX = R["exemplar"]

st.title("'No significant difference' is not 'no difference'")
st.markdown(
    "A t-test that fails to find a difference has not found sameness. To claim two things are the "
    "same you need a margin - the smallest difference that would matter - and a test that the "
    "difference sits inside it (TOST). Paste your two groups."
)

tab1, tab2 = st.tabs(["Your data", "The study"])

with tab1:
    c1, c2 = st.columns(2)
    raw_x = c1.text_area("Group A (control), comma separated", ", ".join(map(str, EX["x"])), key="x")
    raw_y = c2.text_area("Group B (new), comma separated", ", ".join(map(str, EX["y"])), key="y")
    margin = st.number_input("Equivalence margin: the smallest difference that matters, in the data's units",
                             value=float(EX["margin"]), key="margin")
    try:
        x = [float(v) for v in raw_x.replace(";", ",").split(",") if v.strip()]
        y = [float(v) for v in raw_y.replace(";", ",").split(",") if v.strip()]
        res = E.analyse(x, y, margin)
    except ValueError as err:
        st.warning(f"Cannot analyse this input: {err}")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("difference (B - A)", f"{res['delta']:.3f}")
    m2.metric("t-test p (is there a difference?)", f"{res['p_diff']:.4f}")
    m3.metric("TOST p (is it inside the margin?)", f"{res['p_tost']:.4f}")
    st.write(f"90% CI for the difference: ({res['ci90'][0]:.3f}, {res['ci90'][1]:.3f}) "
             f"against margin +-{margin:g}")

    # What could a study this size ever have shown? Exact, in SD units of the pooled data.
    sd = res["se"] / np.sqrt(1 / len(x) + 1 / len(y))
    n_eq = int(round(2 / (1 / len(x) + 1 / len(y))))  # harmonic-mean n per group
    power0 = E.p_equivalent(max(n_eq, 2), 0.0, margin / sd)
    st.caption(f"If the two were EXACTLY equal, a study of this size (n ~ {n_eq} a group, margin = "
               f"{margin / sd:.2f} SD) declares equivalence with probability {power0:.3f}.")

    o = res["outcome"]
    if o == "inconclusive":
        st.error("INCONCLUSIVE. The t-test found no difference AND the data cannot rule out a difference "
                 "as large as the margin. Do not write 'no difference'. Write: 'the study could not "
                 "distinguish the two, and was not large enough to'.")
    elif o == "equiv_only":
        st.success("EQUIVALENT within the margin, and no detectable difference. This is the claim "
                   "'no significant difference' is usually mistaken for.")
    elif o == "both":
        st.info("EQUIVALENT AND DIFFERENT. There is a real difference, and it is too small to matter. "
                "Both statements are true; large studies produce this often.")
    else:
        st.warning("DIFFERENT, and not shown to be within the margin. The difference may matter.")

with tab2:
    G = R["grid"]
    st.subheader("How often the t-test says 'no significant difference', by true difference")
    st.dataframe([{"n per group": n, **{f"{f:.1f} x margin": round(next(
        r for r in G if r["n"] == n and r["delta_frac"] == f)["not_significant"], 4)
        for f in R["config"]["delta_fracs"]}} for n in R["config"]["ns"]], use_container_width=True)
    st.subheader("How often TOST says 'equivalent'")
    st.dataframe([{"n per group": n, **{f"{f:.1f} x margin": round(next(
        r for r in G if r["n"] == n and r["delta_frac"] == f)["equivalent"], 4)
        for f in R["config"]["delta_fracs"]}} for n in R["config"]["ns"]], use_container_width=True)
    st.subheader("n per group for an 80% chance of showing equivalence")
    st.dataframe(R["n_needed"], use_container_width=True)
