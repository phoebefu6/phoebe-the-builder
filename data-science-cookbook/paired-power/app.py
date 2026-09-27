"""Streamlit front end: paste before and after, see what analysing them as two groups costs.

The app computes live on the user's own data: both p-values, the sample correlation, and the exact
power of each analysis at the user's n and correlation. The stored study comes from results.json.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import numpy as np
import paired as P
import streamlit as st

st.set_page_config(page_title="Paired data, analysed as paired", layout="wide")
R = json.load(open("results.json"))
EX = R["exemplar"]

st.title("You have before and after. Analyse it as before and after.")
st.markdown(
    "Running a two-sample t-test on paired data counts every unit's own level as noise. When the "
    "units are consistent over time that noise is most of the variance, and the test goes blind. "
    "Paste your pairs, in the same order in both boxes."
)

tab1, tab2 = st.tabs(["Your data", "The study"])

with tab1:
    c1, c2 = st.columns(2)
    raw_b = c1.text_area("Before, comma separated", ", ".join(map(str, EX["before"])), key="before")
    raw_a = c2.text_area("After, same units in the same order", ", ".join(map(str, EX["after"])), key="after")
    try:
        b = [float(v) for v in raw_b.replace(";", ",").split(",") if v.strip()]
        a = [float(v) for v in raw_a.replace(";", ",").split(",") if v.strip()]
        res = P.analyse(b, a)
    except ValueError as err:
        st.warning(f"Cannot analyse this input: {err}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mean change (after - before)", f"{res['delta']:.3f}")
    m2.metric("sample correlation", f"{res['rho_hat']:.3f}")
    m3.metric("paired t p", f"{res['p_paired']:.4f}")
    m4.metric("two-sample t p (wrong)", f"{res['p_independent']:.4f}")

    n, r = res["n"], float(np.clip(res["rho_hat"], -0.9, 0.95))
    sd = float(np.sqrt((np.var(b, ddof=1) + np.var(a, ddof=1)) / 2))
    shift = abs(res["delta"]) / sd
    if shift > 0:
        pp, pi = P.power_paired(n, r, shift), P.power_independent(n, r, shift)
        st.caption(f"If the true change were the one you observed ({shift:.2f} SD) and rho = {r:.2f}, a study of "
                   f"{n} pairs detects it with probability {pp:.3f} analysed as paired, {pi:.3f} as two groups.")

    be = P.break_even_rho(max(n, 3))
    if res["rho_hat"] < 0:
        st.error(f"NEGATIVE CORRELATION ({res['rho_hat']:.2f}). Here the two-sample t is not just weak, it is "
                 "anti-conservative: it rejects true nulls MORE than its alpha says. Use the paired test.")
    elif res["rho_hat"] < be:
        st.info(f"WEAK PAIRING (rho {res['rho_hat']:.2f} < break-even {be:.3f} at n = {n}). The pairing buys "
                "almost nothing, and the paired test's lost degrees of freedom cost a little power. The paired "
                "test is still valid - report it, because the design was paired.")
    elif res["p_paired"] < P.ALPHA <= res["p_independent"]:
        st.success(f"THE PAIRING IS THE FINDING. Paired p = {res['p_paired']:.4f}, two-sample p = "
                   f"{res['p_independent']:.4f}. Analysed as two groups, this change would have been reported "
                   "as 'no significant difference'.")
    else:
        st.warning("Both analyses agree here. The paired one is still the right one to report, and its "
                   "interval is the narrower, honest one.")

with tab2:
    G = R["grid"]
    st.subheader(f"Power at a {R['config']['delta_sd']} SD shift: paired / two-sample, same data")
    st.dataframe([{"n pairs": n, **{f"rho {rho}": "{:.3f} / {:.3f}".format(*next(
        (x["power_paired"], x["power_independent"]) for x in G if x["n"] == n and x["rho"] == rho))
        for rho in R["config"]["rhos"]}} for n in R["config"]["ns"]], width="stretch")
    st.subheader("Pairs needed for 80% power")
    st.dataframe(R["n_needed"], width="stretch")
    st.subheader("Break-even rho: below it, pairing costs power")
    st.dataframe(R["break_even"], width="stretch")
