"""Streamlit front end: one result, two analysts. The p-value and the Bayes factor side by side.

Works from raw values (one-sample, or paired differences) or from a reported result (n and p). The
Bayes factor is the JZS one JASP / BayesFactor / pingouin report; the prior scale is a slider because
the verdict depends on it. The stored study comes from results.json.

Verified headless with `streamlit.testing.v1.AppTest`, widgets addressed BY KEY.
"""

from __future__ import annotations

import json

import bayes as B
import numpy as np
import streamlit as st
from scipy import stats

st.set_page_config(page_title="p-value vs Bayes factor", layout="wide")
R = json.load(open("results.json"))
EX = R["exemplar"]

st.title("Two analysts, one dataset, two verdicts")
st.markdown(
    "A p-value asks how surprising the data are if there is no effect. A Bayes factor asks which of two "
    "stories - no effect, or an effect of about the size your prior expects - predicted the data better. "
    "At large n a tiny effect can be surprising under the first and still better predicted by 'no effect'."
)

tab1, tab2 = st.tabs(["Your result", "The study"])

with tab1:
    mode = st.radio("Input", ["A reported result (n and p)", "Raw values"], key="mode", horizontal=True)
    r = st.slider("Prior scale r (Cauchy, in SD units): the effect size H1 expects", 0.05, 2.0, B.R_JZS,
                  0.05, key="r")
    try:
        if mode.startswith("Raw"):
            raw = st.text_area("Values (one-sample, or after - before for paired data), comma separated",
                               "0.8, -0.2, 1.4, 0.3, 0.9, -0.5, 1.1, 0.6, 0.2, 1.3", key="values")
            res = B.analyse([float(v) for v in raw.replace(";", ",").split(",") if v.strip()], r)
        else:
            c1, c2 = st.columns(2)
            n = int(c1.number_input("n (observations, or pairs)", 3, 10 ** 8, EX["n"], key="n"))
            p = float(c2.number_input("two-sided p", 1e-12, 1.0, round(EX["p"], 4), format="%.4f", key="p"))
            if not 0 < p < 1:
                raise ValueError("p must be strictly between 0 and 1")
            t = float(stats.t.isf(p / 2, n - 1))
            bf10 = B.bf10_jzs(t, n, r)
            res = {"n": n, "t": t, "p": p, "d": t / np.sqrt(n), "bf10": bf10, "bf01": 1 / bf10, "r": r}
    except ValueError as err:
        st.warning(f"Cannot analyse this input: {err}")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("n", f"{res['n']:,}")
    m2.metric("effect size d", f"{res['d']:.4f}")
    m3.metric("p-value", f"{res['p']:.4f}")
    m4.metric("BF01 (above 1 favours no effect)", f"{res['bf01']:.3g}")

    sig, bf01 = res["p"] < B.ALPHA, res["bf01"]
    if sig and bf01 > 3:
        st.error(f"LINDLEY'S PARADOX. p = {res['p']:.4f} says 'significant'; the Bayes factor says the data are "
                 f"{bf01:.1f}x more likely under NO effect than under an effect of the size r = {r:.2f} expects. "
                 f"Both are right about different questions: the effect is detectably non-zero, and it is "
                 f"tiny (d = {res['d']:.3f}). Report the effect size and its interval, not either verdict alone.")
    elif sig and bf01 < 1 / 3:
        st.success(f"BOTH ANALYSTS AGREE: an effect. p = {res['p']:.4f}, BF10 = {1 / bf01:.1f}.")
    elif not sig and bf01 > 3:
        st.info(f"EVIDENCE OF ABSENCE. The p-value can only say 'not significant'; the Bayes factor adds that "
                f"the data favour no effect {bf01:.1f} to 1. This is the thing a p-value cannot give you.")
    else:
        st.warning(f"INCONCLUSIVE. BF01 = {bf01:.2f} is between 1/3 and 3. "
                   + ("A p below 0.05 with a Bayes factor this weak is common: the most any normal prior can "
                      f"give a p = 0.05 result is {R['ceiling']['closed_form']:.2f} to 1." if sig else
                      "Neither story predicted these data much better; collect more."))

    rs = [0.1, 0.2, 0.5, B.R_JZS, 1.0, 1.5]
    st.caption("Same data, other prior scales:  " + "   ".join(
        f"r = {x:.2f}: BF01 {1 / B.bf10_jzs(res['t'], res['n'], x):.3g}" for x in rs))

with tab2:
    st.subheader("Hold p at exactly 0.05 and grow n: the Bayes factor for the null")
    st.dataframe(R["lindley"], width="stretch")
    st.subheader("When the null is true: share of 'significant' results the Bayes factor calls null")
    st.dataframe([{k: v[k] for k in ("n", "p_significant", "bf10_over_3", "share_of_sig_bf_null")}
                  for v in R["null"]], width="stretch")
    st.subheader("A real 0.02 SD effect: verdict probabilities")
    st.dataframe([{k: v[k] for k in ("n", "p_significant", "bf10_over_3", "bf01_over_3")}
                  for v in R["tiny_effect"]], width="stretch")
