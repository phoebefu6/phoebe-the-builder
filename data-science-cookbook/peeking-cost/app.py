"""Price a stopping rule before you commit to it.

Type in the experiment you are about to run -- traffic, expected lift, how many
times somebody will look -- and this re-solves the boundaries and measures what
each rule would do to it. The point of the app is the second table: the effect
size every rule would report back, which is the number the roadmap gets built on
and the one no significance test protects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sequential import (
    Trial,
    bonferroni_bounds,
    equal_looks,
    first_crossing,
    msprt_crossing,
    naive_bounds,
    obf_bounds,
    pocock_bounds,
    score,
)
from sequential import simulate as _simulate

st.set_page_config(page_title="Peeking cost", page_icon="👀", layout="wide")
st.title("👀 What does looking cost?")
st.caption(
    "Day 164 - peeking-cost. Boundaries solved from the Armitage-McPherson recursion; "
    "every rate below is measured on simulated Bernoulli traffic, not read off a formula."
)

with st.sidebar:
    st.header("The experiment")
    p0 = st.number_input("Control conversion rate", 0.001, 0.9, 0.10, 0.005, format="%.3f")
    lift = st.slider("True relative lift, if there is one", 0.0, 0.50, 0.10, 0.01)
    n_max = st.select_slider("Visitors per arm at the planned end",
                             [2_000, 5_000, 10_000, 20_000, 50_000, 100_000], 20_000)
    k = st.slider("How many times will somebody look?", 1, 60, 20)
    alpha = st.select_slider("Alpha", [0.01, 0.05, 0.10], 0.05)
    sims = st.select_slider("Simulated experiments", [5_000, 20_000, 50_000], 20_000)
    st.caption(
        "The false-positive rate depends on the NUMBER of looks, not the calendar: "
        "ten looks in a day cost the same as ten looks in ten days."
    )

p1 = p0 * (1 + lift)
looks = equal_looks(k, n_max)
tau = max(p1 - p0, 1e-4)


@st.cache_data(show_spinner=False)
def boundaries(k: int, alpha: float):
    return {
        "naive peek": naive_bounds(k, alpha),
        "Bonferroni": bonferroni_bounds(k, alpha),
        "Pocock": pocock_bounds(k, alpha, step=0.005),
        "O'Brien-Fleming": obf_bounds(k, alpha, step=0.005),
    }


@st.cache_data(show_spinner=False)
def run(looks_t: tuple, p0: float, p1: float, sims: int, alpha: float, tau: float):
    looks = np.asarray(looks_t, dtype=np.int64)
    t_null = _simulate(looks, p0, p0, sims, 11)
    t_alt = _simulate(looks, p0, p1, sims, 12)

    def fixed(t, pa, pb):
        return Trial(t.looks[-1:], t.z[:, -1:], t.diff[:, -1:], t.se[:, -1:], pa, pb)

    rows = []
    fx0, fx1 = fixed(t_null, p0, p0), fixed(t_alt, p0, p1)
    pairs = [("fixed horizon", score(fx0, first_crossing(fx0.z, naive_bounds(1, alpha)), "f"),
              score(fx1, first_crossing(fx1.z, naive_bounds(1, alpha)), "f"))]
    for name, b in boundaries(len(looks), alpha).items():
        pairs.append((name, score(t_null, first_crossing(t_null.z, b), name),
                      score(t_alt, first_crossing(t_alt.z, b), name)))
    pairs.append((f"mSPRT (tau={tau:.4f})",
                  score(t_null, msprt_crossing(t_null, tau, alpha), "m"),
                  score(t_alt, msprt_crossing(t_alt, tau, alpha), "m")))
    for name, o0, o1 in pairs:
        rows.append({
            "rule": name,
            "false-positive rate": o0.reject_rate,
            "power": o1.reject_rate,
            "visitors/arm used (effect present)": o1.expected_n,
            "visitors/arm used (nothing there)": o0.expected_n,
            "lift it would report": o1.est_at_stop,
            "overstated by": o1.est_bias,
            "95% CI coverage": o1.ci_coverage,
        })
    return pd.DataFrame(rows).set_index("rule")


df = run(tuple(int(x) for x in looks), p0, p1, sims, alpha, tau)
naive_fpr = float(df.loc["naive peek", "false-positive rate"])

c1, c2, c3 = st.columns(3)
c1.metric("Uncorrected peek, false-positive rate", f"{naive_fpr:.3f}",
          f"{naive_fpr / alpha:.1f}x what the p-value claims")
c2.metric("Pocock, visitors per arm used", f"{df.loc['Pocock', 'visitors/arm used (effect present)']:,.0f}",
          f"{df.loc['Pocock', 'visitors/arm used (effect present)'] / n_max - 1:.0%} vs running to the end")
c3.metric("Lift a naive peek would report", f"{df.loc['naive peek', 'lift it would report']:.4f}",
          f"{df.loc['naive peek', 'overstated by']:+.0%} vs a true lift of {p1 - p0:.4f}")

st.subheader("Is it valid, and what does it cost?")
st.dataframe(
    df[["false-positive rate", "power", "visitors/arm used (effect present)",
        "visitors/arm used (nothing there)"]].style.format({
            "false-positive rate": "{:.3f}", "power": "{:.3f}",
            "visitors/arm used (effect present)": "{:,.0f}",
            "visitors/arm used (nothing there)": "{:,.0f}"}),
    width="stretch",
)

st.subheader("...and what would it tell you the effect was?")
st.caption(
    "A boundary controls how often you are wrong about the SIGN. Nothing here controls "
    "the SIZE - every rule that can stop early reports a lift larger than the truth, and "
    "the weaker the real effect, the worse it gets. Drag the lift slider down to watch it."
)
st.dataframe(
    df[["lift it would report", "overstated by", "95% CI coverage"]].style.format({
        "lift it would report": "{:.5f}", "overstated by": "{:+.1%}",
        "95% CI coverage": "{:.3f}"}),
    width="stretch",
)

st.subheader("The boundaries this schedule needs")
b = boundaries(k, alpha)
chart = pd.DataFrame({name: bounds for name, bounds in b.items()},
                     index=pd.Index(range(1, k + 1), name="look"))
st.line_chart(chart.clip(upper=6.0), height=280)
st.caption(
    f"Clipped at |z| = 6 so the flat boundaries stay readable; O'Brien-Fleming starts at "
    f"{b[chr(79) + chr(39) + 'Brien-Fleming'][0]:.1f} and ends at "
    f"{b[chr(79) + chr(39) + 'Brien-Fleming'][-1]:.2f}. Solved for {k} looks and valid for "
    f"{k} looks -- add a look and the guarantee is gone."
)
