"""Streamlit front end: build a design, run both tests, watch the two verdicts separate.

Every stored number comes from results.json - the app never recomputes a verdict the study
already made. What it does compute live is the interactive design the user builds with the
sliders, which the study does not cover.

Verified headless with `streamlit.testing.v1.AppTest`, addressing widgets BY KEY rather than by
index, because an index-addressed widget silently moves when a layout changes and an
out-of-range set_value passes vacuously.
"""

from __future__ import annotations

import json
import math

import nonparam as N
import numpy as np
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Just use Mann-Whitney?", layout="wide")

R = json.load(open("results.json"))
C = R["config"]

st.title("\"Just use Mann-Whitney\"")
st.markdown(
    "It is not a robust t-test. **A t-test asks about `mu_B - mu_A`; Mann-Whitney asks about "
    "`P(B > A)`.** Under a pure location shift those agree in sign. Off that model they can be "
    "both significant and point opposite ways."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Build a disagreement", "Power (where it is fine)", "Unequal spread", "Ties",
])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Most cases get a little worse, a few get a lot better")
    st.markdown(
        "Control is `N(0, 0.5^2)`. Treated is a two-component mixture. Both quantities below "
        "are **closed form** - move the sliders and watch the mean and the probability of "
        "superiority land on opposite sides of their nulls."
    )

    left, right = st.columns([1, 2])
    with left:
        w_low = st.slider("share of treated cases that get worse", 0.50, 0.95, 0.70, 0.01,
                          key="w_low")
        mu_low = st.slider("how much worse they get", -3.0, -0.1, -1.0, 0.1, key="mu_low")
        mu_high = st.slider("how much better the rest get", 0.5, 8.0, 5.0, 0.1, key="mu_high")
        sd = st.slider("within-component SD", 0.2, 1.5, 0.5, 0.05, key="sd")
        n_per = st.select_slider("n per group", options=[10, 20, 50, 100, 200, 500],
                                 value=100, key="n_per")
        reps = st.select_slider("replicate studies", options=[500, 2000, 5000],
                                value=2000, key="reps")

    weights = (w_low, 1.0 - w_low)
    means = (mu_low, mu_high)
    spread = math.sqrt(2.0) * sd
    mean_diff = sum(w * m for w, m in zip(weights, means))
    psup = sum(w * float(stats.norm.cdf(m / spread)) for w, m in zip(weights, means))

    with right:
        c1, c2 = st.columns(2)
        c1.metric("true mean difference", f"{mean_diff:+.3f}",
                  "t-test target: treated HIGHER" if mean_diff > 0
                  else "t-test target: treated LOWER")
        c2.metric("true P(treated > control)", f"{psup:.4f}",
                  "MW target: treated HIGHER" if psup > 0.5
                  else "MW target: treated LOWER")

        if (mean_diff > 0) != (psup > 0.5):
            st.error(
                "**The two targets disagree in SIGN.** Every honest test of the mean and every "
                "honest test of stochastic dominance will contradict each other here, and the "
                "larger the sample the more reliably they will."
            )
        else:
            st.success(
                "The two targets agree in sign, so the swap is harmless on this design. Push "
                "the minority further out, or raise the share that get worse, to break it."
            )

        x = np.linspace(min(means) - 4 * sd, max(means) + 4 * sd, 900)
        control = stats.norm.pdf(x, 0.0, sd)
        treated = sum(w * stats.norm.pdf(x, m, sd) for w, m in zip(weights, means))
        st.line_chart({"control": control, "treated": treated}, height=230)

    if st.button("Run both tests on this design", key="run_disagreement"):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((reps, n_per)) * sd
        which = rng.random((reps, n_per)) < weights[0]
        b = np.where(which, means[0], means[1]) + rng.standard_normal((reps, n_per)) * sd
        pt, dt = N.welch(a, b)
        pm, dm = N.mannwhitney(a, b)
        t_sig, m_sig = pt < N.ALPHA, pm < N.ALPHA
        opposite = float(np.mean(t_sig & m_sig & (dt * dm < 0)))
        lo, hi = N.wilson(opposite, reps)
        st.metric(f"both significant, opposite directions ({reps:,} studies)",
                  f"{opposite:.1%}", f"99% CI [{lo:.3f}, {hi:.3f}]")
        st.caption(
            f"t-test significant and pointing up {float(np.mean(t_sig & (dt > 0))):.1%} of the "
            f"time; Mann-Whitney significant and pointing down "
            f"{float(np.mean(m_sig & (dm < 0))):.1%}."
        )

    st.divider()
    st.caption(
        f"The study's own design: mean difference {C['mix_mean']:+.2f}, "
        f"P(B>A) {C['mix_prob_superiority']:.4f}, Cohen's d {C['mix_cohens_d']:.3f}."
    )
    st.dataframe(
        [{"n per group": int(r["n"]), "t sig UP": r["t_up"], "MW sig DOWN": r["mw_down"],
          "both, opposite": r["opposite"]} for r in R["disagreement"]],
        hide_index=True, width="stretch",
    )

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Under a pure location shift, the rank test is a good bet")
    st.markdown(
        "Here the two hypotheses coincide, so power is a fair comparison - but only in cells "
        "where **both** tests control Type I. An inflated test detects more of everything."
    )
    st.dataframe(
        [{"shape": e["shape"], "predicted ARE": round(e["are"], 4),
          "measured n_t / n_MW": (round(e["measured"], 4) if e["measured"] is not None else None),
          "comparable cells": e["usable"]} for e in R["efficiency"]],
        hide_index=True, width="stretch",
    )
    st.caption(
        "Above 1 means Mann-Whitney needed fewer observations. The measured column sits below "
        "the predicted one because ARE is an asymptotic small-effect limit and these are "
        f"finite-n readings at d = {C['location_d']}, n = {C['location_ref_n']}."
    )

    shape = st.selectbox("population", list(N.SHAPES), key="power_shape")
    rows = [r for r in R["location"] if r["shape"] == shape]
    st.dataframe(
        [{"n": r["n"], "t null": r["t_null"], "MW null": r["mw_null"],
          "t null verdict": r["t_null_verdict"], "MW null verdict": r["mw_null_verdict"],
          "t power": r["t_power"], "MW power": r["mw_power"],
          "comparable": "yes" if r["comparable"] else "NO"} for r in rows],
        hide_index=True, width="stretch",
    )

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Same centre, different spread - and the Mann-Whitney null is exactly true")
    st.markdown(
        "Two symmetric populations with the same centre give `P(B > A) = 0.5` exactly, so both "
        "nulls hold. Any departure below is the variance formula, which assumes the two "
        "distributions are *identical* rather than merely balanced."
    )
    only_broken = st.checkbox("show only the cells where a test misses 5%", value=False,
                              key="only_broken")
    rows = R["unequal"]
    if only_broken:
        rows = [r for r in rows if r["mw_verdict"] != "ok" or r["welch_verdict"] != "ok"]
    st.dataframe(
        [{"shape": r["shape"], "SD ratio": r["sd_ratio"], "n1": r["n1"], "n2": r["n2"],
          "Welch": r["welch"], "Welch verdict": r["welch_verdict"],
          "Mann-Whitney": r["mw"], "MW verdict": r["mw_verdict"]} for r in rows],
        hide_index=True, width="stretch",
    )
    mw_bad = sum(1 for r in R["unequal"] if r["mw_verdict"] != "ok")
    w_bad = sum(1 for r in R["unequal"] if r["welch_verdict"] != "ok")
    c1, c2 = st.columns(2)
    c1.metric("Mann-Whitney off nominal", f"{mw_bad} of {len(R['unequal'])}")
    c2.metric("Welch off nominal", f"{w_bad} of {len(R['unequal'])}")
    st.caption(
        "Welch is not clean either - it is mildly conservative on the contaminated population. "
        "The difference is the size: a fifth conservative against an order of magnitude."
    )

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("A rank test on a rating scale")
    st.markdown(
        "Ties only ever shrink `Var(U)`, so the no-ties formula divides by a standard error "
        "that is too large and the test goes quiet. SciPy corrects for it; many hand-rolled "
        "rank tests do not."
    )
    st.dataframe(
        [{"scale points": int(r["levels"]), "SD corrected / uncorrected": round(r["sd_ratio"], 4),
          "null, corrected": r["null_corrected"], "null, uncorrected": r["null_uncorrected"],
          "power, corrected": r["power_corrected"],
          "power, uncorrected": r["power_uncorrected"],
          "uncorrected verdict": r["null_uncorrected_verdict"]} for r in R["ties"]],
        hide_index=True, width="stretch",
    )
    worst = R["ties"][0]
    st.metric(f"power lost on a {int(worst['levels'])}-point scale by dropping the correction",
              f"{worst['power_corrected'] - worst['power_uncorrected']:.3f}")

st.divider()
st.caption(
    f"alpha = {C['alpha']}. A rate is called broken only when its whole 99% Wilson interval "
    f"clears [{C['band'][0]}, {C['band'][1]}]. Stored study: {C['null_reps']:,} replicates for "
    f"null cells, {C['study_reps']:,} for power. Regenerate with `python evidence.py`."
)
