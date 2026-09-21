"""Pick a distribution shape, drag d, and watch the other effect-size metrics disagree.

Everything here calls effectsize.py. The app never recomputes a verdict with its own rule - that
was the defect `normality-test-trap` shipped, where three artifacts had each grown a private copy
of one comparison and disagreed about four cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from effectsize import (
    ALPHA,
    METRICS,
    SHAPES,
    analytic_prob_superiority,
    contaminate,
    dichotomisation_cost,
    dichotomisation_is_costly,
    draw_pair,
    metrics,
    n_for_significance,
    smallest_significant_d,
    split_beats_t,
    true_prob_superiority,
)

st.set_page_config(page_title="Cohen's d is not the effect", layout="wide")

st.title("Cohen's d is not the effect")
st.caption(
    "Every population below is standardised to mean 0 and SD 1 by its ANALYTIC moments, then "
    "shifted by exactly d - so the population Cohen's d is identical whichever shape you pick. "
    "Anything that changes when you change the shape is the metrics answering different "
    "questions, not the effect getting bigger."
)

with st.sidebar:
    st.header("The comparison")
    shape = st.selectbox("Distribution shape", SHAPES, index=0, key="shape")
    d = st.slider("True Cohen's d", 0.0, 1.2, 0.5, 0.05, key="d")
    n = st.select_slider("Sample size per group", [10, 20, 30, 50, 100, 200, 500], 50, key="n")
    reps = st.select_slider("Replicates", [2000, 5000, 10000, 20000], 5000, key="reps")
    st.divider()
    st.caption(
        "The reference truth uses 4,000,000 draws rather than the 20,000,000 in `evidence.py`, "
        "so the app stays responsive. Its Monte Carlo error is about 0.0006."
    )

ref = 4_000_000
ps_true, ps_err = true_prob_superiority(shape, d, ref, seed=0)
ps_normal = analytic_prob_superiority(d)

rng = np.random.default_rng(0)
a, b = draw_pair(shape, d, n, n, reps, rng)
m = {k: float(v.mean()) for k, v in metrics(a, b).items()}

c1, c2, c3, c4 = st.columns(4)
c1.metric("True Cohen's d", f"{d:.2f}", help="Held exactly, by construction, for every shape.")
c2.metric("True P(treated wins)", f"{ps_true:.4f}", help=f"+/- {ps_err:.5f} at 99%")
c3.metric("What a normal curve would say", f"{ps_normal:.4f}",
          delta=f"{ps_true - ps_normal:+.4f} vs this shape")
c4.metric("n for 80% power", f"{n_for_significance(d):,}" if d > 0 else "-")

if d == 0:
    st.info(
        "No effect at all, so every metric should sit at its null value: d = 0, P(X>Y) = 0.5, "
        "Cliff's delta = 0. Move d off zero to make the shapes disagree."
    )
elif abs(ps_true - ps_normal) > 0.01:
    st.warning(
        f"**Same d, different answer.** On {shape} data a true d of {d:.2f} means the treated "
        f"value beats the control {ps_true:.1%} of the time. The textbook gloss - which assumes "
        f"a normal curve - would have told you {ps_normal:.1%}. Nothing about the effect changed; "
        f"the denominator d divides by means something different in this shape."
    )
else:
    st.success(
        f"On {shape} data the normal gloss is close: {ps_true:.1%} measured against "
        f"{ps_normal:.1%} assumed. Light-tailed symmetric shapes behave like the textbook. "
        f"Try lognormal or contaminated."
    )

# ------------------------------------------------------------------ metrics
st.subheader(f"Every summary of the same comparison, at n = {n}")
st.dataframe(
    pd.DataFrame({
        "metric": [k.replace("_", " ") for k in METRICS],
        "mean estimate": [m[k] for k in METRICS],
    }).set_index("metric"),
    use_container_width=True, height=290,
)
st.caption(
    f"Cohen's d estimates {m['cohens_d']:.3f} against a truth of {d:.2f} "
    f"({m['cohens_d'] - d:+.3f}); Hedges' g corrects it to {m['hedges_g']:.3f} "
    f"({m['hedges_g'] - d:+.3f}). P(X>Y) estimates {m['prob_superiority']:.4f} against a truth of "
    f"{ps_true:.4f} ({m['prob_superiority'] - ps_true:+.4f}) - the rank summary needs no "
    f"small-sample correction, because it never divides by an estimated SD."
)

# ------------------------------------------------------------------ robustness
st.subheader("One contaminated value in a hundred")
rob = contaminate(shape, d, max(n, 100), min(reps, 10000), seed=31)
st.bar_chart(
    pd.DataFrame({"% change": [r.pct_change for r in rob]},
                 index=[r.metric.replace("_", " ") for r in rob]),
    height=260, color="#c8562b",
)
worst = max(rob, key=lambda r: abs(r.pct_change))
st.caption(
    f"1% of the treated values replaced with +10 SD, pointing the same way as the effect. "
    f"Largest move: **{worst.metric.replace('_', ' ')}** at {worst.pct_change:+.1f}%. "
    f"Glass's delta is usually the one that breaks - it divides by the CONTROL group's SD, which "
    f"the contamination never touches, so the numerator grows and the denominator does not."
)

# ------------------------------------------------------------------ magnitude
st.subheader("What a significant result still means at this sample size")
sd = smallest_significant_d(n)
a1, a2 = st.columns(2)
a1.metric("Smallest d this n can see", f"{sd:.4f}",
          help="The effect at which the study is a coin flip - 50% power.")
a2.metric("...which is a win rate of", f"{analytic_prob_superiority(sd):.2%}")
st.caption(
    "Any effect above zero is detected sometimes, so there is no smallest detectable effect - "
    "only a smallest effect the study is not guessing about. Push n to 500 and watch what "
    "'statistically significant' is capable of meaning."
)

# ------------------------------------------------------------------ median split
st.subheader("And what a median split would cost here")
if st.checkbox("Run it (also runs a null cell, so the comparison is legitimate)"):
    r = dichotomisation_cost(shape, max(d, 0.05), n, min(reps, 10000), seed=20)
    b1, b2, b3 = st.columns(3)
    b1.metric("Power, t-test", f"{r.power_t:.4f}")
    b2.metric("Power, median split", f"{r.power_chi2:.4f}")
    b3.metric("Split has the power of n =", f"{r.equivalent_n:,}" if r.equivalent_n > 0 else "-")
    if not r.comparable:
        st.warning(
            f"**Not comparable, so no verdict.** Under a true null the t-test rejects at "
            f"{r.null_t:.4f} and the median split at {r.null_chi2:.4f} against a nominal "
            f"{ALPHA}. A test that over-rejects detects more of everything, so its power lead "
            f"would be the inflation restated rather than a finding. The split is "
            f"anticonservative at small n on every shape tested."
        )
    elif dichotomisation_is_costly(r):
        st.info(
            f"Both tests hold a nominal {ALPHA}, so the comparison is legitimate - and the "
            f"t-test wins. The split has the power of n = {r.equivalent_n}, so about "
            f"{r.n_wasted_share:.0%} of the sample was thrown away. This is the textbook case, "
            f"and it is a claim about NORMAL data."
        )
    elif split_beats_t(r):
        st.info(
            f"Both tests hold nominal, and **the median split wins**. It has the power of "
            f"n = {r.equivalent_n} - about {-r.n_wasted_share:.0%} more data than you collected. "
            f"On skewed and heavy-tailed data the split is a rank method and the t-test is not, "
            f"so 'never dichotomise' is simply false here."
        )
    else:
        st.info("Both hold nominal and neither power interval clears the other - no verdict.")

st.divider()
st.caption(
    "Run `python evidence.py` for the full grid and `pytest` for the assertions behind every "
    "claim above. Sibling builds: t-test-variants, normality-test-trap, assumption-pretest-cost, "
    "p-value-dance - all four interrogate the test. This one interrogates the estimate."
)
