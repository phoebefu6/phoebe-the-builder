"""Drag the design around and watch the p-value move.

Everything here calls pvalue.py. The app never recomputes a verdict with its own rule - that was
the defect `normality-test-trap` shipped, where the chart, the notebook and the evidence file had
each grown a copy of the comparison and disagreed about four cells.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
from pvalue import (
    ALPHA,
    analytic_power,
    ppv_study,
    replication_study,
    required_n,
    simulate,
    wilson,
)

st.set_page_config(page_title="A p-value is a random variable", layout="wide")

st.title("A p-value is a random variable")
st.caption(
    "Replay the SAME study and watch p move. Student's two-sample t on equal-n, equal-variance, "
    "normal data - the one configuration verified exact in the sibling build `t-test-variants`, "
    "so nothing you see below is a violated assumption."
)

with st.sidebar:
    st.header("The study")
    d = st.slider("True effect (Cohen's d)", 0.0, 1.2, 0.35, 0.05, key="d",
                  help="0.0 makes the null exactly true. 0.2 / 0.5 / 0.8 are the conventional "
                       "small / medium / large.")
    n = st.select_slider("Sample size per group", [5, 10, 20, 30, 50, 75, 100, 200, 500], 30,
                         key="n")
    reps = st.select_slider("Replays", [5000, 10000, 20000, 40000], 20000, key="reps")
    st.divider()
    st.header("The wider world")
    prior = st.slider("Share of hypotheses that are actually true", 0.02, 0.90, 0.20, 0.02,
                      key="prior",
                      help="Used only by the last panel. It is the thing alpha never sees.")

p, d_hat = simulate(d, n, reps, seed=0)
sig = p < ALPHA
power = float(sig.mean())
exact = analytic_power(d, n) if d > 0 else ALPHA
lo, hi = wilson(power, reps)
q = {k: float(np.percentile(p, k)) for k in (5, 25, 50, 75, 95)}
straddles = q[5] < ALPHA < q[95]

# ------------------------------------------------------------------ the top line
c1, c2, c3, c4 = st.columns(4)
c1.metric("Power", f"{power:.3f}", help=f"analytic {exact:.4f}; 99% CI [{lo:.4f}, {hi:.4f}]")
c2.metric("Median p", f"{q[50]:.4g}")
c3.metric("Central 90% of p",
          f"{q[5]:.2g} to {q[95]:.2g}",
          help="Nine reruns in ten land in here. Nothing changes between them but the sample.")
c4.metric("Orders of magnitude",
          f"{math.log10(q[95]) - math.log10(max(q[5], 1e-300)):.1f}")

if not (lo <= exact <= hi):
    st.error(
        f"Calibration failed: measured power {power:.4f} (99% CI [{lo:.4f}, {hi:.4f}]) does not "
        f"contain the analytic {exact:.4f}. Raise the replay count before reading anything below."
    )

if d == 0:
    st.info(
        "The null is exactly true, so every 'significant' replay is a false positive - and there "
        f"are {sig.sum():,} of them, {power:.1%}, which is alpha doing precisely its job. Note the "
        "spread: about 1.3 orders of magnitude, the NARROWEST anywhere in this app. The p-value "
        "dances least when there is nothing to find."
    )
elif straddles:
    st.warning(
        f"**The central 90% straddles 0.05.** An identical rerun of this study returns the "
        f"opposite verdict routinely - it comes back non-significant {1 - power:.0%} of the time. "
        f"Note what that is, though: `p95 > 0.05` is the same statement as `power < 0.95`, "
        f"exactly. The dance is not an extra hazard on top of low power. It is low power."
    )
else:
    st.success(
        f"**The dance has stopped where it counts.** Even the worst of the central 90% "
        f"({q[95]:.2g}) is below 0.05, so a rerun is a formality. The raw spread is now enormous "
        f"- and irrelevant, because all of it sits on one side of the line."
    )

# ------------------------------------------------------------------ 1. the distribution
st.subheader("Where the replays land")
edges = np.logspace(math.log10(max(p.min(), 1e-300)), 0, 60)
counts, _ = np.histogram(p, bins=edges)
st.bar_chart(
    pd.DataFrame({"replays": counts}, index=[f"{e:.1e}" for e in edges[:-1]]),
    height=240, color="#b8860b",
)
st.caption(
    f"Log-spaced bins. The 0.05 line falls at bin {int(np.searchsorted(edges, ALPHA))} of "
    f"{len(edges) - 1}. Percentiles: 5th {q[5]:.3g} | 25th {q[25]:.3g} | median {q[50]:.3g} | "
    f"75th {q[75]:.3g} | 95th {q[95]:.3g}."
)

# ------------------------------------------------------------------ 2. winner's curse
st.subheader("What the significant replays look like")
if d == 0 or sig.sum() == 0:
    st.write("No true effect, so there is no effect to exaggerate or get the sign of wrong.")
else:
    winners = d_hat[sig]
    type_m = float(np.mean(np.abs(winners)) / d)
    type_s = float((winners < 0).sum() / sig.sum())
    s_lo, s_hi = wilson(type_s, int(sig.sum()))
    a, b, c = st.columns(3)
    a.metric("Exaggeration (type M)", f"{type_m:.2f}x",
             help="Mean |observed d| among significant replays, over the true d.")
    b.metric("Wrong sign (type S)", f"{type_s:.2%}",
             help=f"99% CI [{s_lo:.4f}, {s_hi:.4f}]")
    c.metric("Median published effect", f"{float(np.median(winners)):.3f}",
             delta=f"{float(np.median(winners)) - d:+.3f} vs truth")
    st.caption(
        f"True effect {d:.2f}. This is the part that is **not** the power calculation restated: "
        f"a study can be honestly run, correctly analysed and significant, and still report "
        f"{type_m:.1f}x the real effect. For 80% power at this effect you would need "
        f"**n = {required_n(d) if d > 0 else 0} per group** against the {n} you have."
    )

# ------------------------------------------------------------------ 3. replication
st.subheader("You got p = 0.05. What does an identical rerun say?")
if d == 0:
    st.write("Set a non-zero effect to run the replication study.")
elif st.checkbox("Run it (draws until enough studies land in p in [0.045, 0.055])"):
    r = replication_study(d, n, 200000, seed=900)
    if r.hits < 100:
        st.write(f"Only {r.hits} studies landed in the window - too few to read.")
    else:
        a, b, c = st.columns(3)
        a.metric("Rerun significant again", f"{r.rep_sig_rate:.3f}",
                 help=f"99% CI [{r.rep_sig_ci[0]:.4f}, {r.rep_sig_ci[1]:.4f}], from {r.hits:,} hits")
        b.metric("Plain unconditional power", f"{r.power_analytic:.3f}")
        c.metric("Rerun comes back above 0.10", f"{r.rep_above_10pct:.1%}")
        if r.differs_from_power:
            st.warning("These differ - which would be news. Check the replay count.")
        else:
            st.info(
                "They are the same number. With the true effect held fixed, replicate p-values "
                "are independent, so observing 0.05 told you nothing about the rerun you did not "
                "already know. The alarming replication interval is the ordinary sampling "
                "distribution of p at this power - it was there before the first study ran."
            )

# ------------------------------------------------------------------ 4. what it is worth
st.subheader("What a significant result is worth, once most hypotheses are wrong")
if d == 0:
    st.write("Set a non-zero effect: the mixture needs a real alternative to mix in.")
else:
    v = ppv_study(prior, d, n, 120000, seed=700)
    a, b = st.columns(2)
    a.metric("Significant findings that were null all along", f"{v.false_share:.1%}",
             help=f"99% CI [{v.false_share_ci[0]:.4f}, {v.false_share_ci[1]:.4f}] "
                  f"from {v.n_sig:,} significant studies")
    b.metric("What people think that number is", f"{ALPHA:.0%}")
    if not v.matches_formula:
        st.warning("The simulation missed its own closed form - raise the study count.")
    st.caption(
        f"{int(prior * 100)}% of hypotheses true, power {v.power:.2f}. Alpha is the false-positive "
        f"rate among **true nulls**; it is not the error rate of the claims you publish, and the "
        f"gap between them is set by power and by the prior - neither of which appears anywhere "
        f"in the p-value."
    )

st.divider()
st.caption(
    "Run `python evidence.py` for the full grid and `pytest` for the assertions behind every "
    "claim above. Sibling builds: t-test-variants, normality-test-trap, assumption-pretest-cost."
)
