"""Streamlit front end: pick a guardrail suite, see whether it could ever fire.

The point of the app is the one number a checklist never carries -- the probability that
this guardrail, at this sample size, on this decision day, notices harm that is really
there.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import guardrails as G
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Guardrail Metric", layout="wide")

st.title("We hit the KPI and broke the business")
st.caption(
    "A guardrail is not a second metric. It is a constraint, and a constraint has a "
    "threshold, a maturity and a power. Set the experiment below and read the power the "
    "checklist never shows you."
)

with st.sidebar:
    st.header("The experiment")
    intensity = st.slider("Lever intensity", 0.0, 1.0, 1.0, 0.05,
                          help="0 = a harmless change. 1 = the most aggressive version of it.")
    day = st.slider("Decision day", 3, 180, 14,
                    help="Users enrol uniformly, so this sets both sample size and maturity.")
    scale = st.slider("Sample size multiple", 0.25, 8.0, 1.0, 0.25,
                      help="1.0 = exactly the n that powers the WIN to 80%.")
    alpha = st.select_slider("Guardrail alpha", [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30], 0.05)
    st.divider()
    suite_choice = st.radio("Suite", ["What most teams run", "Everything computable", "Custom"])

base_n = G.n_for_power(0.80, 1.0)
n = max(int(base_n / 14 * day * scale), 50)

if suite_choice == "What most teams run":
    suite = list(G.DASHBOARD_SUITE)
elif suite_choice == "Everything computable":
    suite = [g.name for g in G.GUARDRAILS]
else:
    suite = st.sidebar.multiselect("Guardrails", [g.name for g in G.GUARDRAILS],
                                   default=list(G.DASHBOARD_SUITE))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reported lift", f"+{G.primary_lift(intensity)*100:.1f}%")
c2.metric("180-day retention rate", f"{G.quality_change(intensity)*100:.1f}%")
c3.metric("Retained users per 1,000", f"{G.value_change(intensity)*100:.2f}%")
c4.metric("Power on the win", f"{G.primary_power(intensity, n, 0.05):.2f}")

if intensity > 0:
    st.error(
        f"At intensity {intensity:.2f} this change genuinely harms the business: the 180-day "
        f"retention rate falls {abs(G.quality_change(intensity))*100:.1f}%. Everything below is "
        f"about whether anything would notice."
    )
else:
    st.success("At intensity 0 the change is harmless. Everything that fires below is a false alarm.")

st.subheader("Can each guardrail actually fire?")
rows = []
for name in suite:
    g = G.GUARDRAIL_BY_NAME[name]
    n_c, _ = G.guardrail_n(g, intensity, n, day)
    p = G.analytic_power(g, intensity, n, day, alpha)
    need = G.n_for_power(0.80, intensity, alpha, g, day) if intensity > 0 else None
    rows.append({
        "guardrail": name,
        "what it watches": g.blurb,
        "denominator": f"{n_c:,.0f}",
        "matures after": f"{g.maturity_days}d",
        "observable": f"{G.observable_fraction(day, g.maturity_days):.0%}",
        "power": "cannot be computed" if np.isnan(p) else f"{p:.3f}",
        "n for 80%": "unreachable" if need is None else f"{need:,}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

dead = [r["guardrail"] for r in rows if r["power"] == "cannot be computed"]
weak = [r["guardrail"] for r in rows
        if r["power"] != "cannot be computed" and float(r["power"]) < 0.30]
if dead:
    st.warning(f"No denominator on day {day}: **{', '.join(dead)}**. These are on the checklist "
               f"and cannot be evaluated at all.")
if weak:
    st.warning(f"Below 0.30 power: **{', '.join(weak)}**. A tick from one of these is close to "
               f"no information.")

st.subheader("One test instead of many")
if suite:
    reps = 8000
    rng = np.random.default_rng(7)
    z0 = G.simulate_experiment(0.0, n, day, reps, rng)
    z1 = G.simulate_experiment(intensity, n, day, reps, rng)
    live = [s for s in suite if G.observable_fraction(day, G.GUARDRAIL_BY_NAME[s].maturity_days) > 0]
    w = G.sensitivity_weights(live, max(intensity, 0.05), n, day)

    split_false = float(np.mean(G.any_fires(z0, live, alpha)))
    split_detect = float(np.mean(G.any_fires(z1, live, alpha)))
    crit = float(np.quantile(G.composite_z(z0, live, w), 1 - split_false)) if live else 0.0
    pooled_detect = float(np.mean(G.composite_z(z1, live, w) > crit)) if live else 0.0

    a, b = st.columns(2)
    a.metric(f"{len(live)} separate tests: blocks a harmless change", f"{split_false:.1%}")
    a.metric("...and detects this harm", f"{split_detect:.1%}")
    b.metric("One pooled index at the SAME false-block rate", f"{split_false:.1%}")
    b.metric("...and detects this harm", f"{pooled_detect:.1%}",
             delta=f"{(pooled_detect - split_detect)*100:+.1f} pts")
    st.caption(
        "Both rows use exactly the same metrics and the same data. The only difference is "
        "whether they are tested one at a time or combined into a single directional index."
    )

    fig, ax = plt.subplots(figsize=(9, 3.1))
    for name, color in [("split", "#c0392b"), ("pooled", "#4a7c8c")]:
        vals = (G.composite_z(z1, live, w) if name == "pooled"
                else np.nanmax(np.vstack([np.nan_to_num(z1[s], nan=-9) for s in live]), axis=0))
        ax.hist(vals, bins=60, alpha=0.55, color=color,
                label="largest single z" if name == "split" else "pooled index")
    ax.axvline(crit, color="#4b7f52", lw=1.6, ls="--", label="threshold")
    ax.set_xlabel("evidence of harm (z)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

st.divider()
st.caption("Day 160 of Phoebe's FDE portfolio. Run `python evidence.py` for the full ten-section "
           "study, or open demo.ipynb.")
