"""Novelty decay explorer: pick a world, read the plot, then read what the plot cannot say."""

from __future__ import annotations

import matplotlib.pyplot as plt
import novelty as nv
import numpy as np
import streamlit as st

st.set_page_config(page_title="Novelty decay", layout="wide")
st.title("The lift vanished after two weeks")
st.caption(
    "A measured lift is a lift AT AN AGE. This tool shows the three numbers a decaying "
    "effect gives you, the number the launch decision actually needs, and the two worlds "
    "that produce an identical plot with answers five times apart."
)

with st.sidebar:
    st.header("The effect")
    tau0 = st.slider("first-day effect tau_0", 0.0, 0.30, 0.10, 0.01, key="tau0")
    tau_inf = st.slider("long-run effect tau_inf", 0.0, 0.20, 0.02, 0.01, key="tau_inf")
    lam = st.slider("decay constant lambda (days)", 1.0, 28.0, 7.0, 1.0, key="lam")
    st.header("The test")
    weeks = st.slider("window length (weeks)", 1, 16, 4, key="weeks")
    enroll = st.radio("enrollment", ["uniform", "bigbang"],
                      format_func=lambda s: "continuous (daily cohorts)" if s == "uniform" else "big bang (all day 0)",
                      key="enroll")
    n_users = st.select_slider("users", [2800, 5600, 8400, 16800], value=8400, key="n_users")
    st.header("The world")
    world = st.radio("what is really happening",
                     ["real novelty decay", "cohort mix shift (no decay)", "survivorship (no decay)"],
                     key="world")
    seed = st.number_input("seed", 0, 9999, 7, key="seed")

T = weeks * 7
n_users = (n_users // T) * T if T > 0 else n_users
rng = np.random.default_rng(int(seed))

if world == "real novelty decay":
    d = nv.panel_decay(rng, n_users=n_users, T=T, enroll=enroll, tau0=tau0, tau_inf=tau_inf, lam=lam)
elif world == "cohort mix shift (no decay)":
    d = nv.panel_mixshift(rng, n_users=n_users, T=T, tau0=tau0, tau_inf=tau_inf, lam=lam)
else:
    d = nv.panel_survivor(rng, n_users=n_users, T=T, tau=tau_inf)

curve = nv.lift_by_tenure(d, min_arm=5)
pooled, pooled_se = nv.pooled_lift(d)
truth = d["truth_long_run"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("pooled lift (the dashboard)", f"{pooled:.4f}", f"{pooled / truth:.1f}x the truth" if truth else None)
if curve["tenure"].size:
    late = curve["lift"][curve["tenure"] >= curve["tenure"].max() - 6].mean()
    c2.metric("final-week lift", f"{late:.4f}", f"{late / truth:.1f}x the truth" if truth else None)
c3.metric("TRUE long-run effect", f"{truth:.4f}", "what the launch decides on")
c4.metric("novelty weight A", f"{nv.attenuation(T, lam, enroll):.4f}",
          "property of the calendar, not the data")

left, right = st.columns([3, 2])

with left:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=130)
    ax.errorbar(curve["tenure"], curve["lift"], yerr=1.96 * curve["se"], fmt="o-",
                color="#16222e", ms=3, lw=1.3, elinewidth=0.7, capsize=1.5, label="measured")
    ax.axhline(truth, color="#1f7a5c", ls="--", lw=1.6, label=f"true long-run {truth:.3f}")
    ax.axhline(pooled, color="#c98a1a", ls=":", lw=1.3, label=f"pooled {pooled:.3f}")
    ax.set_xlabel("user tenure (days)")
    ax.set_ylabel("lift")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    st.pyplot(fig)

with right:
    st.subheader("What the plot cannot tell you")
    g = nv.guard_early_vs_late(d, early=(0, 7), late=(max(T - 7, 7), T))
    st.write(
        f"**Early-vs-late guard**: week 1 reads {g['early']:.4f}, the last week reads "
        f"{g['late']:.4f}, z = {g['z']:.2f} (p = {g['p']:.4f}). "
        + ("It fires." if g["p"] < 0.05 else "It does not fire.")
    )
    st.caption(
        "This guard detects only that the lift MOVED. It fires identically on real decay and "
        "on a cohort mix shift, and it says nothing about where the effect settles."
    )
    if world != "survivorship (no decay)":
        sl = nv.within_cohort_slope(d)
        st.write(
            f"**Within-cohort tenure slope**: {sl['slope']:+.5f} per day, z = {sl['z']:.2f} "
            f"over {sl['cells']} cohort-tenure cells."
        )
        st.caption(
            "This is the separator, and it needs a COLUMN rather than a test: real decay bends "
            "every cohort's own curve, a mix shift bends none of them. A dashboard without the "
            "enrollment date cannot run it at any sample size."
        )
    else:
        alive = d["alive"]
        at = alive[d["w"] == 1][:, -1].mean()
        ac = alive[d["w"] == 0][:, -1].mean()
        st.write(f"**Retention at day {T - 1}**: treated {at:.4f}, control {ac:.4f} "
                 f"(difference {at - ac:+.4f}).")
        st.caption(
            "The obvious guard is nearly inert here. The treatment sheds low-value users and "
            "keeps high-value ones, so the two flows almost cancel in the headcount while doing "
            "all the damage to the mean. Counting users cannot see a change in who they are."
        )

st.divider()
st.subheader("What the fix costs")
f1, f2 = st.columns(2)
with f1:
    share = st.slider("long-term holdout share", 0.01, 0.50, 0.05, 0.01, key="share")
    ratio = nv.mde_ratio(share)
    st.write(f"A {share:.0%} holdout carries **{ratio:.2f}x** the MDE of the 50/50 test that "
             f"shipped the feature - {ratio ** 2:.1f}x the traffic-time to match it.")
    if truth > 0 and tau0 > 0:
        st.write(f"It must also resolve a **smaller** quantity (tau_inf/tau_0 = {truth / tau0:.2f}), "
                 f"so reaching the original power takes **{(ratio * tau0 / truth) ** 2:.0f}x** the user-days.")
with f2:
    gap = st.slider("days between launches into that holdout", 0, 30, 7, key="gap")
    a = nv.holdout_attribution(30 + np.arange(6) * gap, 240, 1.0 * np.sqrt(1 / 10000 + 1 / 190000))
    ded = 1.0 * np.sqrt(2 / 100000)
    if a is None:
        st.error("Six launches on the same day: the per-launch effects are NOT IDENTIFIED at any "
                 "sample size. The design matrix loses rank - this is not a precision problem.")
    else:
        st.write(f"Six launches {gap} days apart: SE per launch **{np.mean(a['se_per_launch']):.5f}** "
                 f"= {np.mean(a['se_per_launch']) / ded:.2f}x a dedicated test, while the SE of the "
                 f"total is {a['se_total']:.5f}.")
        st.caption("A holdout is a joint cost. The release calendar, not the statistics, decides "
                   "whether one launch's share is recoverable - and it is computable before any "
                   "data exists.")

st.divider()
st.caption(
    "Every number here is computed live. The measurement run behind the README is `evidence.py`; "
    "the arithmetic is in `novelty.py` and is covered by `test_novelty.py`."
)
