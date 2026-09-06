"""Streamlit front end: drive a DiD design and watch what it can and cannot see."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from did import (
    aggregate_att,
    event_dummies,
    fe_ols,
    group_time_att,
    heterogeneity_bound,
    make_panel,
    make_staggered,
    twfe,
    twfe_cell_weights,
)
from scipy import stats

st.set_page_config(page_title="diff-in-diff", page_icon="📐", layout="wide")

st.title("Parallel trends is an assumption, and the test for it has a power")
st.caption(
    "Day 167 · diff-in-diff · every panel below is simulated from a world whose true "
    "treatment effect is 1.00, so the bias on screen is a measurement."
)

TAB_A, TAB_B, TAB_C = st.tabs(
    ["① Common timing: what the pre-trends test can see", "② Staggered adoption: negative weights", "③ Levels or logs"]
)


# ---------------------------------------------------------------- tab A
with TAB_A:
    c1, c2, c3, c4 = st.columns(4)
    delta = c1.slider("Parallel-trends violation δ (per period)", 0.0, 0.30, 0.05, 0.01)
    n_arm = c2.select_slider("Units per arm", [50, 100, 200, 400, 800, 1600, 3200], value=200)
    n_pre = c3.slider("Pre-periods observed", 2, 20, 5)
    reps = c4.select_slider("Simulations", [200, 400, 800, 1500], value=400)

    T = n_pre + 1 + 6
    t0 = n_pre + 1
    ev = list(range(-n_pre, 0)) + list(range(0, 6))
    adopt = np.where(np.arange(2 * n_arm) < n_arm, float(t0), np.inf)
    cols = event_dummies(adopt, T, [e for e in ev if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]

    rng = np.random.default_rng(20260904)
    fired = 0
    est, cov, est_pass = [], [], []
    prog = st.progress(0.0, text="simulating…")
    for i in range(reps):
        Y, D, _, _ = make_panel(
            rng, n_treated=n_arm, n_control=n_arm, T=T, t0=t0, effect=1.0, diff_trend=delta
        )
        f = fe_ols(Y, [D], ["D"], vcov="cluster")
        b, se = float(f.beta[0]), float(f.se()[0])
        est.append(b)
        cov.append(abs(b - 1.0) <= stats.t.ppf(0.975, f.dof) * se)
        p = fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1]
        if p < 0.05:
            fired += 1
        else:
            est_pass.append(b)
        if i % 25 == 0:
            prog.progress((i + 1) / reps, text=f"simulating… {i + 1}/{reps}")
    prog.empty()
    est = np.array(est)
    gap = float(np.mean(np.arange(t0, T)) - np.mean(np.arange(0, t0)))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimate (true = 1.00)", f"{est.mean():.3f}", f"{est.mean() - 1:+.3f} bias")
    m2.metric("Predicted bias  δ × Δt", f"{delta * gap:.3f}", f"Δt = {gap:.1f}")
    m3.metric("Coverage of the 95% CI", f"{np.mean(cov):.3f}", "nominal 0.95", delta_color="off")
    m4.metric("Pre-trends test fires", f"{fired / reps:.3f}", f"{len(est_pass)} runs passed")

    if est_pass:
        bp = float(np.mean(est_pass)) - 1.0
        mc = float(np.std(est_pass, ddof=1) / np.sqrt(len(est_pass)))
        st.markdown(
            f"**Bias among the runs that PASSED the pre-trends test: `{bp:+.4f}`** "
            f"against `{est.mean() - 1:+.4f}` unconditional — a shift of `{bp - (est.mean() - 1):+.4f}` "
            f"on a Monte Carlo error of `{mc:.4f}`. Screening on the pre-trend does not "
            f"remove the bias, because the test reads noise in the leads while the bias "
            f"lives in the trend."
        )
    st.info(
        f"With {n_pre} pre-periods this design detects δ = {delta:.2f} about "
        f"{fired / reps:.0%} of the time. Raising units per arm raises that too, but a "
        f"linear violation accumulates over TIME — the pre-window is the axis with the "
        f"leverage, and it is usually already in the warehouse.",
        icon="📐",
    )

# ---------------------------------------------------------------- tab B
with TAB_B:
    c1, c2, c3 = st.columns(3)
    growth = c1.slider("Effect growth per period of exposure", 0.0, 1.5, 0.5, 0.05)
    g_late = c2.slider("Late cohort adopts at period", 8, 16, 10)
    n_never = c3.select_slider("Never-treated units", [0, 25, 50, 100], value=0)

    Y, D, adopt, tau = make_staggered(
        np.random.default_rng(1),
        [(4, 50), (g_late, 50)],
        T=20,
        n_never=n_never,
        growth=growth,
        sigma=0.0,
        unit_sd=0.0,
    )
    W = twfe_cell_weights(D)
    b = twfe(Y, D)
    truth = float(tau[D > 0].mean())
    w = W[D > 0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True mean effect on the treated", f"{truth:.3f}")
    m2.metric("Two-way FE estimate", f"{b:.3f}", f"{b - truth:+.3f}")
    m3.metric("Treated cells with negative weight", f"{(w < 0).mean():.1%}", f"{w[w < 0].sum():.3f} total")
    m4.metric("Heterogeneity tolerance ratio", f"{heterogeneity_bound(W, D, 1.0):.3f}", "× the ATT")

    if b < 0 < tau[D > 0].min():
        st.error(
            f"**Every true effect in this panel is positive** (min {tau[D > 0].min():.2f}, "
            f"max {tau[D > 0].max():.2f}) **and the estimate is {b:.3f}.** The negative weights sit on "
            f"the early cohort's periods after the late cohort adopts — where an "
            f"already-treated group is serving as the control.",
            icon="🚨",
        )
    elif not (tau[D > 0].min() <= b <= tau[D > 0].max()):
        st.warning(
            f"The estimate ({b:.3f}) lies outside the range of every individual true effect "
            f"in the data ({tau[D > 0].min():.2f} to {tau[D > 0].max():.2f}).",
            icon="⚠️",
        )

    if n_never > 0:
        cs = aggregate_att(group_time_att(Y, adopt), adopt, 20)
        st.success(
            f"**Not-yet-treated group-time ATT: `{cs:.4f}`** against a truth of `{truth:.4f}` "
            f"(TWFE: `{b:.4f}`). The whole correction is one restriction — never use an "
            f"already-treated unit as a control.",
            icon="✅",
        )
    else:
        st.info(
            "Add a never-treated cohort to compute the not-yet-treated estimator. With no "
            "clean comparison at the end of the panel, the last cohort's late periods "
            "genuinely cannot be estimated — which is information, and is what TWFE "
            "spends a negative weight to paper over.",
            icon="ℹ️",
        )

    st.subheader("Weight on each treated cell")
    wm = pd.DataFrame(np.where(D > 0, W, np.nan))
    wm.index = [f"unit {i}" for i in range(len(wm))]
    show = wm.iloc[list(range(0, 50, 12)) + list(range(50, 100, 12))]
    st.dataframe(
        show.style.format("{:+.5f}", na_rep="·").background_gradient(cmap="RdYlGn", axis=None),
        width="stretch",
    )

# ---------------------------------------------------------------- tab C
with TAB_C:
    c1, c2, c3 = st.columns(3)
    c_pre = c1.number_input("Control, before", value=100.0, min_value=1.0)
    c_post = c2.number_input("Control, after", value=120.0, min_value=1.0)
    t_pre = c3.number_input("Treated, before", value=200.0, min_value=1.0)
    obs = st.slider(
        "Treated, after (observed)",
        float(t_pre * 0.8),
        float(t_pre * 1.6),
        float(t_pre + (c_post - c_pre) + 10.0),
    )

    cf_lv = t_pre + (c_post - c_pre)
    cf_lg = t_pre * (c_post / c_pre)
    lv = (obs - t_pre) - (c_post - c_pre)
    lg = float(np.log(obs / t_pre) - np.log(c_post / c_pre))

    m1, m2, m3 = st.columns(3)
    m1.metric("DiD in levels", f"{lv:+.2f}")
    m2.metric("DiD in logs", f"{lg:+.4f}", f"{100 * (np.exp(lg) - 1):+.2f}%")
    m3.metric("Counterfactual window", f"{min(cf_lv, cf_lg):.1f} – {max(cf_lv, cf_lg):.1f}")

    if np.sign(lv) != np.sign(lg):
        st.error(
            f"**The two specifications disagree on the SIGN.** Levels says {lv:+.2f}, logs says "
            f"{100 * (np.exp(lg) - 1):+.2f}%. Both are defensible, both get reported as "
            f"'the effect', and nothing in either output flags the conflict.",
            icon="🚨",
        )
    else:
        st.success("Both scales agree on the sign here.", icon="✅")

    term = (t_pre - c_pre) * (c_post / c_pre - 1.0)
    st.markdown(
        "Parallel in levels and parallel in logs are **different assumptions**. Both hold only if "
        f"`(Yt0 - Yc0)(Yc1/Yc0 - 1) = 0`, which here is `{term:.4f}` — so at most one of them is "
        "true. The constructive move: run the pre-trends test on `Y` **and** on `log Y`. "
        "Whichever scale passes its own pre-trend test is the scale the data supports — "
        "unless the pre-period is flat, in which case both pass and the sign of the "
        "reported effect is the analyst's choice rather than the data's."
    )

st.divider()
st.caption(
    "Run `python evidence.py` for the full eight-section measurement, and "
    "`python -m pytest test_did.py` for the 45 assertions behind it."
)
