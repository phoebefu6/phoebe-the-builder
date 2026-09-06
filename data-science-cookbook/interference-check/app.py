"""Streamlit front end: pick an interference mechanism, pick a design, and watch
the gap between what the test reports and what shipping it would do."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from interference import (
    assign_by_group,
    assign_within_group,
    cluster_estimate,
    dose_response_check,
    market_global_effect,
    market_split_estimate,
    spillover_outcomes,
    spillover_split_bias_closed_form,
    switchback_bias_closed_form,
    switchback_run,
    tightness,
    user_estimate,
)

st.set_page_config(page_title="interference-check", page_icon="🔀", layout="wide")

st.title("A split test measures a transfer between the arms; the decision needs the total")
st.caption(
    "Day 168 · interference-check · the ground truth on every screen is a second simulation of the "
    "same world under global treatment and global control - the quantity a real experiment can never see."
)

TAB_A, TAB_B, TAB_C, TAB_D = st.tabs(
    [
        "① Shared supply: the test overstates",
        "② Peer effects: the test understates",
        "③ Can the guard test see it?",
        "④ Switchback carryover",
    ]
)

# ------------------------------------------------------------------ tab A
with TAB_A:
    c1, c2, c3, c4 = st.columns(4)
    n = c1.select_slider("Buyers in the test", [5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)
    p_c = c2.slider("Control attempt rate", 0.02, 0.30, 0.10, 0.01)
    lift = c3.slider("True lift in attempt rate (pp)", 0.5, 8.0, 3.0, 0.5) / 100
    util = c4.slider("Supply per expected attempt", 0.6, 2.2, 1.0, 0.05)

    p_t = p_c + lift
    supply = int(round(util * n * p_c))
    rng = np.random.default_rng(0)

    est = np.array([market_split_estimate(n, p_c, p_t, supply, rng)[0] for _ in range(60)])
    ses = np.array([market_split_estimate(n, p_c, p_t, supply, rng)[1] for _ in range(20)])
    truth = market_global_effect(n, p_c, p_t, supply, rng, reps=60)
    cf_truth = min(p_t, supply / n) - min(p_c, supply / n)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("What the A/B test reports", f"{est.mean():+.4f}", f"SE {ses.mean():.4f}")
    m2.metric("What shipping it does", f"{truth:+.4f}", f"closed form {cf_truth:+.4f}")
    over = (est.mean() - truth) / truth if truth > 1e-9 else float("inf")
    m3.metric("Overstatement", "infinite" if not np.isfinite(over) else f"{100 * over:+.0f}%")
    m4.metric("Utilisation", f"{tightness(n, p_c, supply):.2f}", "rationed" if util < 1.3 else "slack")

    if util >= 1.4:
        st.success(
            f"Supply is slack ({util:.2f} per attempt), so nobody is taking anything off anybody. "
            "The split test is measuring what the decision needs."
        )
    elif util >= 1.25:
        st.info(
            "Borderline. The bias is still small here, but it is a cliff: drop utilisation by 0.10 "
            "and a third of the reported number becomes bias."
        )
    else:
        st.error(
            f"The reported effect is {100 * (est.mean() - truth) / est.mean():.0f}% bias. Both arms face the same "
            "rationing factor, so it cancels out of the difference and nothing in the readout can warn you. "
            "The estimate is the supply the treated arm won off the control arm."
        )

    band = pd.DataFrame(
        {
            "utilisation": np.round(np.arange(0.7, 2.21, 0.15), 2),
        }
    )
    band["true global effect"] = [min(p_t, u * p_c) - min(p_c, u * p_c) for u in band["utilisation"]]
    band["split estimate (closed form)"] = [
        lift * min(1.0, (u * p_c) / ((p_t + p_c) / 2)) for u in band["utilisation"]
    ]
    st.line_chart(band.set_index("utilisation"))
    st.caption(
        "Both curves are closed forms, verified against simulation in `evidence.py` section 2. "
        "The split estimate is (p_t - p_c) x min(1, S / (n x mean attempt rate)); the truth is "
        "min(p_t, S/n) - min(p_c, S/n)."
    )

# ------------------------------------------------------------------ tab B
with TAB_B:
    c1, c2, c3, c4 = st.columns(4)
    m_size = c1.select_slider("Peer group size", [4, 8, 20, 50, 100], value=20)
    n_groups = c2.select_slider("Number of groups", [50, 100, 300, 600], value=300)
    tau = c3.slider("Direct effect tau", 0.0, 2.0, 1.0, 0.1)
    gamma = c4.slider("Indirect (peer) effect gamma", 0.0, 2.0, 0.5, 0.1)

    rng = np.random.default_rng(1)
    group = np.repeat(np.arange(n_groups), m_size)
    sp, cl = [], []
    for _ in range(80):
        z = assign_within_group(group, rng)
        sp.append(user_estimate(spillover_outcomes(z, group, tau, gamma, 1.0, rng), z)[0])
        z2 = assign_by_group(group, rng)
        cl.append(cluster_estimate(spillover_outcomes(z2, group, tau, gamma, 1.0, rng), z2, group)[0])

    truth = tau + gamma
    bias_cf = spillover_split_bias_closed_form(gamma, m_size)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("User-level split", f"{np.mean(sp):+.3f}", f"bias {np.mean(sp) - truth:+.3f}")
    m2.metric("Cluster randomised", f"{np.mean(cl):+.3f}", f"bias {np.mean(cl) - truth:+.3f}")
    m3.metric("True global effect", f"{truth:+.3f}", "tau + gamma")
    m4.metric("Closed-form split bias", f"{bias_cf:+.3f}", "-gamma x m/(m-1)")
    st.warning(
        f"The split recovers the direct effect only. It misses {100 * abs(bias_cf) / truth:.0f}% of the effect, "
        f"and the bias has no n in it: -gamma x m/(m-1) = {bias_cf:.3f}. Note the SIGN - in tab ① the same "
        "estimator overstated. Which way you are wrong is a property of the mechanism, not of the output."
    )
    st.caption(
        "Cluster randomisation is unbiased here because the peer group IS the interference boundary. "
        "In tab ① it would not be, if supply were pooled above the cluster - see `evidence.py` section 6."
    )

# ------------------------------------------------------------------ tab C
with TAB_C:
    st.markdown(
        "The standard defence: run the feature at two treated shares and test whether the estimated "
        "effect depends on the share. Under no interference it cannot. The logic is right; the power is the problem."
    )
    c1, c2, c3 = st.columns(3)
    n_chk = c1.select_slider("Buyers in the check", [20_000, 50_000, 100_000, 400_000, 1_000_000], value=20_000)
    util_chk = c2.slider("Supply per expected attempt ", 0.6, 2.2, 1.0, 0.05, key="util_chk")
    reps = c3.select_slider("Simulated runs", [40, 100, 200], value=100)

    rng = np.random.default_rng(2)
    supply_chk = int(round(util_chk * n_chk * 0.10))
    res = [dose_response_check(n_chk, 0.10, 0.13, supply_chk, rng, shares=(0.1, 0.5)) for _ in range(reps)]
    p = np.array([r["p"] for r in res])
    power = float(np.mean(p < 0.05))
    truth_chk = min(0.13, supply_chk / n_chk) - min(0.10, supply_chk / n_chk)
    split_chk = 0.03 * min(1.0, (supply_chk / n_chk) / 0.115)

    m1, m2, m3 = st.columns(3)
    m1.metric("Check fires at 0.05", f"{power:.3f}")
    m2.metric("Bias it is trying to catch", f"{split_chk - truth_chk:+.4f}")
    m3.metric("Mean share-to-share gap", f"{np.mean([r['diff'] for r in res]):+.5f}")
    if util_chk >= 1.4:
        st.info("No interference in this world, so the check should fire about 5% of the time - and does.")
    elif power < 0.30:
        st.error(
            f"There is real interference here and the check finds it {power:.1%} of the time, against a ~5% "
            "false-alarm rate. A pass is not evidence of no interference; it is evidence that you ran it."
        )
    else:
        st.success(f"At this traffic the check has usable power ({power:.2f}).")
    st.caption("`evidence.py` section 5 extrapolates ~1.29M buyers for power 0.80 - about 100x the traffic "
               "at which the experiment itself is fully powered.")

# ------------------------------------------------------------------ tab D
with TAB_D:
    st.markdown(
        "Randomise TIME instead of users and interference inside a period is no longer between arms. "
        "The failure mode is that the system does not switch instantly."
    )
    c1, c2, c3, c4 = st.columns(4)
    periods = c1.select_slider("Periods", [100, 200, 400, 800], value=400)
    carry = c2.slider("Carryover c (share of each period still under the old condition)", 0.0, 0.45, 0.20, 0.05)
    burn = c3.slider("Burn-in discarded", 0.0, 0.60, 0.0, 0.05)
    scheme = c4.radio("Assignment", ["coin-flip", "strict ABAB"], index=0)

    rng = np.random.default_rng(3)
    alt = scheme == "strict ABAB"
    e = np.array([switchback_run(periods, 1.0, carry, 1.0, rng, alternating=alt, burn_in=burn)[0] for _ in range(300)])
    pred = switchback_bias_closed_form(1.0, carry, alt, burn_in=burn)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimate (true effect = 1.00)", f"{e.mean():.3f}", f"closed form {pred:.3f}")
    m2.metric("Attenuation", f"{100 * (1 - e.mean()):.1f}%")
    m3.metric("Variance", f"{e.var():.5f}")
    if burn >= carry and carry > 0:
        st.success(
            f"Burn-in {burn:.2f} covers the carryover {carry:.2f}, so the bias is gone. Everything past "
            f"c costs 1/(1-b) of the variance and buys nothing."
        )
    elif carry == 0:
        st.info("No carryover, so the switchback is unbiased under either assignment scheme.")
    else:
        st.error(
            f"{'Strict alternation attenuates by 2c' if alt else 'Coin-flip attenuates by c'}: the estimate is "
            f"{e.mean():.3f} against a truth of 1.00. Switch the assignment scheme and watch the bias halve or "
            "double - balance in the assignment is not balance in the exposure."
        )
    st.caption(
        "Periods, not users, are the sample here: 400 periods is n=400. That is why switchbacks are run on a "
        "metric measured per period. See `evidence.py` section 8 for the burn-in MSE curve."
    )

st.divider()
st.markdown(
    "**Behind this app:** `python evidence.py` (the nine-section measurement) · "
    "`python -m pytest test_interference.py` (25 assertions) · `python make_chart.py` (the six-panel figure)."
)
