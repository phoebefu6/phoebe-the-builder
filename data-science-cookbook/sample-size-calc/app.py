"""Streamlit UI for the sample size and power calculator.

Run: streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import power as pw

st.set_page_config(page_title="Sample Size & Power Calculator", layout="wide")

st.title("Sample Size & Power Calculator")
st.caption(
    "Before the test: how many users, how many days, and what you actually "
    "have the traffic to detect."
)

# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Your test")
    metric_kind = st.radio(
        "Primary metric", ["Conversion rate", "Continuous (revenue, AOV, time)"]
    )

    if metric_kind == "Conversion rate":
        baseline = (
            st.number_input(
                "Baseline rate (%)", 0.01, 99.0, 4.2, 0.1, format="%.2f"
            )
            / 100.0
        )
        effect_kind = st.radio("Effect expressed as", ["relative", "absolute"])
        if effect_kind == "relative":
            effect = (
                st.number_input("Lift you want to detect (%)", 0.1, 500.0, 10.0, 0.5)
                / 100.0
            )
            st.caption(
                f"{baseline * 100:.2f}% -> {baseline * (1 + effect) * 100:.2f}%"
            )
        else:
            effect = (
                st.number_input(
                    "Absolute lift (percentage points)", 0.01, 50.0, 0.42, 0.01
                )
                / 100.0
            )
            st.caption(f"{baseline * 100:.2f}% -> {(baseline + effect) * 100:.2f}%")
    else:
        sigma = st.number_input("Std dev of the metric", 0.01, 1e7, 42.0, 1.0)
        delta = st.number_input("Shift you want to detect", 0.01, 1e7, 3.0, 0.5)

    st.header("Rigour")
    alpha = st.select_slider(
        "Alpha (false positive rate)", [0.01, 0.05, 0.10], value=0.05
    )
    target_power = st.select_slider(
        "Power (chance of catching a real effect)",
        [0.70, 0.80, 0.90, 0.95],
        value=0.80,
    )
    n_variants = st.number_input("Arms including control", 2, 10, 2)
    correction = st.selectbox(
        "Multiple-comparison correction", ["none", "bonferroni", "sidak"]
    )

    st.header("Traffic")
    daily = st.number_input("Daily eligible users", 1.0, 1e8, 1800.0, 50.0)
    exposure = st.slider("Share of traffic in the test (%)", 5, 100, 100) / 100.0
    window = st.number_input("Decision window (days)", 1, 180, 28)

# ----------------------------------------------------------------- results
if metric_kind == "Conversion rate":
    plan = pw.plan(
        baseline,
        effect,
        daily,
        effect_kind,
        alpha,
        target_power,
        int(n_variants),
        correction,
        exposure,
        int(window),
    )
    size = plan["size"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users per arm", f"{int(size['n_per_arm']):,}")
    c2.metric("Total users", f"{int(size['n_total']):,}")
    c3.metric(
        "Days needed",
        f"{plan['days_rounded']:,}",
        delta=f"window is {plan['window_days']}",
        delta_color="normal" if plan["feasible"] else "inverse",
    )
    reach = plan["reachable_in_window"]
    c4.metric(
        f"Detectable in {plan['window_days']}d",
        f"{reach['relative_mde'] * 100:.1f}%" if reach["detectable"] else "n/a",
    )

    if plan["feasible"]:
        st.success(
            f"Feasible: {plan['days_rounded']} days inside a "
            f"{plan['window_days']}-day window."
        )
    else:
        st.error(
            f"Not feasible as scoped. {plan['days_rounded']} days needed, "
            f"{plan['window_days']} available."
        )

    for w in plan["warnings"]:
        st.warning(w)

    if int(n_variants) > 2:
        st.info(
            "Family-wise error if uncorrected: "
            f"{pw.familywise_error(alpha, int(n_variants)) * 100:.1f}%. "
            f"Alpha per comparison in use: "
            f"{size['alpha_per_comparison']:.4f}."
        )

    st.subheader("What each lift costs you")
    rows = pw.sensitivity_table(
        baseline,
        daily,
        alpha=alpha,
        power=target_power,
        n_variants=int(n_variants),
        correction=correction,
        exposure_share=exposure,
    )
    df = pd.DataFrame(rows)
    show = pd.DataFrame(
        {
            "Relative lift": [f"{r['relative_lift'] * 100:.0f}%" for r in rows],
            "Treated rate": [f"{r['treated_rate'] * 100:.2f}%" for r in rows],
            "Users / arm": [f"{int(r['n_per_arm']):,}" for r in rows],
            "Total users": [f"{int(r['n_total']):,}" for r in rows],
            "Days": [f"{r['days']:.1f}" for r in rows],
            "Fits window": [
                "yes" if r["days"] <= window else "no" for r in rows
            ],
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["relative_lift"] * 100, df["days"], marker="o", color="#2b6cb0")
    ax.axhline(
        window, color="#c53030", ls="--", label=f"{int(window)}-day window"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Relative lift you are hunting (%)")
    ax.set_ylabel("Days needed (log scale)")
    ax.set_title("Small effects are expensive")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    st.pyplot(fig)

    st.download_button(
        "Download plan (CSV)",
        df.to_csv(index=False).encode(),
        file_name="sample_size_plan.csv",
        mime="text/csv",
    )

else:
    res = pw.n_for_means(
        sigma, delta, alpha, target_power, int(n_variants), correction
    )
    days = pw.duration_days(int(res["n_total"]), daily, exposure)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users per arm", f"{int(res['n_per_arm']):,}")
    c2.metric("Total users", f"{int(res['n_total']):,}")
    c3.metric("Days needed", f"{days:.0f}")
    c4.metric("Cohen's d", f"{res['effect_size_d']:.3f}")
    st.caption(
        "Exact two-sample t-test power via the noncentral t distribution. "
        f"Achieved power at this n: {res['achieved_power']:.3f}."
    )
    if res["effect_size_d"] < 0.1:
        st.warning(
            f"d = {res['effect_size_d']:.3f} is a very small effect relative "
            "to the noise in this metric. Revenue-style metrics are heavy "
            "tailed - consider winsorising or switching to a bounded proxy."
        )
    if days > window:
        st.error(f"{days:.0f} days needed, {int(window)} available.")

st.divider()
st.caption(
    "Day 123 of Phoebe's FDE portfolio - Data Science Cookbook. "
    "Sizing assumes a single look at the end of the test."
)
