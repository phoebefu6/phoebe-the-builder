from __future__ import annotations

"""Streamlit UI: an A/B test significance calculator with a clear verdict."""

import streamlit as st
from abtest import required_sample_size, run_ab_test

st.set_page_config(page_title="A/B Test Calculator", page_icon="🧪", layout="wide")

st.title("🧪 A/B Test Calculator")
st.caption("Stop eyeballing significance. Enter your numbers and get a proper two-proportion z-test: p-value, confidence interval, and a clear verdict.")

with st.sidebar:
    st.header("Test data")
    cn = st.number_input("Control visitors", min_value=1, value=5000, step=100)
    cc = st.number_input("Control conversions", min_value=0, value=500, step=10)
    vn = st.number_input("Variant visitors", min_value=1, value=5000, step=100)
    vc = st.number_input("Variant conversions", min_value=0, value=560, step=10)
    alpha = st.select_slider("Significance level (α)", options=[0.10, 0.05, 0.01], value=0.05)

try:
    r = run_ab_test(int(cn), int(cc), int(vn), int(vc), alpha=alpha)
except ValueError as exc:
    st.error(f"Check your inputs: {exc}")
    st.stop()

# Verdict banner.
if r.significant:
    st.success(f"✅ **Significant** at α={alpha:g}. Winner: **{r.winner}** "
               f"(p = {r.p_value:.4f}).")
else:
    st.warning(f"🤔 **Not significant** at α={alpha:g} (p = {r.p_value:.4f}). "
               f"The difference could be noise - keep testing or collect more data.")

c = st.columns(4)
c[0].metric("Control rate", f"{r.control_rate*100:.2f}%")
c[1].metric("Variant rate", f"{r.variant_rate*100:.2f}%",
            delta=f"{r.abs_diff*100:+.2f} pp")
c[2].metric("Relative lift", f"{r.rel_lift:+.1f}%")
c[3].metric("p-value", f"{r.p_value:.4f}")

st.subheader("Conversion rate with 95% CI on the difference")
# Bar chart of the two rates with an error bar derived from the CI on the difference.
import pandas as pd

chart_df = pd.DataFrame({"group": ["control", "variant"],
                         "rate_%": [r.control_rate*100, r.variant_rate*100]}).set_index("group")
st.bar_chart(chart_df, height=280)
st.caption(f"Absolute difference: {r.abs_diff*100:+.2f} pp · "
           f"{int((1-alpha)*100)}% CI: [{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}] pp · "
           f"z = {r.z:.3f}")
if r.ci_low <= 0 <= r.ci_high:
    st.caption("ℹ️ The CI includes 0 - consistent with 'no real difference'.")

st.divider()
st.subheader("📐 Sample size planner")
sc = st.columns(3)
with sc[0]:
    base = st.number_input("Baseline rate (%)", min_value=0.1, max_value=99.0, value=10.0) / 100
with sc[1]:
    mde = st.number_input("Min detectable lift (pp)", min_value=0.1, value=2.0) / 100
with sc[2]:
    power = st.select_slider("Power", options=[0.8, 0.9, 0.95], value=0.8)
try:
    n = required_sample_size(base, mde, alpha=alpha, power=power)
    st.info(f"You need ~**{n:,} visitors per variant** to detect a {mde*100:g}pp lift "
            f"at α={alpha:g}, power={power:g}.")
except ValueError as exc:
    st.error(str(exc))

st.caption("Stats core lives in `abtest.py` - reusable as an A/B Test app on the platform shell.")
