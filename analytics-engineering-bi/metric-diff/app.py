from __future__ import annotations

import pandas as pd
import streamlit as st
from diff import diff_mean, diff_rate, sample_mean_metric, sample_rate_metric


def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _show(r) -> None:
    badge = "🟢 SIGNIFICANT" if r.significant else "⚪ NOT SIGNIFICANT"
    st.markdown(f"### {badge}")
    m1, m2, m3, m4 = st.columns(4)
    fmt = (lambda v: f"{v:.2%}") if r.kind == "rate" else (lambda v: f"{v:,.2f}")
    m1.metric("Baseline", fmt(r.baseline))
    m2.metric("Current", fmt(r.current), delta=f"{r.rel_delta * 100:+.1f}%")
    m3.metric("p-value", f"{r.p_value:.4f}")
    m4.metric("95% CI (abs)", f"[{r.ci_low:+.4g}, {r.ci_high:+.4g}]")
    (st.success if r.significant else st.info)(r.verdict)
    st.dataframe(pd.DataFrame([r.as_row()]), use_container_width=True)


st.set_page_config(page_title="Metric Diff", layout="wide")
st.title("Metric Diff — did this number really change?")
st.caption(
    "Period-over-period delta + a significance test, so you stop reacting to noise."
)

alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
st.sidebar.caption("p < alpha => the change is unlikely to be sampling noise.")

tab_mean, tab_rate = st.tabs(["Mean metric (samples)", "Rate metric (proportion)"])

with tab_mean:
    st.subheader("Continuous metric — Welch's t-test")
    st.write("Row-level values per period. Uses the sample (Avg Order Value) by default.")
    name, last_wk, this_wk = sample_mean_metric()
    metric_name = st.text_input("Metric name", name, key="mean_name")

    use_sample = st.checkbox("Use full sample dataset", value=True)
    if use_sample:
        base_vals, cur_vals = last_wk, this_wk
        st.caption(f"Sample: {len(last_wk)} baseline vs {len(this_wk)} current observations.")
    else:
        c1, c2 = st.columns(2)
        b_txt = c1.text_area("Baseline values (comma/space separated)", "", height=120)
        c_txt = c2.text_area("Current values", "", height=120)
        base_vals = [float(x) for x in b_txt.replace(",", " ").split() if _isnum(x)]
        cur_vals = [float(x) for x in c_txt.replace(",", " ").split() if _isnum(x)]

    if st.button("Compare means", type="primary"):
        try:
            _show(diff_mean(metric_name, base_vals, cur_vals, alpha=alpha))
        except ValueError as e:
            st.error(str(e))

with tab_rate:
    st.subheader("Rate metric — two-proportion z-test")
    st.write("The classic '5.1% vs 5.4% — is that real?' question.")
    rn, bs, bt, cs, ct = sample_rate_metric()
    metric_name_r = st.text_input("Metric name", rn, key="rate_name")
    c1, c2 = st.columns(2)
    bs = c1.number_input("Baseline successes", 0, value=bs)
    bt = c1.number_input("Baseline total", 1, value=bt)
    cs = c2.number_input("Current successes", 0, value=cs)
    ct = c2.number_input("Current total", 1, value=ct)
    if st.button("Compare rates", type="primary"):
        try:
            _show(diff_rate(metric_name_r, int(bs), int(bt), int(cs), int(ct), alpha=alpha))
        except ValueError as e:
            st.error(str(e))
