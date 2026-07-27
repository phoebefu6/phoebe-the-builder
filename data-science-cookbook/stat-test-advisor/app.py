from __future__ import annotations

import numpy as np
import streamlit as st

from advisor import recommend, run_test, sample_data

st.set_page_config(page_title="Statistical Test Advisor", layout="wide")
st.title("Statistical Test Advisor")
st.caption('"Which test do I use?" — describe your data, get the right test, and run it.')

d = sample_data()

with st.sidebar:
    st.subheader("What are you comparing?")
    kind = st.selectbox("Question type", [
        "numeric_2groups", "numeric_multi", "categorical_2vars", "correlation",
    ], format_func=lambda k: {
        "numeric_2groups": "Means of 2 groups",
        "numeric_multi": "Means of 3+ groups",
        "categorical_2vars": "Association of 2 categories",
        "correlation": "Correlation of 2 numeric vars",
    }[k])
    paired = st.checkbox("Paired / repeated measures", value=False) if kind == "numeric_2groups" else False
    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05)

if kind == "numeric_2groups":
    groups = [d["group_A"], d["group_B"]]
    contingency = None
elif kind == "numeric_multi":
    groups = [d["group_A"], d["group_B"], d["group_C"]]
    contingency = None
elif kind == "correlation":
    groups = [d["group_A"], d["group_A"] * 0.6 + np.random.default_rng(2).normal(0, 10, len(d["group_A"]))]
    contingency = None
else:
    groups = []
    contingency = d["contingency"]

rec = recommend(kind, groups, paired=paired)

st.subheader("Recommended test")
st.success(f"**{rec.test}** — {rec.reason}")
st.caption("Assumptions: " + (", ".join(rec.assumptions) or "none") + f" · {'parametric' if rec.parametric else 'non-parametric'}")

if st.button("Run the test", type="primary"):
    result = run_test(rec, groups, alpha=alpha, contingency=contingency)
    c1, c2 = st.columns(2)
    c1.metric("Statistic", f"{result.statistic:.4f}")
    c2.metric("p-value", f"{result.p_value:.4f}")
    if result.effect:
        st.caption(f"Effect size: {result.effect}")
    (st.success if result.p_value < alpha else st.info)(result.conclusion)

with st.expander("Sample data"):
    st.write({k: (np.round(v[:8], 1).tolist() if hasattr(v, "__len__") else v) for k, v in d.items()})
