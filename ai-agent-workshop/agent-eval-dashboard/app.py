from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agent_eval import (
    SAMPLE_CASES,
    SAMPLE_TRACES_V1,
    SAMPLE_TRACES_V2,
    category_breakdown,
    evaluate,
    results_frame,
    summarize,
)

st.set_page_config(page_title="Agent Evaluation Dashboard", page_icon="🧪", layout="wide")
st.title("🧪 Agent Evaluation Dashboard")
st.caption(
    "\"We don't know if our agents are actually good.\" Run a fixed eval suite against two "
    "agent versions and see pass rate, quality, latency, and per-category regressions."
)

with st.sidebar:
    st.subheader("Settings")
    use_claude = st.checkbox("Use Claude as judge (needs ANTHROPIC_API_KEY)", value=False)
    st.caption(f"Eval suite: {len(SAMPLE_CASES)} cases across "
               f"{len(set(c.category for c in SAMPLE_CASES))} categories.")
    st.caption("A case passes if quality ≥ 0.6, within its latency SLA, and the correct tool was called.")

r1 = evaluate(SAMPLE_CASES, SAMPLE_TRACES_V1, use_claude=use_claude)
r2 = evaluate(SAMPLE_CASES, SAMPLE_TRACES_V2, use_claude=use_claude)
s1, s2 = summarize(r1), summarize(r2)

st.subheader("v1 (baseline) vs v2 (candidate)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pass rate", f"{s2['pass_rate']:.0%}", f"{(s2['pass_rate']-s1['pass_rate']):+.0%}")
c2.metric("Avg quality", f"{s2['avg_quality']:.2f}", f"{(s2['avg_quality']-s1['avg_quality']):+.2f}")
c3.metric("Avg latency (ms)", f"{s2['avg_latency_ms']:.0f}",
          f"{(s2['avg_latency_ms']-s1['avg_latency_ms']):+.0f}", delta_color="inverse")
c4.metric("Total tokens", f"{s2['total_tokens']}",
          f"{(s2['total_tokens']-s1['total_tokens']):+d}", delta_color="inverse")

st.subheader("Pass rate by category")
b1 = category_breakdown(r1).rename(columns={"pass_rate": "v1"})
b2 = category_breakdown(r2).rename(columns={"pass_rate": "v2"})
merged = b1[["category", "v1"]].merge(b2[["category", "v2"]], on="category")
fig, ax = plt.subplots(figsize=(8, 3.6))
x = range(len(merged))
ax.bar([i - 0.2 for i in x], merged["v1"], width=0.4, label="v1", color="#b0b0b0")
ax.bar([i + 0.2 for i in x], merged["v2"], width=0.4, label="v2", color="#4361ee")
ax.set_xticks(list(x))
ax.set_xticklabels(merged["category"])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Pass rate")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("v1 case results")
    st.dataframe(results_frame(r1), hide_index=True, use_container_width=True)
with col_b:
    st.subheader("v2 case results")
    st.dataframe(results_frame(r2), hide_index=True, use_container_width=True)

regressions = [c for c in merged["category"] if merged.loc[merged["category"] == c, "v2"].iloc[0]
               < merged.loc[merged["category"] == c, "v1"].iloc[0]]
if regressions:
    st.error(f"⚠️ Regressions in: {', '.join(regressions)}")
else:
    st.success("✅ No category regressed from v1 to v2 — safe to ship the candidate.")
