from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from okr_tracker import SAMPLE_OBJECTIVES, advise, quarter_summary

st.set_page_config(page_title="OKR Tracker & Advisor", page_icon="🎯", layout="wide")
st.title("🎯 OKR Tracker & Advisor")
st.caption(
    "OKRs get set and forgotten. This tracks progress against pace and tells you which key "
    "results are behind — while there's still time to act."
)

objectives = SAMPLE_OBJECTIVES  # sample set; real app would load from a store

with st.sidebar:
    st.subheader("Period")
    time_elapsed = st.slider("Time elapsed in quarter", 0.0, 1.0, 0.6, 0.05,
                             help="A KR is 'behind pace' if its progress trails the time elapsed.")
    st.caption("At-risk if ≥10% behind pace · off-track if ≥25% behind.")

summary = quarter_summary(objectives, time_elapsed)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg progress", f"{summary['avg_progress']:.0%}")
c2.metric("On track", summary["on_track"])
c3.metric("At risk", summary["at_risk"])
c4.metric("Off track", summary["off_track"])

color = {"on-track": "#2e9e5b", "done": "#2e9e5b", "at-risk": "#e0a800", "off-track": "#d64545"}

st.subheader("Key results — progress vs. pace")
for obj in objectives:
    st.markdown(f"**{obj.name}** — {obj.progress:.0%} · `{obj.status(time_elapsed)}`")
    for kr in obj.key_results:
        st.write(f"{kr.name}: {kr.fmt(kr.current)} → {kr.fmt(kr.target)}")
        st.progress(kr.progress, text=f"{kr.progress:.0%} · {kr.status(time_elapsed)}")

st.subheader("Progress vs. pace line")
rows = [(f"{kr.name}", kr.progress) for o in objectives for kr in o.key_results]
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
stats = [kr.status(time_elapsed) for o in objectives for kr in o.key_results]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(labels, vals, color=[color[s] for s in stats])
ax.axvline(time_elapsed, color="#333", linestyle="--", label=f"pace ({time_elapsed:.0%})")
ax.set_xlim(0, 1)
ax.set_xlabel("Progress")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.subheader("🚦 Advisor")
advice = advise(objectives, time_elapsed)
if not advice:
    st.success("All key results on pace. Keep going.")
for a in advice:
    (st.error if a.status == "off-track" else st.warning)(f"**{a.key_result}** ({a.objective}): {a.message}")
