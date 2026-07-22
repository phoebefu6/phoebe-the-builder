from __future__ import annotations

# Streamlit UI for the LLM Model Router. Route a mixed traffic stream to the
# cheapest capable tier, see the cost saved vs always-large, and test any
# single request to see which tier it lands on and why. Fully offline.

import pandas as pd
import streamlit as st

from router import SAMPLE_TRAFFIC, TIERS, route, run_traffic

st.set_page_config(page_title="LLM Model Router", page_icon="🚦", layout="wide")

st.title("🚦 LLM Model Router")
st.caption(
    "Sending every request to your biggest model is simple and expensive. "
    "Most traffic is easy and a small model handles it fine. This router "
    "scores each request's complexity from cheap signals and sends it to the "
    "smallest capable tier - then shows the cost saved. Transparent, rule-based."
)

res = run_traffic()
c1, c2, c3 = st.columns(3)
c1.metric("Always-large cost", f"${res['always_large_cost']:.4f}")
c2.metric("Routed cost", f"${res['routed_cost']:.4f}")
c3.metric("Saved", f"{res['saved_pct']}%", f"${res['saved']:.4f}")

st.subheader("Tier distribution")
tc = res["tier_counts"]
st.bar_chart(pd.DataFrame({"requests": tc}))

st.subheader("Routing decisions")
st.dataframe(pd.DataFrame(res["rows"]), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Test a single request")
preset = st.selectbox("Load a sample", ["(custom)"] + [t[:70] for t, _ in SAMPLE_TRAFFIC])
default = "" if preset == "(custom)" else dict((t[:70], t) for t, _ in SAMPLE_TRAFFIC)[preset]
text = st.text_area("Request", value=default, height=110,
                    placeholder="e.g. Explain step-by-step why this query is slow...")

if st.button("Route", type="primary") and text.strip():
    r = route(text)
    tier_price = {t.name: t.price_per_mtok for t in TIERS}
    st.markdown(f"### → **{r.tier}** tier · complexity **{r.score}** "
                f"(${tier_price[r.tier]:.2f}/Mtok)")
    st.markdown("**Why:** " + "; ".join(r.reasons))
    st.caption("Route bars: small ≤ 0.34 · medium ≤ 0.67 · large otherwise. "
               "Tune thresholds in router.py to your quality bar.")

st.divider()
st.caption("Day 90 of Phoebe's FDE build sprint · LLMOps & GenAI Platform · "
           "cheapest capable model per request; prices illustrative, edit TIERS.")
