from __future__ import annotations

# Streamlit UI for the LLM Cost & Token Tracker. See spend by model, tag, and
# day; check the monthly budget; and estimate the cost of a single call before
# you make it. Fully offline - rates are an editable table.
import pandas as pd
import streamlit as st
from tracker import PRICING, call_cost, estimate_tokens, sample_tracker

st.set_page_config(page_title="LLM Cost & Token Tracker", page_icon="💸", layout="wide")

st.title("💸 LLM Cost & Token Tracker")
st.caption(
    "No more surprise API bills. Log every call, roll spend up by model / tag / "
    "day, and watch it against a monthly budget. Rates are example values - "
    "edit PRICING in tracker.py for your contract."
)

t = sample_tracker()
b = t.budget_status()

col1, col2, col3 = st.columns(3)
col1.metric("Total spend", f"${t.total_cost}")
col2.metric("Total tokens", f"{t.total_tokens:,}")
col3.metric("Budget used", f"{b['pct']}%", f"${b['remaining']} left")
st.progress(min(1.0, b["pct"] / 100))
if b["over"]:
    st.error("Over budget for the month.")

tab_model, tab_tag, tab_day, tab_calc = st.tabs(
    ["🤖 By model", "🏷️ By tag", "📅 By day", "🧮 Estimate a call"]
)

with tab_model:
    df = pd.DataFrame(t.by_model())
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("model")["cost_usd"])
    st.info(
        "The cheapest per-call model is rarely the cheapest line item. Watch "
        "which model dominates *total* spend, not per-call price."
    )

with tab_tag:
    df = pd.DataFrame(t.by_tag())
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("tag")["cost_usd"])

with tab_day:
    df = pd.DataFrame(t.by_day())
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.line_chart(df.set_index("day")["cost_usd"])

with tab_calc:
    st.subheader("What will this call cost?")
    model = st.selectbox("Model", list(PRICING))
    prompt = st.text_area("Prompt text", "Summarize this 2-page contract in 5 bullets.", height=100)
    est_out = st.number_input("Expected completion tokens", 1, 100000, 400)
    pt = estimate_tokens(prompt)
    cost = call_cost(model, pt, int(est_out))
    st.write(f"Estimated prompt tokens: **{pt}** · completion: **{int(est_out)}**")
    st.metric("Estimated cost", f"${cost}")
    st.caption(f"At 10,000 calls/month: ~${round(cost * 10000, 2)}")

st.divider()
st.caption("Day 85 of Phoebe's FDE portfolio · LLMOps & GenAI Platform · `python tracker.py` for the CLI.")
