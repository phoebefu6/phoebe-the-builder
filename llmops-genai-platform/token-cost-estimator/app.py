from __future__ import annotations

# Streamlit UI for the Token & Cost Estimator. Describe a workload, price it
# across every candidate model, find the break-even volume for a budget, and
# see which lever actually moves the bill - all before you write the
# integration. Fully offline; rates are an editable table in estimate.py.
import pandas as pd
import streamlit as st
from estimate import (
    CHARS_PER_TOKEN,
    PRICING,
    Workload,
    break_even_volume,
    compare_models,
    estimate_tokens,
    portfolio,
    sample_workloads,
    sensitivity,
)

st.set_page_config(page_title="Token & Cost Estimator", page_icon="🧮", layout="wide")

st.title("🧮 Token & Cost Estimator")
st.caption(
    "Price an LLM feature *before* you build it. Day 85's tracker tells you what "
    "you spent; this tells you what you're about to spend. Rates below are example "
    "values - edit PRICING in estimate.py for your contract."
)

# ---------------------------------------------------------------- sidebar spec
st.sidebar.header("Workload spec (per call)")
name = st.sidebar.text_input("Feature name", "support-rag-bot")
system_tokens = st.sidebar.number_input("System prompt tokens", 0, 200_000, 600, step=50)
user_tokens = st.sidebar.number_input("User input tokens", 0, 200_000, 120, step=50)
context_tokens = st.sidebar.number_input(
    "Context tokens (RAG chunks, few-shot, tool schemas)", 0, 500_000, 4_000, step=250
)
output_tokens = st.sidebar.number_input("Output tokens", 0, 200_000, 350, step=50)

st.sidebar.header("Volume & efficiency")
calls_per_month = st.sidebar.number_input("Calls per month", 0, 50_000_000, 40_000, step=1_000)
retry_rate = st.sidebar.slider("Retry rate", 0.0, 1.0, 0.05, 0.01)
cache_hit_rate = st.sidebar.slider("Cache hit rate", 0.0, 1.0, 0.30, 0.05)
monthly_budget = st.sidebar.number_input("Monthly budget (USD)", 0.0, 1_000_000.0, 500.0, step=50.0)

w = Workload(
    name=name,
    system_tokens=int(system_tokens),
    user_tokens=int(user_tokens),
    context_tokens=int(context_tokens),
    output_tokens=int(output_tokens),
    calls_per_month=int(calls_per_month),
    retry_rate=float(retry_rate),
    cache_hit_rate=float(cache_hit_rate),
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Input tokens / call", f"{w.input_tokens:,}")
c2.metric("Output tokens / call", f"{w.output_tokens:,}")
c3.metric("Billed calls / mo", f"{w.billed_calls:,.0f}", f"{w.calls_per_month:,} attempted")
c4.metric("Context share of input", f"{(w.context_tokens / max(1, w.input_tokens)):.0%}")

tab_models, tab_budget, tab_levers, tab_product, tab_counter = st.tabs(
    ["🤖 Model comparison", "🎯 Break-even", "🎚️ Sensitivity", "📦 Whole product", "✍️ Token counter"]
)

with tab_models:
    rows = compare_models(w)
    df = pd.DataFrame(rows)[
        ["model", "per_call_usd", "billed_calls", "monthly_usd", "annual_usd", "input_cost_share"]
    ]
    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df.set_index("model")["monthly_usd"])
    cheapest, dearest = rows[0], rows[-1]
    st.info(
        f"**{cheapest['model']}** runs this workload at "
        f"${cheapest['monthly_usd']:,.2f}/mo vs ${dearest['monthly_usd']:,.2f}/mo on "
        f"**{dearest['model']}** - a "
        f"{dearest['monthly_usd'] / max(0.01, cheapest['monthly_usd']):.0f}x spread on "
        f"identical traffic. `input_cost_share` shows how much of the bill is *prompt*, "
        f"not completion: when it is high, trimming context beats trimming output."
    )
    over = [r["model"] for r in rows if r["monthly_usd"] > monthly_budget]
    if over:
        st.warning(f"Over the ${monthly_budget:,.0f}/mo budget: {', '.join(over)}")

with tab_budget:
    models = list(PRICING)
    cheap = st.selectbox("Cheap option", models, index=len(models) - 2)
    premium = st.selectbox("Premium option", models, index=0)
    be = break_even_volume(w, cheap, premium, monthly_budget)
    b1, b2 = st.columns(2)
    b1.metric(
        f"{be['cheap']['model']} covers",
        f"{be['cheap']['calls_covered']:,} calls/mo",
        f"${be['cheap']['per_call_usd']:.6f}/call",
    )
    b2.metric(
        f"{be['premium']['model']} covers",
        f"{be['premium']['calls_covered']:,} calls/mo",
        f"{be['premium_multiple']}x per-call price",
    )
    st.caption(
        f"On ${monthly_budget:,.0f}/mo, the same budget buys "
        f"{be['cheap']['calls_covered'] / max(1, be['premium']['calls_covered']):.0f}x more "
        "traffic on the cheap model. Route the easy majority there and reserve the premium model "
        "for the calls that actually need it (Day 90 - LLM Model Router)."
    )

with tab_levers:
    lever = st.radio(
        "Lever", ["volume", "context_tokens", "output_tokens", "cache_hit_rate"], horizontal=True
    )
    model = st.selectbox("Model", list(PRICING), index=1, key="lever_model")
    rows = sensitivity(w, model, lever)
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.line_chart(df.set_index("factor")["monthly_usd"])
    st.caption(
        "Widest spread wins your engineering time. For long-context RAG the context "
        "lever usually beats the output lever - the opposite of the common "
        "'output costs 5x' rule of thumb."
    )

with tab_product:
    st.write("Roll every feature up onto one model - the number finance actually asks for.")
    wl = sample_workloads()
    st.caption("Sample product: " + ", ".join(x.name for x in wl) + " (edit `sample_workloads()`)")
    recs = []
    for m in PRICING:
        p = portfolio(wl, m)
        recs.append(
            {
                "model": m,
                "monthly_usd": p["monthly_usd"],
                "annual_usd": p["annual_usd"],
                "top_line_item": p["top_line_item"],
                "top_line_share": p["top_line_share"],
            }
        )
    pdf = pd.DataFrame(recs).sort_values("monthly_usd")
    st.dataframe(pdf, width="stretch", hide_index=True)
    st.bar_chart(pdf.set_index("model")["annual_usd"])
    st.info(
        "One feature dominates the bill on every model - so optimize that one and "
        "ignore the rest. Uniform 'let's use the cheap model everywhere' is a worse "
        "trade than fixing the top line item."
    )

with tab_counter:
    st.write("Paste real prompt text to turn it into a token estimate for the spec above.")
    kind = st.selectbox("Content type", list(CHARS_PER_TOKEN), index=0)
    text = st.text_area(
        "Text", "You are a support assistant. Answer only from the provided context."
    )
    n = estimate_tokens(text, kind)
    t1, t2 = st.columns(2)
    t1.metric("Estimated tokens", f"{n:,}")
    t2.metric("Chars per token", f"{CHARS_PER_TOKEN[kind]}")
    st.caption(
        "Heuristic by design - a design-time estimate wants ±10-15%, not exactness. "
        "Get exact counts from the provider at runtime and log them with Day 85's tracker."
    )

st.divider()
st.caption(
    "Day 125 of Phoebe's daily FDE build - part of the LLMOps & GenAI Platform line. "
    "Pairs with Day 85 (cost tracker), Day 86 (semantic cache), Day 90 (model router)."
)
