from __future__ import annotations

import pandas as pd
import streamlit as st
from router import Ticket, route_batch, route_ticket

st.set_page_config(page_title="Ticket Router", page_icon="🎫")
st.title("🎫 Customer Ticket Router")
st.caption("Classifies inbound tickets into the right team, flags low-confidence cases for human triage instead of guessing.")

SAMPLE_TICKETS = [
    Ticket("T1", "Overcharged on my invoice", "I was charged twice for my subscription this month, please refund."),
    Ticket("T2", "Getting a 500 error on login", "App keeps crashing with a 500 error every time I try to log in, this is broken."),
    Ticket("T3", "Forgot my password", "I got locked out of my account and need a password reset."),
    Ticket("T4", "Interested in enterprise plan", "Can I get a quote for the enterprise plan and a demo?"),
    Ticket("T5", "Hello", "just checking in, no specific issue"),
]

with st.sidebar:
    st.subheader("Settings")
    st.info("Set ANTHROPIC_API_KEY for Claude-based classification. Falls back to weighted keyword matching otherwise.")
    threshold = st.slider("Low-confidence threshold (routes to human triage below this)", 0.0, 1.0, 0.35, 0.05)
    st.caption("Teams: billing, technical, account, sales, general, human-triage")

tab1, tab2 = st.tabs(["Single ticket", "Batch (sample inbox)"])

with tab1:
    subject = st.text_input("Subject", value="App keeps crashing")
    body = st.text_area("Body", value="Getting a 500 error every time I try to log in, this is broken.", height=120)
    if st.button("Route ticket", type="primary"):
        result = route_ticket(Ticket("manual", subject, body), low_confidence_threshold=threshold)
        st.metric("Routed to", result.team, f"{result.confidence:.0%} confidence")
        st.write(result.reasoning)

with tab2:
    if st.button("Route sample inbox"):
        results = route_batch(SAMPLE_TICKETS, low_confidence_threshold=threshold)
        df = pd.DataFrame([
            {
                "Ticket": r.ticket_id,
                "Team": r.team,
                "Confidence": f"{r.confidence:.0%}",
                "Reasoning": r.reasoning,
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        team_counts = pd.Series([r.team for r in results]).value_counts()
        st.subheader("Routing distribution")
        st.bar_chart(team_counts)
