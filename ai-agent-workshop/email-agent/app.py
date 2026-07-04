from __future__ import annotations

import streamlit as st

from agent import draft_email_reply

st.set_page_config(page_title="Email Draft Agent", page_icon="📧")
st.title("📧 Email Draft Agent")
st.caption("Paste an inbound email. Agent drafts a reply, self-critiques, and revises before showing you the final draft.")

with st.sidebar:
    st.subheader("Settings")
    tone = st.selectbox("Reply tone", ["friendly", "formal", "brief"])
    my_name = st.text_input("Your name", value="Phoebe")
    max_iterations = st.slider("Max revision loops", 1, 5, 3)
    st.info("Set ANTHROPIC_API_KEY env var to use real Claude drafts. Falls back to a template agent otherwise.")

col1, col2 = st.columns(2)
with col1:
    email_from = st.text_input("From", value="Jordan Lee <jordan@acmeco.com>")
with col2:
    subject = st.text_input("Subject", value="Question about next week's rollout")

body = st.text_area(
    "Email body",
    height=180,
    value=(
        "Hi, can you confirm whether the rollout is still happening next Tuesday? "
        "Also, please let me know if we need to update the client-facing docs before then."
    ),
)

if st.button("Draft Reply", type="primary"):
    if not body.strip():
        st.error("Email body can't be empty.")
    else:
        with st.spinner("Agent drafting, critiquing, and revising..."):
            result = draft_email_reply(
                subject=subject,
                email_from=email_from,
                body=body,
                tone=tone,
                my_name=my_name,
                max_iterations=max_iterations,
            )

        st.subheader("Final Draft")
        st.text_area("Reply", value=result.final_draft, height=220)

        st.subheader(f"Agent Trace ({result.iterations} iteration(s))")
        for i, (draft, critique) in enumerate(zip(result.history, result.critiques + [None]), start=1):
            with st.expander(f"Iteration {i}"):
                st.text(draft)
                if critique:
                    if critique.passed:
                        st.success("Self-critique: passed")
                    else:
                        st.warning("Self-critique flagged:\n- " + "\n- ".join(critique.issues))
