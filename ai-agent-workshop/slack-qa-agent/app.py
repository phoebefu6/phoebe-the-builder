from __future__ import annotations

import pandas as pd
import streamlit as st

from slack_qa_agent import SAMPLE_KB, TfidfRetriever, answer_question, post_to_slack

st.set_page_config(page_title="Slack Q&A Agent", page_icon="💬")
st.title("💬 Slack Q&A Agent")
st.caption(
    "Answers repetitive #general questions from your knowledge base via RAG — and routes "
    "anything it isn't confident about to a human instead of guessing."
)


@st.cache_resource
def get_retriever() -> TfidfRetriever:
    return TfidfRetriever(SAMPLE_KB)


retriever = get_retriever()

with st.sidebar:
    st.subheader("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.35, 0.05,
                          help="Below this retrieval score, the agent escalates to a human.")
    escalation_channel = st.text_input("Escalation channel", "#people-ops")
    st.info("Set ANTHROPIC_API_KEY for a Claude-synthesized answer, and SLACK_BOT_TOKEN to actually post. Both are optional — the demo runs fully offline.")
    st.divider()
    st.caption(f"Knowledge base: {len(SAMPLE_KB)} docs")
    st.dataframe(pd.DataFrame([{"Doc": d.title, "Source": d.source} for d in SAMPLE_KB]),
                 hide_index=True, use_container_width=True)

st.subheader("Ask a question")
examples = ["How many vacation days do I get?", "How do I connect to the VPN?",
            "When is payday?", "Can I bring my dog to the office?"]
cols = st.columns(len(examples))
if "q" not in st.session_state:
    st.session_state.q = examples[0]
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        st.session_state.q = ex

question = st.text_input("Question", st.session_state.q)

if st.button("Ask the agent", type="primary"):
    ans = answer_question(question, retriever, threshold=threshold, escalation_channel=escalation_channel)

    if ans.escalated:
        st.warning(f"🙋 Escalated to {escalation_channel} (confidence {ans.confidence} < {threshold})")
    else:
        st.success(f"✅ Answered from knowledge base (confidence {ans.confidence})")

    st.markdown(ans.text)

    if ans.citations:
        st.caption("Sources: " + ", ".join(f"`{d.title}`" for d in ans.citations))

    target = escalation_channel if ans.escalated else "#general"
    posted, detail = post_to_slack(target, ans.text)
    st.caption(f"Slack: {'posted' if posted else detail}")

    with st.expander("Retrieval scores"):
        hits = retriever.search(question, k=len(SAMPLE_KB))
        st.dataframe(
            pd.DataFrame([{"Doc": h.doc.title, "Score": round(h.score, 3)} for h in hits]),
            hide_index=True, use_container_width=True,
        )
