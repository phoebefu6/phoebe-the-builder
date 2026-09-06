from __future__ import annotations

import streamlit as st
from extractor import SAMPLE_CONTRACT, extract_text_from_pdf, review_contract

RISK_COLOR = {"high": "🔴", "medium": "🟠", "low": "🟢"}

st.set_page_config(page_title="Contract Clause Extractor", layout="wide")
st.title("Contract Clause Extractor")
st.caption("Turn a 3-day legal read into a 30-second clause + risk triage.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using heuristic extraction (regex clause cues).")
    uploaded = st.file_uploader("Upload a contract PDF", type=["pdf"])
    use_sample = st.checkbox("Use sample contract", value=True)

if uploaded is not None:
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        contract = extract_text_from_pdf(tmp_path)
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()
elif use_sample:
    contract = st.text_area("Contract text", value=SAMPLE_CONTRACT, height=300)
else:
    contract = st.text_area("Paste your contract text", height=300)

if st.button("Review Contract", type="primary"):
    if not contract.strip():
        st.warning("Add a contract first.")
        st.stop()

    with st.spinner("Reviewing..."):
        try:
            review = review_contract(contract, api_key=api_key or None)
        except Exception as e:
            st.error(f"Review failed: {e}")
            st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Parties", " ↔ ".join(review.parties) if review.parties else "—")
    c2.metric("Effective date", review.effective_date or "—")
    counts = review.risk_counts()
    c3.metric("High-risk clauses", counts["high"])

    if review.missing_clauses:
        st.warning("Missing expected clauses: " + ", ".join(review.missing_clauses))
    else:
        st.success("All expected clauses present.")

    st.subheader(f"Extracted clauses ({len(review.clauses)})")
    for c in review.clauses:
        with st.expander(f"{RISK_COLOR.get(c.risk, '⚪')} {c.clause_type} — {c.risk.upper()} risk"):
            st.write(c.text)
            if c.note:
                st.caption(f"Note: {c.note}")
