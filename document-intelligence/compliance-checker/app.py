from __future__ import annotations

import pandas as pd
import streamlit as st
from checker import (
    DEFAULT_RULES,
    SAMPLE_POLICY_BAD,
    SAMPLE_POLICY_GOOD,
    run_compliance_check,
)

SEV_ICON = {"critical": "🟥", "high": "🟧", "medium": "🟨", "low": "🟦"}

st.set_page_config(page_title="Compliance Checker", layout="wide")
st.title("Compliance Checker")
st.caption("Run a document against a policy ruleset before the auditor does.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key (optional)", type="password")
    if not api_key:
        st.info("No key set — using the deterministic regex rules engine.")
    st.subheader("Active ruleset")
    for r in DEFAULT_RULES:
        st.caption(f"{SEV_ICON[r.severity]} `{r.id}` — {r.description} ({r.mode})")

sample = st.radio("Sample document", ["Good policy", "Bad policy", "Paste my own"], horizontal=True)
if sample == "Good policy":
    text = st.text_area("Document", value=SAMPLE_POLICY_GOOD, height=280)
elif sample == "Bad policy":
    text = st.text_area("Document", value=SAMPLE_POLICY_BAD, height=280)
else:
    text = st.text_area("Paste a policy / doc to check", height=280)

if st.button("Check compliance", type="primary"):
    if not text.strip():
        st.warning("Add a document first.")
        st.stop()

    with st.spinner("Checking..."):
        try:
            report = run_compliance_check(text, api_key=api_key or None)
        except Exception as e:
            st.error(f"Check failed: {e}")
            st.stop()

    score = report.score()
    counts = report.severity_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Compliance score", f"{score}/100")
    c2.metric("Violations", len(report.violations))
    c3.metric("Critical", counts["critical"])

    if score == 100:
        st.success("No violations found against the active ruleset.")
    elif counts["critical"]:
        st.error("Critical violation(s) present — fix before shipping.")
    else:
        st.warning("Violations found — review below.")

    st.subheader("Findings")
    rows = [
        {
            "Rule": f.rule_id,
            "Category": f.category,
            "Severity": f.severity,
            "Status": "❌ violation" if f.status == "violation" else "✅ pass",
            "Requirement": f.description,
            "Evidence": f.evidence,
        }
        for f in report.findings
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
