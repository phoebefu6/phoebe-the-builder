from __future__ import annotations

import pandas as pd
import streamlit as st

from optimizer import SAMPLE_QUERIES, analyze_query

SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

st.set_page_config(page_title="SQL Optimizer Advisor", layout="wide")
st.title("SQL Optimizer Advisor")
st.caption("Slow queries, no idea why? This lints your SQL for the anti-patterns that cause full scans.")

with st.sidebar:
    st.subheader("Examples")
    pick = st.radio("Load an example", ["(write my own)"] + list(SAMPLE_QUERIES))
    st.caption("Static heuristic linter — no database connection, no query execution.")

default = SAMPLE_QUERIES.get(pick, SAMPLE_QUERIES["Slow dashboard query"])
sql = st.text_area("SQL query", value=default, height=220)

if st.button("Analyze", type="primary") or sql:
    if not sql.strip():
        st.stop()
    report = analyze_query(sql)
    score = report.score()

    c1, c2 = st.columns(2)
    c1.metric("Query health", f"{score}/100")
    c2.metric("Issues found", len(report.findings))

    if score >= 85:
        st.success("Looks healthy — no significant anti-patterns.")
    elif score >= 50:
        st.warning("Some anti-patterns worth fixing.")
    else:
        st.error("Multiple performance anti-patterns — likely slow at scale.")

    if report.findings:
        st.subheader("Findings")
        for f in sorted(report.findings, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.severity]):
            with st.expander(f"{SEV_ICON[f.severity]} {f.severity.upper()} — {f.issue}"):
                st.markdown(f"**Why it hurts:** {f.why}")
                st.markdown(f"**Fix:** {f.suggestion}")

        st.subheader("Summary table")
        st.dataframe(
            pd.DataFrame([{"severity": f.severity, "rule": f.rule, "issue": f.issue, "fix": f.suggestion}
                          for f in report.findings]),
            use_container_width=True,
        )
