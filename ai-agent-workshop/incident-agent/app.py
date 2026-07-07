from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from incident_agent import RUNBOOKS, Incident, execute_runbook

st.set_page_config(page_title="Incident Response Agent", page_icon="🚨")
st.title("🚨 Incident Response Agent")
st.caption("Agent executes the runbook step-by-step so it actually gets followed, and escalates when a remediation step fails.")

with st.sidebar:
    st.subheader("Incident")
    incident_type = st.selectbox("Incident type", list(RUNBOOKS.keys()))
    severity = st.selectbox("Severity", ["P1", "P2", "P3"])
    st.info("Set ANTHROPIC_API_KEY for a Claude-written summary. Falls back to a rule-based summary otherwise.")

st.subheader("Simulated conditions")
context = {}
if incident_type == "service_down":
    context["service_healthy"] = st.checkbox("Service health check passes", value=False)
    context["restart_recovers"] = st.checkbox("Restart recovers the service", value=True)
elif incident_type == "disk_space_critical":
    context["disk_usage_pct"] = st.slider("Disk usage %", 0, 100, 95)
    context["cleanup_frees_space"] = st.checkbox("Cleanup frees sufficient space", value=True)
elif incident_type == "database_high_latency":
    context["query_latency_ms"] = st.slider("Query latency (ms)", 0, 2000, 900)
    context["kill_queries_resolves"] = st.checkbox("Killing queries resolves latency", value=True)

if st.button("Run Incident Response Agent", type="primary"):
    incident = Incident(incident_type=incident_type, severity=severity, detected_at=datetime.now(), context=context)
    with st.spinner("Agent executing runbook..."):
        report = execute_runbook(incident)

    if report.escalated:
        st.error(f"ESCALATED to {report.escalated_to} — critical remediation step failed")
    else:
        st.success("No escalation needed")

    st.subheader("Summary")
    st.write(report.summary)

    st.subheader("Runbook Execution Trace")
    status_icon = {"done": "✅", "failed": "🔴", "awaiting_human": "🟡", "skipped": "⚪"}
    df = pd.DataFrame([
        {
            "Status": f"{status_icon[e.status]} {e.status}",
            "Step": e.step.description,
            "Type": "auto" if e.step.auto else "manual",
            "Output": e.output,
        }
        for e in report.executions
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
