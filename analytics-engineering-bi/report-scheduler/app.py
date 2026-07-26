from __future__ import annotations

from datetime import datetime

import streamlit as st

from scheduler import (
    SAMPLE_REPORTS,
    ScheduledReport,
    build_send_plan,
    describe_cron,
    next_runs,
    render_report,
    sample_dataframe,
)

st.set_page_config(page_title="Scheduled Report Sender", layout="wide")
st.title("Scheduled Report Sender")
st.caption("Stop assembling the same weekly report by hand — define it once, preview every scheduled send.")

st.info("This is a **dry run**: it renders reports and previews the send schedule. It never sends email — wire the render step to your mail provider when you deploy.")

df = sample_dataframe()
now = datetime.now()

with st.sidebar:
    st.subheader("Schedule preview")
    horizon = st.slider("Preview horizon (days)", 1, 30, 7)

st.subheader("Configured reports")
for r in SAMPLE_REPORTS:
    with st.expander(f"📄 {r.name} — {describe_cron(r.cron)}"):
        st.markdown(f"**Recipients:** {', '.join(r.recipients)}")
        st.markdown(f"**Metric:** {r.agg}({r.metric}) by {r.group_by}")
        st.caption("Next 3 fires: " + ", ".join(d.strftime("%a %m-%d %H:%M") for d in next_runs(r.cron, now, 3)))

st.subheader(f"Send plan — next {horizon} days")
plan = build_send_plan(SAMPLE_REPORTS, now, horizon)
if plan.empty:
    st.write("No sends scheduled in this window.")
else:
    st.dataframe(plan, use_container_width=True)
    st.metric("Total sends in window", len(plan))

st.subheader("Report preview")
pick = st.selectbox("Preview which report", [r.name for r in SAMPLE_REPORTS])
report = next(r for r in SAMPLE_REPORTS if r.name == pick)
st.markdown(render_report(report, df, now))

st.divider()
st.subheader("Add a custom report")
c1, c2, c3 = st.columns(3)
name = c1.text_input("Name", value="Ad-hoc Report")
cron = c2.text_input("Cron (min hour dom mon dow)", value="0 9 * * 1")
metric = c3.selectbox("Metric", [c for c in df.columns if df[c].dtype != object])
group_by = c1.selectbox("Group by", [c for c in df.columns if df[c].dtype == object])
if st.button("Preview custom report"):
    custom = ScheduledReport(name=name, cron=cron, recipients=["you@acme.com"], metric=metric, group_by=group_by)
    st.caption("Schedule: " + describe_cron(cron))
    st.markdown(render_report(custom, df, now))
