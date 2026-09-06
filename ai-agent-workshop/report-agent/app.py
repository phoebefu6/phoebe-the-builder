from __future__ import annotations

import pandas as pd
import streamlit as st
from report_agent import SAMPLE_CONTEXT, Metric, ReportContext, generate_report

st.set_page_config(page_title="Report Generation Agent", page_icon="📄")
st.title("📄 Report Generation Agent")
st.caption(
    "A team of role-specialized agents each draft one section from your metrics, "
    "then a coordinator assembles a ready-to-send monthly report."
)

with st.sidebar:
    st.subheader("Report")
    title = st.text_input("Title", SAMPLE_CONTEXT.title)
    period = st.text_input("Period", SAMPLE_CONTEXT.period)
    notes = st.text_area("Context notes (optional)", SAMPLE_CONTEXT.notes)
    st.info("Set ANTHROPIC_API_KEY to have Claude polish each section. Falls back to rule-based prose otherwise.")

st.subheader("Metrics")
st.caption("Edit the table, then generate. `higher_is_better` controls whether an increase counts as a win.")

default_df = pd.DataFrame(
    [
        {"name": m.name, "value": m.value, "prior": m.prior, "unit": m.unit, "higher_is_better": m.higher_is_better}
        for m in SAMPLE_CONTEXT.metrics
    ]
)
edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True, hide_index=True)

if st.button("Generate Report", type="primary"):
    metrics = []
    for _, row in edited.iterrows():
        if not str(row["name"]).strip():
            continue
        metrics.append(
            Metric(
                name=str(row["name"]),
                value=float(row["value"]),
                prior=float(row["prior"]),
                unit=str(row["unit"]) if row["unit"] is not None else "",
                higher_is_better=bool(row["higher_is_better"]),
            )
        )

    if not metrics:
        st.warning("Add at least one metric with a name.")
    else:
        ctx = ReportContext(title=title, period=period, metrics=metrics, notes=notes)
        with st.spinner("Agents drafting sections..."):
            report = generate_report(ctx)
        md = report.to_markdown()
        st.success(f"Report generated — {len(report.sections)} sections by {len(report.sections)} agents.")
        st.download_button("Download report.md", md, file_name="report.md", mime="text/markdown")
        st.markdown("---")
        st.markdown(md)
