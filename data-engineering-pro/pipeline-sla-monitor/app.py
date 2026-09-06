from __future__ import annotations

import pandas as pd
import streamlit as st
from sla import (
    SAMPLE_SLAS,
    evaluate,
    make_sample_runs,
    scorecard,
)

st.set_page_config(page_title="Pipeline SLA Monitor", page_icon="⏱️", layout="wide")
st.title("⏱️ Pipeline SLA Monitor")
st.caption("Declare what on-time, fast, fresh, and complete mean for each pipeline. "
           "See who kept the promise - and page before a consumer notices.")

runs = make_sample_runs()
reports = evaluate(runs, SAMPLE_SLAS)
card = scorecard(reports)

# --- Fleet scorecard -------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pipelines", card["pipelines"])
c2.metric("Healthy", card["healthy"])
c3.metric("At risk", card["at_risk"])
c4.metric("In breach", card["breach"])
c5.metric("Fleet compliance", f"{card['fleet_compliance']}%")

st.divider()

# --- Per-pipeline status ---------------------------------------------------
BADGE = {"HEALTHY": "🟢", "AT_RISK": "🟡", "BREACH": "🔴"}
rows = []
for r in sorted(reports, key=lambda x: x.compliance_pct):
    sla = SAMPLE_SLAS[r.pipeline]
    rows.append({
        "": BADGE[r.status],
        "Pipeline": r.pipeline,
        "Status": r.status,
        "Compliance": f"{r.compliance_pct}%",
        "Runs": r.total_runs,
        "Breaches": r.breach_count,
        "Land by": sla.landing_by.strftime("%H:%M"),
        "Max min": sla.max_duration_min,
        "Fresh h": sla.freshness_hours,
        "Min rows": f"{sla.min_rows:,}",
    })
st.subheader("Per-pipeline SLA status")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Breach log ------------------------------------------------------------
st.subheader("Breach log")
all_breaches = [
    {"Pipeline": b.pipeline, "Date": b.run_date.isoformat(), "Type": b.kind, "Detail": b.detail}
    for r in reports for b in r.breaches
]
if all_breaches:
    bdf = pd.DataFrame(all_breaches).sort_values(["Date", "Pipeline"])
    kinds = st.multiselect("Filter by breach type", sorted(bdf["Type"].unique()),
                           default=sorted(bdf["Type"].unique()))
    st.dataframe(bdf[bdf["Type"].isin(kinds)], use_container_width=True, hide_index=True)
    st.caption(f"{len(bdf)} breaches across the window. "
               "LATE = missed landing time · SLOW = over duration budget · "
               "STALE = no fresh good run · LOW_VOLUME = thin load · FAILED = run errored.")
else:
    st.success("No SLA breaches in the window. 🎉")

with st.expander("How compliance is computed"):
    st.markdown(
        "- Each pipeline declares 4 promises: **land-by time, max duration, freshness window, min rows.**\n"
        "- Every run is graded; a run with any breach is a strike for that date.\n"
        "- **Compliance %** = clean run-dates / total runs.\n"
        "- **Freshness** checks the newest *successful* run against `now` (latest end time in the data).\n"
        "- Status: 🟢 HEALTHY (100%) · 🟡 AT_RISK (90-99%) · 🔴 BREACH (<90% or stale)."
    )
