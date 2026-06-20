from __future__ import annotations

"""Streamlit UI: a live-feeling database health dashboard.

Pulls a metrics snapshot (simulated by default), scores every metric RAG, and
shows an overall health gauge so on-call engineers get DB visibility at a glance.
"""

import pandas as pd
import streamlit as st

from health import build_report, overall_health, simulate_metrics

st.set_page_config(page_title="DB Health Dashboard", page_icon="🩺", layout="wide")

STATUS_EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴"}
GRADE_COLOR = {"Healthy": "normal", "Watch": "off", "Critical": "inverse"}

st.title("🩺 Database Health Dashboard")
st.caption("One screen of DB visibility - cache hits, latency, locks, replication lag - scored red/amber/green so problems surface before users notice.")

with st.sidebar:
    st.header("Snapshot")
    stressed = st.toggle("Simulate DB under load", value=False, help="Flip to degraded numbers to see red/amber states.")
    seed = st.number_input("Random seed", value=42, step=1)
    st.caption("In production, swap `simulate_metrics` for live `pg_stat_*` queries.")

metrics = simulate_metrics(seed=int(seed), stressed=stressed)
report = build_report(metrics)
summary = overall_health(report)

top = st.columns([1, 1, 1, 1])
top[0].metric("Health score", f"{summary['score']} / 100")
top[1].metric("Grade", summary["grade"], delta_color=GRADE_COLOR.get(str(summary["grade"]), "normal"))
top[2].metric("🔴 Red", summary["reds"])
top[3].metric("🟡 Amber", summary["ambers"])

if summary["grade"] == "Critical":
    st.error("Database health is **Critical** - investigate red metrics now.")
elif summary["grade"] == "Watch":
    st.warning("Database health is in **Watch** - some metrics drifting.")
else:
    st.success("Database health is **Healthy**.")

st.subheader("Metrics")
display = report.copy()
display["status"] = display["status"].map(lambda s: f"{STATUS_EMOJI[s]} {s}")
display["value"] = display.apply(
    lambda r: f"{r['value']:g}{(' ' + r['unit']) if r['unit'] else ''}", axis=1
)
st.dataframe(
    display[["metric", "value", "status"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Score by metric")
chart_df = report[["metric", "points"]].set_index("metric")
st.bar_chart(chart_df, height=320)

reds = report[report["status"] == "red"]
if not reds.empty:
    with st.expander("⚠ What to do about the red metrics", expanded=True):
        for _, row in reds.iterrows():
            st.markdown(f"- **{row['metric']}** = {row['value']:g}{row['unit']} - past the red threshold; page the on-call DBA.")

st.caption("Thresholds live in `health.py` → `METRIC_SPECS`. Tune them to your SLOs.")
