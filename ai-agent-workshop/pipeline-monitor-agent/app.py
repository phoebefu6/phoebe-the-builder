from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from monitor import PipelineRun, monitor_runs

st.set_page_config(page_title="Pipeline Monitor Agent", page_icon="📡")
st.title("📡 Data Pipeline Monitor Agent")
st.caption("Agent watches pipeline run history for failures, slow runs, row-count drops, and silent staleness — before users notice.")


def sample_runs() -> list[PipelineRun]:
    base = datetime.now().replace(minute=0, second=0, microsecond=0)
    runs = []
    for i in range(6):
        runs.append(PipelineRun("etl_orders", "success", base - timedelta(hours=(6 - i)), duration_seconds=120 + i * 2, rows_processed=10000 + i * 15, expected_interval_minutes=60))
    runs.append(PipelineRun("etl_orders", "success", base, duration_seconds=410, rows_processed=9600, expected_interval_minutes=60))

    for i in range(3):
        runs.append(PipelineRun("etl_users", "success", base - timedelta(hours=(6 - i) * 2), duration_seconds=60, rows_processed=500, expected_interval_minutes=120))
    runs.append(PipelineRun("etl_users", "failed", base - timedelta(hours=1), duration_seconds=15, expected_interval_minutes=120))

    runs.append(PipelineRun("etl_inventory", "success", base - timedelta(hours=10), duration_seconds=90, rows_processed=2000, expected_interval_minutes=60))
    return runs


with st.sidebar:
    st.subheader("Settings")
    st.info("Set ANTHROPIC_API_KEY for Claude-written recommended actions. Falls back to rule-based actions otherwise.")
    use_sample = st.checkbox("Use sample pipeline history", value=True)

if use_sample:
    runs = sample_runs()
    now = runs[-1].start_time + timedelta(hours=2, minutes=30)
else:
    st.write("Upload a CSV with columns: job_name, status, start_time, duration_seconds, rows_processed, expected_interval_minutes")
    uploaded = st.file_uploader("Pipeline run log CSV", type="csv")
    runs = []
    now = datetime.now()
    if uploaded:
        df = pd.read_csv(uploaded, parse_dates=["start_time"])
        runs = [
            PipelineRun(
                job_name=row["job_name"],
                status=row["status"],
                start_time=row["start_time"].to_pydatetime(),
                duration_seconds=float(row["duration_seconds"]),
                rows_processed=int(row["rows_processed"]) if pd.notna(row.get("rows_processed")) else None,
                expected_interval_minutes=int(row.get("expected_interval_minutes", 60)),
            )
            for _, row in df.iterrows()
        ]

st.subheader("Pipeline Run History")
if runs:
    st.dataframe(
        pd.DataFrame([{"job": r.job_name, "status": r.status, "start_time": r.start_time, "duration_s": r.duration_seconds, "rows": r.rows_processed} for r in runs]),
        use_container_width=True,
        hide_index=True,
    )

if st.button("Run Monitor Agent", type="primary"):
    if not runs:
        st.error("No pipeline runs to check.")
    else:
        with st.spinner("Agent scanning run history for anomalies..."):
            alerts = monitor_runs(runs, now=now)

        severity_icon = {"critical": "🔴", "warning": "🟡", "info": "⚪"}
        st.subheader(f"Alerts ({len(alerts)})")
        if alerts:
            df = pd.DataFrame([
                {
                    "Severity": f"{severity_icon[a.severity]} {a.severity}",
                    "Job": a.job_name,
                    "Reason": a.reason,
                    "Recommended Action": a.recommended_action,
                }
                for a in alerts
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.success("All pipelines healthy — no alerts.")
