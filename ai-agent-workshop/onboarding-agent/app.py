from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from onboarding import DEFAULT_CHECKLIST, NewHire, evaluate_onboarding, progress_summary

st.set_page_config(page_title="Onboarding Checklist Agent", page_icon="🧭")
st.title("🧭 Onboarding Checklist Agent")
st.caption("Multi-step agent walks a new hire's checklist, resolves dependencies, and flags overdue/blocked steps so nothing slips.")

with st.sidebar:
    st.subheader("New Hire")
    name = st.text_input("Name", value="Alex Kim")
    role = st.selectbox("Role", ["Engineer", "Data Scientist", "Sales", "Other"])
    start_date = st.date_input("Start date", value=date(2026, 6, 1))
    today = st.date_input("Evaluate as of", value=date(2026, 7, 5))
    st.info("Set ANTHROPIC_API_KEY for Claude-written nudges. Falls back to rule-based nudges otherwise.")

st.subheader("Mark completed steps")
completed_ids = set()
cols = st.columns(2)
for i, item in enumerate(DEFAULT_CHECKLIST):
    if item.roles and role not in item.roles:
        continue
    with cols[i % 2]:
        default_checked = item.item_id in ("it_setup", "hr_paperwork", "welcome_meeting")
        if st.checkbox(f"[{item.phase}] {item.title}", value=default_checked, key=item.item_id):
            completed_ids.add(item.item_id)

if st.button("Run Onboarding Agent", type="primary"):
    hire = NewHire(name=name, role=role, start_date=start_date, completed_ids=completed_ids)
    statuses = evaluate_onboarding(hire, today=today)
    summary = progress_summary(statuses)

    total = len(statuses)
    pct_complete = summary["completed"] / total if total else 0
    st.progress(pct_complete, text=f"{summary['completed']}/{total} steps complete ({pct_complete:.0%})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overdue", summary["overdue"])
    m2.metric("Due soon", summary["due_soon"])
    m3.metric("Blocked", summary["blocked"])
    m4.metric("Upcoming", summary["upcoming"])

    status_icon = {"completed": "✅", "overdue": "🔴", "due_soon": "🟡", "blocked": "⛔", "upcoming": "⚪"}
    status_order = {"overdue": 0, "blocked": 1, "due_soon": 2, "upcoming": 3, "completed": 4}
    sorted_statuses = sorted(statuses, key=lambda s: status_order[s.status])

    df = pd.DataFrame([
        {
            "Status": f"{status_icon[s.status]} {s.status}",
            "Phase": s.item.phase,
            "Step": s.item.title,
            "Expected by day": s.item.expected_by_day,
            "Nudge": s.nudge,
        }
        for s in sorted_statuses
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
