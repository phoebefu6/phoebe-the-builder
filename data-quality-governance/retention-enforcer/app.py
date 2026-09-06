from __future__ import annotations

# Streamlit front-end for the Data Retention Enforcer. Upload a records CSV (or
# use the built-in sample), view/edit the retention policies, run the engine,
# and read the action plan grouped by action - past-due, approaching, within.
# The engine only PLANS; nothing is deleted here. Legal owns the policy.
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from enforcer import (
    AS_OF,
    RetentionPolicy,
    evaluate,
    make_sample_data,
    summarize,
)

st.set_page_config(page_title="Data Retention Enforcer", page_icon="🗂️", layout="wide")

# Colors match the verdict semantics: red = act now, amber = heads-up, green =
# fine, grey = governance gap. One color per meaning, so the plan reads at a glance.
VERDICT_COLOR = {
    "past_retention": "#d64545",
    "approaching": "#e0a458",
    "within_policy": "#4a9d6f",
    "no_policy": "#7a7a7a",
}
ACTION_COLOR = {
    "delete": "#d64545",
    "anonymize": "#c77dff",
    "review": "#e0a458",
    "define_policy": "#7a7a7a",
    "none": "#4a9d6f",
}

st.title("🗂️ Data Retention Enforcer")
st.caption(
    "Plan which records are past their retention purpose - explainably, and for "
    "human review. This tool never auto-deletes. Legal owns the policy."
)

# ---- Data source -----------------------------------------------------------
sample_records, sample_policies = make_sample_data()

st.sidebar.header("1. Records")
uploaded = st.sidebar.file_uploader(
    "Upload records CSV", type="csv",
    help="Needs columns: record_id, created_date, data_class",
)
if uploaded is not None:
    records = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(records)} records from CSV")
else:
    records = sample_records.copy()
    st.sidebar.info(f"Using built-in sample ({len(records)} records)")

st.sidebar.header("2. AS_OF baseline")
as_of = pd.Timestamp(
    st.sidebar.date_input(
        "Evaluate ages as of", value=AS_OF.date(),
        help="Fixed baseline for reproducible ages (defaults to the sample AS_OF).",
    )
)

# ---- Policy editor ---------------------------------------------------------
st.sidebar.header("3. Retention policies")
st.sidebar.caption("Edit the limits and actions, then re-run.")
policy_df = pd.DataFrame([
    {
        "data_class": p.data_class,
        "max_age_days": p.max_age_days,
        "action": p.action,
        "warning_days": p.warning_days,
        "basis": p.basis,
    }
    for p in sample_policies
])
edited = st.sidebar.data_editor(
    policy_df, num_rows="dynamic", use_container_width=True, key="policies",
)

policies = [
    RetentionPolicy(
        data_class=str(r["data_class"]),
        max_age_days=int(r["max_age_days"]),
        action=str(r["action"]),
        warning_days=int(r["warning_days"]),
        basis=str(r.get("basis", "")),
    )
    for _, r in edited.iterrows()
    if pd.notna(r["data_class"]) and pd.notna(r["max_age_days"])
]

# ---- Run -------------------------------------------------------------------
plan = evaluate(records, policies, as_of=as_of)

if plan.empty:
    st.warning("No records to evaluate - upload a CSV or use the sample.")
    st.stop()

# ---- Rollup ----------------------------------------------------------------
roll = summarize(plan)
st.subheader("Rollup summary")
c1, c2, c3, c4 = st.columns(4)
counts = plan["verdict"].value_counts()
c1.metric("Past retention", int(counts.get("past_retention", 0)))
c2.metric("Approaching", int(counts.get("approaching", 0)))
c3.metric("Within policy", int(counts.get("within_policy", 0)))
c4.metric("No policy", int(counts.get("no_policy", 0)))

left, right = st.columns([1, 1])
with left:
    st.dataframe(roll, use_container_width=True, hide_index=True)
with right:
    fig, ax = plt.subplots(figsize=(5, 3))
    bar_colors = [ACTION_COLOR.get(a, "#888888") for a in roll["action_due"]]
    ax.bar(roll["action_due"], roll["records"], color=bar_colors)
    ax.set_ylabel("records")
    ax.set_title("Records by action due")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)

# ---- Action plan by action -------------------------------------------------
st.subheader("Action plan")


def _style(df: pd.DataFrame):
    def _row(row):
        color = VERDICT_COLOR.get(row["verdict"], "#ffffff")
        return [f"background-color: {color}22"] * len(row)
    return df.style.apply(_row, axis=1)


show_cols = ["record_id", "data_class", "age_days", "max_age_days",
             "verdict", "action_due", "reason"]

tabs = st.tabs(["Past retention", "Approaching", "Within policy", "No policy", "All"])
verdict_for_tab = ["past_retention", "approaching", "within_policy", "no_policy", None]
for tab, v in zip(tabs, verdict_for_tab):
    with tab:
        sub = plan if v is None else plan[plan["verdict"] == v]
        if sub.empty:
            st.info("Nothing in this bucket.")
        else:
            st.dataframe(_style(sub[show_cols]), use_container_width=True,
                         hide_index=True)

# ---- Export ----------------------------------------------------------------
st.subheader("Export")
buf = io.StringIO()
plan.to_csv(buf, index=False)
st.download_button(
    "Download action plan (CSV)", buf.getvalue(),
    file_name="retention_action_plan.csv", mime="text/csv",
)
st.caption(
    "This plan is a prioritized review queue, not an instruction to delete. "
    "Confirm legal holds and business need before any disposal."
)
