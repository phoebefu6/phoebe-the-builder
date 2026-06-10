from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# --- Core Logic ---

def generate_mock_sources() -> List[Dict]:
    """Generate realistic mock data sources with varying freshness."""
    sources = [
        {"name": "orders", "schema": "production", "owner": "data-eng", "sla_hours": 1},
        {"name": "customers", "schema": "production", "owner": "data-eng", "sla_hours": 6},
        {"name": "page_views", "schema": "analytics", "owner": "analytics", "sla_hours": 2},
        {"name": "transactions", "schema": "finance", "owner": "finance-eng", "sla_hours": 1},
        {"name": "inventory", "schema": "warehouse", "owner": "supply-chain", "sla_hours": 4},
        {"name": "user_sessions", "schema": "analytics", "owner": "analytics", "sla_hours": 2},
        {"name": "marketing_spend", "schema": "marketing", "owner": "marketing-ops", "sla_hours": 24},
        {"name": "support_tickets", "schema": "support", "owner": "support-eng", "sla_hours": 12},
        {"name": "product_catalog", "schema": "production", "owner": "product-eng", "sla_hours": 24},
        {"name": "employee_directory", "schema": "hr", "owner": "hr-ops", "sla_hours": 168},
    ]
    now = datetime.now()
    random.seed(42)
    for s in sources:
        hours_ago = random.uniform(0, s["sla_hours"] * 3)
        s["last_updated"] = now - timedelta(hours=hours_ago)
        s["row_count"] = random.randint(1000, 5_000_000)
        s["avg_latency_sec"] = round(random.uniform(0.5, 120), 1)
    return sources


def check_freshness(source: Dict) -> Dict:
    """Evaluate freshness status for a single source."""
    now = datetime.now()
    age_hours = (now - source["last_updated"]).total_seconds() / 3600
    sla = source["sla_hours"]

    if age_hours <= sla:
        status = "Fresh"
        color = "green"
    elif age_hours <= sla * 2:
        status = "Warning"
        color = "orange"
    else:
        status = "Stale"
        color = "red"

    return {
        "Source": f"{source['schema']}.{source['name']}",
        "Owner": source["owner"],
        "SLA (hrs)": sla,
        "Age (hrs)": round(age_hours, 1),
        "Last Updated": source["last_updated"].strftime("%Y-%m-%d %H:%M"),
        "Row Count": f"{source['row_count']:,}",
        "Status": status,
        "_color": color,
        "_age_hours": age_hours,
        "_sla_hours": sla,
    }


def generate_history(sources: List[Dict], days: int = 7) -> pd.DataFrame:
    """Generate mock freshness history for trend charts."""
    random.seed(123)
    rows = []
    now = datetime.now()
    for day_offset in range(days, -1, -1):
        ts = now - timedelta(days=day_offset)
        for s in sources:
            sla = s["sla_hours"]
            breach = random.random() < 0.15
            age = random.uniform(sla * 1.5, sla * 3) if breach else random.uniform(0, sla * 0.9)
            rows.append({
                "date": ts.strftime("%Y-%m-%d"),
                "source": f"{s['schema']}.{s['name']}",
                "age_hours": round(age, 1),
                "sla_hours": sla,
                "breached": breach,
            })
    return pd.DataFrame(rows)


# --- Streamlit UI ---

st.set_page_config(page_title="Data Freshness Monitor", page_icon="📡", layout="wide")
st.title("📡 Data Freshness Monitor")
st.caption("Track data source freshness against SLAs — catch stale data before your stakeholders do.")

sources = generate_mock_sources()
results = [check_freshness(s) for s in sources]

fresh = sum(1 for r in results if r["Status"] == "Fresh")
warning = sum(1 for r in results if r["Status"] == "Warning")
stale = sum(1 for r in results if r["Status"] == "Stale")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sources", len(results))
col2.metric("Fresh", fresh)
col3.metric("Warning", warning)
col4.metric("Stale", stale)

st.divider()

status_filter = st.multiselect("Filter by status", ["Fresh", "Warning", "Stale"], default=["Warning", "Stale"])
filtered = [r for r in results if r["Status"] in status_filter]

if filtered:
    df = pd.DataFrame(filtered)
    display_cols = ["Source", "Owner", "SLA (hrs)", "Age (hrs)", "Last Updated", "Row Count", "Status"]

    def highlight_status(val):
        colors = {"Fresh": "#2ecc71", "Warning": "#f39c12", "Stale": "#e74c3c"}
        return f"background-color: {colors.get(val, '')}; color: white; font-weight: bold"

    styled = df[display_cols].style.map(highlight_status, subset=["Status"])
    st.dataframe(styled, width="stretch", hide_index=True)
else:
    st.info("No sources match the selected filters.")

st.divider()
st.subheader("Freshness Trend (Last 7 Days)")

history_df = generate_history(sources)
daily_breaches = history_df.groupby("date")["breached"].sum().reset_index()
daily_breaches.columns = ["Date", "SLA Breaches"]
st.bar_chart(daily_breaches, x="Date", y="SLA Breaches", color="#e74c3c")

st.divider()
st.subheader("SLA Breach Details")

worst = sorted(results, key=lambda r: r["_age_hours"] / r["_sla_hours"], reverse=True)
for r in worst[:3]:
    ratio = r["_age_hours"] / r["_sla_hours"]
    if ratio > 1:
        st.warning(f"**{r['Source']}** — {r['Age (hrs)']}h old vs {r['SLA (hrs)']}h SLA ({ratio:.1f}x over). Owner: {r['Owner']}")
