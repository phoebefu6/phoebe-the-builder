from __future__ import annotations

import pandas as pd
import streamlit as st
from catalog import sample_catalog

SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}

st.set_page_config(page_title="Metric Catalog & Ownership", layout="wide")
st.title("Metric Catalog & Ownership")
st.caption("Which metrics exist, who owns them, and which ones nobody is minding?")

cat = sample_catalog()

with st.sidebar:
    st.subheader("Governance")
    stale_days = st.slider("Stale after (days)", 30, 365, 90)

health = cat.health(stale_days=stale_days)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Metrics", health["total"])
c2.metric("Ownership", f"{health['owned_pct']}%")
c3.metric("Governance issues", health["issues"])
c4.metric("High severity", health["high_issues"])

st.subheader("Governance issues")
issues = cat.governance_issues(stale_days=stale_days)
if issues:
    for i in sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.severity]):
        st.markdown(f"{SEV_ICON[i.severity]} **{i.metric}** — {i.message}")
else:
    st.success("No governance issues.")

st.divider()
st.subheader("Catalog")
fc1, fc2, fc3 = st.columns(3)
q = fc1.text_input("Search", value="")
team = fc2.selectbox("Team", ["(all)"] + sorted({m.team for m in cat.metrics.values() if m.team}))
tier = fc3.selectbox("Tier", ["(all)", 1, 2, 3])

results = cat.search(
    query=q,
    team="" if team == "(all)" else team,
    tier=None if tier == "(all)" else int(tier),
)
rows = []
for m in results:
    rows.append({
        "metric": m.name,
        "tier": m.tier,
        "owner": m.owner or "⚠️ none",
        "team": m.team,
        "status": m.status,
        "definition": m.definition,
        "depends_on": ", ".join(m.depends_on),
        "last_reviewed": m.last_reviewed or "never",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Tier distribution")
tier_df = pd.DataFrame(
    [{"tier": f"Tier {k}", "metrics": v} for k, v in health["by_tier"].items()]
).set_index("tier")
st.bar_chart(tier_df)
