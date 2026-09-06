from __future__ import annotations

# Streamlit UI for the Data Access Auditor. Upload a grants CSV (or use the
# built-in sample), audit access grants for governance risks - over-privileged
# access, stale grants, sensitive-data over-exposure, orphaned grants, and
# segregation-of-duties conflicts - then triage findings by severity, each with
# a plain-English reason. Defensive least-privilege hygiene, fully offline.
import pandas as pd
import streamlit as st
from auditor import (
    AS_OF,
    EXPOSURE_MAX_USERS,
    STALE_DAYS,
    audit_grants,
    findings_frame,
    make_sample_grants,
    summarize,
)

st.set_page_config(page_title="Data Access Auditor", page_icon="🔐", layout="wide")

st.title("🔐 Data Access Auditor")
st.caption(
    "\"Who can see the PII table?\" should take seconds, not a week of digging. "
    "Upload your access grants and this flags over-privileged access, stale "
    "grants, over-exposed sensitive datasets, orphaned grants, and "
    "segregation-of-duties conflicts - each with a plain-English reason. "
    "Rule-based, offline. Findings are review candidates, not auto-revokes."
)

with st.sidebar:
    st.header("Grants data")
    up = st.file_uploader("Upload a grants CSV", type=["csv"])
    st.caption(
        "Columns: user, role, dataset, sensitivity "
        "(public/internal/confidential/restricted), permission "
        "(read/write/admin), granted_date, last_used_date."
    )
    st.markdown("or")
    use_sample = st.button("Load sample grants", use_container_width=True)
    st.divider()
    st.markdown(
        "**Rules**\n\n"
        f"- **over-privileged** - write/admin on restricted data, non-admin role\n"
        f"- **stale** - unused >= {STALE_DAYS} days\n"
        f"- **exposure** - > {EXPOSURE_MAX_USERS} users on one restricted dataset\n"
        "- **orphaned** - contractor/intern/guest on restricted data\n"
        "- **sod** - one user holds conflicting roles"
    )
    st.caption(f"Reference date for staleness (sample): {AS_OF.date()}")

if up is not None:
    df = pd.read_csv(up)
    for c in ("granted_date", "last_used_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df, source, as_of = df, up.name, None  # uploaded data -> as_of from data max
elif use_sample or "df" not in st.session_state:
    df = make_sample_grants()
    source, as_of = "sample grants (planted risks)", AS_OF
else:
    df = st.session_state["df"]
    source = st.session_state.get("source", "current")
    as_of = st.session_state.get("as_of", None)

st.session_state["df"] = df
st.session_state["source"] = source
st.session_state["as_of"] = as_of

st.write(f"**Source:** {source} - {len(df):,} grants x {df.shape[1]} columns")

findings = audit_grants(df, as_of=as_of)
summary = summarize(findings)
flat = findings_frame(findings)

total = len(flat)
highs = int((flat["severity"] == "high").sum()) if total else 0
users_flagged = flat["user"].replace("*", pd.NA).dropna().nunique() if total else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total findings", f"{total:,}")
c2.metric("High severity", f"{highs:,}")
c3.metric("Users flagged", f"{users_flagged}")

st.subheader("Triage rollup by rule")
st.dataframe(summary, use_container_width=True, hide_index=True)

if total:
    st.bar_chart(summary.set_index("rule")[["high", "medium", "low"]])

st.subheader("Findings")
if total == 0:
    st.success("No governance risks found under the current policy bar. ✅")
else:
    pick = st.multiselect(
        "Filter by severity", ["high", "medium", "low"], default=["high", "medium"]
    )
    view = flat[flat["severity"].isin(pick)] if pick else flat
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download findings CSV",
        view.to_csv(index=False).encode(),
        file_name="access_findings.csv",
        mime="text/csv",
    )

st.caption(
    "Policy bar lives in auditor.py (STALE_DAYS, EXPOSURE_MAX_USERS, "
    "PRIVILEGED_ROLES, DISALLOWED_ON_RESTRICTED, SOD_CONFLICTS) - tune to your "
    "own governance policy. Findings are hygiene signals for a human access "
    "review, not automatic revocation; expect some false positives."
)
