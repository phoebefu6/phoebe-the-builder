"""Streamlit front end: paste a cron line, name a zone, see when it really runs.

Run:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
import streamlit as st

import cron as C
from evidence import SAMPLE

ZONES = [
    "UTC",
    "Europe/London",
    "Europe/Berlin",
    "America/New_York",
    "America/Chicago",
    "America/Los_Angeles",
    "Asia/Singapore",
    "Asia/Kolkata",
    "Australia/Sydney",
    "Pacific/Auckland",
]

SEV_ICON = {C.MISREAD: "🔴", C.TIMING: "🟠", C.PORTABILITY: "🔵"}
SEV_HELP = {
    C.MISREAD: "means something other than it reads",
    C.TIMING: "fires at an instant you probably did not intend",
    C.PORTABILITY: "means something else on another scheduler",
}

st.set_page_config(page_title="Cron Explainer", page_icon="🕰️", layout="wide")
st.title("🕰️ Cron Explainer")
st.caption(
    "Reads the line the way cron reads it - the union day rule, steps that do not "
    "tile their field, and the two days a year the local clock is not 24 hours long."
)

left, right = st.columns([3, 2])
with left:
    expr = st.text_input("Cron expression", value="0 0 13 * 5")
with right:
    tz_name = st.selectbox("Time zone the host keeps", ZONES, index=1)

c1, c2 = st.columns([3, 2])
with c1:
    utc_sched = st.checkbox(
        "Run it on a UTC scheduler (GitHub Actions, EventBridge, Kubernetes CronJob)",
        value=False,
    )
with c2:
    count = st.slider("Fire times to show", 5, 40, 12)

st.divider()

try:
    c = C.parse(expr)
except C.CronError as e:
    st.error(f"Cannot parse: {e}")
    st.stop()

st.subheader("What it means")
st.markdown(C.describe(c))
if c.union_day_rule:
    st.warning(
        "Both day fields are restricted, so cron takes the **union**. This is the "
        "single most common cron misreading and it is specified behaviour, not a bug."
    )

st.subheader("Findings")
findings: List[C.Finding] = C.audit(c, tz_name, datetime(datetime.now().year, 1, 1))
if not findings:
    st.success("No findings. This expression means what it reads in this zone.")
else:
    for f in findings:
        with st.container(border=True):
            st.markdown(
                f"{SEV_ICON[f.severity]} **{f.code}** &nbsp; "
                f"`{f.severity.lower()}` - {SEV_HELP[f.severity]}"
            )
            st.markdown(f.message)
            if f.detail:
                st.caption(f.detail)

st.subheader("Next fire times")
start = datetime.now().replace(second=0, microsecond=0)
got = C.fires(c, start, count, tz_name, utc_scheduler=utc_sched)
if not got:
    st.error(
        "No fire time within five years. Check the day/month combination - some "
        "dates never occur."
    )
else:
    tz = C._zone(tz_name)
    rows = []
    for f in got:
        rows.append(
            {
                "wall clock": f"{f.local:%Y-%m-%d %H:%M}",
                "actual instant (UTC)": (
                    f"{f.instant:%Y-%m-%d %H:%M}" if f.instant else "— never runs —"
                ),
                "local": (
                    f"{f.instant.astimezone(tz):%H:%M %Z}" if f.instant else "—"
                ),
                "DST": {"normal": "", "skipped": "⚠ skipped", "repeated": "⚠ repeated"}[
                    f.kind
                ],
                "note": f.note,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    odd = [f for f in got if f.kind != C.NORMAL]
    if odd:
        st.info(
            f"{len(odd)} of the next {len(got)} fire times land on a wall clock that "
            "either does not exist or happens twice. Whether the job runs, runs once "
            "or runs twice is the scheduler's choice."
        )

with st.expander("A crontab where every line is valid and most of them are wrong"):
    rows = []
    for e, purpose in SAMPLE:
        cc = C.parse(e)
        fs = C.audit(cc, tz_name, datetime(datetime.now().year, 1, 1))
        rows.append(
            {
                "expression": e,
                "what someone thought it did": purpose,
                "findings": len(fs),
                "codes": ", ".join(f.code for f in fs) or "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Audited against {tz_name}. Switch the zone above and the counts move - the "
        "findings are computed from the time line, not matched against the text."
    )
