"""Streamlit UI: verdict first, then the rows it could not decide.

The undecidable rows are shown before the successful ones on purpose. A
normaliser that hands you a full column of UTC values has already made the
decisions the input did not contain.
"""

from __future__ import annotations

import datetime as dt
import io
from typing import Any, Dict, List
from zoneinfo import ZoneInfo, available_timezones

import pandas as pd
import streamlit as st
from tznorm import (
    UTC,
    audit,
    build_session_log,
    classify,
    ground_truth,
    local_day,
    normalize,
    resolve,
    tzdata_version,
    utc_day,
)

st.set_page_config(page_title="Timezone normalizer", layout="wide")

ZONES = sorted(available_timezones())


def readings_frame(rows: List[Dict[str, Any]], readings) -> pd.DataFrame:
    out = []
    for row, r in zip(rows, readings):
        out.append(
            {
                "session": row.get("session_id", ""),
                "office": row.get("office", ""),
                "zone": r.zone,
                "local_ts": r.raw,
                "status": r.status,
                "utc": "" if r.utc is None else r.utc.strftime("%Y-%m-%d %H:%M %Z"),
                "offset": "" if r.offset is None else f"{r.offset_hours():+.2f}",
                "utc_day": str(utc_day(r) or ""),
                "local_day": str(local_day(r) or ""),
                "note": r.note,
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------

st.title("Timezone normalizer")
st.caption(
    "A local wall-clock timestamp is not a point in time. Twice a year it is two points "
    "or none - and `zoneinfo` returns an answer either way."
)

with st.sidebar:
    st.header("Input")
    source = st.radio("Data", ["Sample session log", "Paste your own"])
    with_api = st.checkbox(
        "include the partner feed that sends offsets", value=False, help="the control group"
    )

    st.header("Policy")
    ambiguous = st.selectbox(
        "When a wall time occurs twice",
        ["flag", "earlier", "later", "raise"],
        help="`earlier` is fold=0, which is what pandas and a bare `.replace(tzinfo=)` do",
    )
    nonexistent = st.selectbox(
        "When a wall time never happened", ["flag", "shift_forward", "raise"]
    )

    st.header("Single value")
    probe_ts = st.text_input("Wall clock", "2024-11-03 01:30")
    probe_zone = st.selectbox(
        "Zone", ZONES, index=ZONES.index("America/New_York") if "America/New_York" in ZONES else 0
    )

    st.caption(f"Resolved against **{tzdata_version()}**")

if source == "Sample session log":
    rows = build_session_log(with_partner_feed=with_api)
else:
    default = "local_ts,zone\n2024-11-03 01:30,America/New_York\n2024-03-10 02:30,America/New_York"
    text = st.text_area("CSV with a `local_ts` and a `zone` column", default, height=140)
    try:
        rows = pd.read_csv(io.StringIO(text)).to_dict("records")
    except Exception as exc:  # noqa: BLE001 - surface any paste error
        st.error(f"Could not read that: {exc}")
        st.stop()
    if not rows or "local_ts" not in rows[0] or "zone" not in rows[0]:
        st.error("Needs both a `local_ts` and a `zone` column.")
        st.stop()

# ---- single-value probe ---------------------------------------------------

st.subheader("One timestamp")
kind = None
try:
    parsed = dt.datetime.fromisoformat(probe_ts.strip().replace(" ", "T", 1))
    kind = "offset carried" if parsed.tzinfo else classify(parsed, probe_zone)
except Exception:  # noqa: BLE001 - a free-text box
    kind = "unparsed"

cols = st.columns([1, 3])
cols[0].metric("classification", kind)
if kind == "ambiguous":
    tz = ZoneInfo(probe_zone)
    naive = dt.datetime.fromisoformat(probe_ts.strip().replace(" ", "T", 1))
    a = naive.replace(tzinfo=tz, fold=0).astimezone(UTC)
    b = naive.replace(tzinfo=tz, fold=1).astimezone(UTC)
    cols[1].error(
        f"**{probe_ts}** happens twice in {probe_zone}: **{a:%H:%M UTC}** and **{b:%H:%M UTC}**, "
        f"{(b - a).total_seconds() / 60:.0f} minutes apart. Python picks the first one by "
        f"default and raises nothing."
    )
elif kind == "nonexistent":
    tz = ZoneInfo(probe_zone)
    naive = dt.datetime.fromisoformat(probe_ts.strip().replace(" ", "T", 1))
    back = naive.replace(tzinfo=tz).astimezone(UTC).astimezone(tz).replace(tzinfo=None)
    cols[1].error(
        f"**{probe_ts}** never happened in {probe_zone} - the clock skipped it. Converting it "
        f"to UTC and back gives **{back:%Y-%m-%d %H:%M}**, a different wall time from the one "
        f"you started with. That round trip is the only reliable test."
    )
elif kind == "offset carried":
    cols[1].success(
        "This value carries its own offset, so the instant is pinned. It still cannot answer "
        "'same local time next month' - that needs the zone."
    )
elif kind == "ok":
    r = resolve(probe_ts, probe_zone)
    cols[1].success(f"Unambiguous: **{r.utc:%Y-%m-%d %H:%M UTC}** (offset {r.offset_hours():+.2f}h)")
else:
    cols[1].warning("Could not parse that as a date.")

# ---- verdict --------------------------------------------------------------

st.subheader("The column")
report = audit(rows)
if report.verdict.startswith("NOT SAFE"):
    st.error(report.verdict)
elif report.verdict.startswith("CLEAN"):
    st.success(report.verdict)
else:
    st.warning(report.verdict)
for finding in report.findings:
    (st.warning if finding.startswith("WARNING") else st.info)(finding)

readings = normalize(rows, ambiguous=ambiguous, nonexistent=nonexistent)
df = readings_frame(rows, readings)

undecided = df[df["utc"] == ""]
if len(undecided):
    st.subheader(f"{len(undecided)} row(s) the input does not determine")
    st.dataframe(undecided, use_container_width=True)

st.subheader("Normalized")
st.dataframe(df, use_container_width=True)

# ---- day bucketing --------------------------------------------------------

if "amount" in (rows[0] if rows else {}):
    st.subheader("Which day did it happen on")
    utc_rev: Dict[str, float] = {}
    local_rev: Dict[str, float] = {}
    for row, r in zip(rows, readings):
        u, loc = utc_day(r), local_day(r)
        if u is None or loc is None:
            continue
        utc_rev[str(u)] = utc_rev.get(str(u), 0.0) + float(row.get("amount", 0) or 0)
        local_rev[str(loc)] = local_rev.get(str(loc), 0.0) + float(row.get("amount", 0) or 0)
    days = sorted(set(list(utc_rev) + list(local_rev)))
    comp = pd.DataFrame(
        {
            "day": days,
            "by UTC day": [utc_rev.get(d, 0.0) for d in days],
            "by local day": [local_rev.get(d, 0.0) for d in days],
        }
    )
    comp["delta"] = comp["by local day"] - comp["by UTC day"]
    st.dataframe(comp, use_container_width=True)
    st.caption(
        "Neither column is wrong; they answer different questions. The totals reconcile "
        "exactly and only the boundaries move, which is why this one survives audits."
    )

# ---- ground truth, when we have it ----------------------------------------

if source == "Sample session log":
    truth = ground_truth()
    by: Dict[str, Dict[str, Any]] = {}
    for row, r in zip(rows, readings):
        by.setdefault(row["session_id"], {})[row["event"]] = r
    recs = []
    for sid, ev in sorted(by.items()):
        if "open" not in ev or "close" not in ev:
            continue
        base = sid.replace("-api", "")
        t_open, t_close = truth.get((base, "open")), truth.get((base, "close"))
        got = (
            None
            if ev["open"].utc is None or ev["close"].utc is None
            else (ev["close"].utc - ev["open"].utc).total_seconds() / 60
        )
        real = None if t_open is None or t_close is None else (t_close - t_open).total_seconds() / 60
        recs.append(
            {
                "session": sid,
                "recovered (min)": got,
                "truth (min)": real,
                "error": None if got is None or real is None else got - real,
            }
        )
    st.subheader("Recovered duration vs the instants that actually happened")
    st.caption(
        "The sample's local strings are rendered *from* true UTC instants, so this is a real "
        "recovery error rather than two guesses compared. Turn on the partner feed in the "
        "sidebar to see the same sessions arrive decidable."
    )
    st.dataframe(pd.DataFrame(recs), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download the normalized column as CSV", csv, "normalized.csv", "text/csv")

st.caption(
    "Store the zone, not the offset, for anything about a clock that has not run yet - "
    "reminders, SLAs, business hours, batch schedules. Store the offset alongside the wall "
    "clock for anything that already happened. Storing neither is the default, and it is "
    "the one case that cannot be repaired later."
)
