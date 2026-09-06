from __future__ import annotations

"""Streamlit UI: paste or upload logs, get structured records, a level
breakdown, top errors, and a live alert-rule panel.
"""

import pandas as pd
import streamlit as st
from parser import AlertRule, evaluate_rules, parse_lines, summarize

st.set_page_config(page_title="Log Parser & Alerter", page_icon="📜", layout="wide")

SAMPLE = """2026-06-20 10:15:01 INFO service started on :8000
2026-06-20 10:15:04 INFO request /health 200
2026-06-20 10:16:22 WARNING slow query took 1200ms
2026-06-20 10:16:23 ERROR database connection timeout
{"timestamp":"2026-06-20T10:16:25","level":"ERROR","message":"database connection timeout"}
2026-06-20 10:16:31 ERROR database connection timeout
127.0.0.1 - - [20/Jun/2026:10:17:00 +0000] "GET /api/orders HTTP/1.1" 500 1320
2026-06-20 10:17:10 CRITICAL out of memory, worker killed
2026-06-20 10:18:00 INFO request /health 200
"""

st.title("📜 Log Parser & Alerter")
st.caption("Stop grepping logs by hand. Paste mixed-format logs (JSON, syslog, access logs) and get structured records, a severity breakdown, top errors, and alert rules that fire on spikes.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload a .log / .txt file", type=["log", "txt"])
    use_sample = st.checkbox("Use sample logs", value=uploaded is None)
    st.divider()
    st.header("Alert rule")
    rule_level = st.selectbox("Fire at / above", ["WARNING", "ERROR", "CRITICAL"], index=1)
    rule_threshold = st.number_input("Threshold (matching lines)", min_value=1, value=3, step=1)
    rule_contains = st.text_input("Message contains (optional)", value="timeout")

if uploaded is not None:
    text = uploaded.read().decode("utf-8", errors="replace")
elif use_sample:
    text = SAMPLE
else:
    st.info("Upload a log file or tick **Use sample logs** to begin.")
    st.stop()

records = parse_lines(text.splitlines())
summary = summarize(records)

c = st.columns(4)
c[0].metric("Lines parsed", summary["total"])
c[1].metric("Errors+", summary["error_count"])
c[2].metric("Distinct levels", len(summary["by_level"]))
c[3].metric("Top error count", summary["top_errors"][0][1] if summary["top_errors"] else 0)

# Alert rule evaluation.
rule = AlertRule(
    name=f"{rule_level}+ containing '{rule_contains}'" if rule_contains else f"{rule_level}+",
    min_level=rule_level,
    threshold=int(rule_threshold),
    contains=rule_contains or None,
)
result = evaluate_rules(records, [rule])[0]
if result["fired"]:
    st.error(f"🚨 ALERT FIRED - **{result['rule']}**: {result['matches']} matches "
             f"(threshold {result['threshold']}). Sample: _{result['sample']}_")
else:
    st.success(f"✅ Quiet - **{result['rule']}**: {result['matches']} matches, under threshold {result['threshold']}.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Levels")
    lvl_df = pd.DataFrame(list(summary["by_level"].items()), columns=["level", "count"])
    st.bar_chart(lvl_df.set_index("level"), height=280)

with right:
    st.subheader("Top errors")
    if summary["top_errors"]:
        st.dataframe(
            pd.DataFrame(summary["top_errors"], columns=["message", "count"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No errors found. 🎉")

st.subheader("Parsed records")
rec_df = pd.DataFrame(
    [{"level": r.level, "severity": r.severity, "timestamp": r.timestamp,
      "format": r.source_format, "message": r.message} for r in records]
)
st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.caption("Parsing core lives in `parser.py` - reusable as an Observability app on the platform shell.")
