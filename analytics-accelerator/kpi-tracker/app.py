from __future__ import annotations

"""Streamlit UI: a self-serve Monday KPI scorecard.

Upload a long-format CSV (date, metric, value), set targets, and read the same
exec answer every week - latest value, WoW/MoM deltas, trend, and RAG vs target -
without rebuilding the pull by hand.
"""

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from kpi import build_scorecard, scorecard_summary  # noqa: E402

st.set_page_config(page_title="KPI Tracker", page_icon="📊", layout="wide")

RAG_EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴", "none": "⚪"}

st.title("📊 KPI Tracker")
st.caption("Stop rebuilding the Monday metrics email. Upload a metric time series, set "
           "targets once, and read latest value, WoW/MoM deltas, trend, and RAG in seconds.")


def _sample() -> pd.DataFrame:
    """13 weeks of three business metrics in long format."""
    rng = np.random.default_rng(7)
    weeks = pd.date_range("2026-03-30", periods=13, freq="W-MON")
    rows = []
    revenue = 120_000.0
    signups = 800.0
    churn = 4.5
    for d in weeks:
        revenue *= 1 + rng.normal(0.02, 0.03)
        signups *= 1 + rng.normal(0.015, 0.05)
        churn += rng.normal(-0.05, 0.2)
        rows += [
            {"date": d, "metric": "Revenue ($)", "value": round(revenue, 0)},
            {"date": d, "metric": "New signups", "value": round(signups, 0)},
            {"date": d, "metric": "Churn (%)", "value": round(max(churn, 0.5), 2)},
        ]
    return pd.DataFrame(rows)


DEFAULT_TARGETS = {
    "Revenue ($)": {"target": 160_000.0, "direction": "up"},
    "New signups": {"target": 1000.0, "direction": "up"},
    "Churn (%)": {"target": 3.0, "direction": "down"},
}

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV (date, metric, value)", type=["csv"])
    use_sample = st.checkbox("Use sample data", value=uploaded is None)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = _sample()
else:
    st.info("Upload a long-format CSV (columns: date, metric, value) or tick **Use sample data**.")
    st.stop()

missing = {"date", "metric", "value"} - set(df.columns)
if missing:
    st.error(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

metrics = sorted(df["metric"].unique())

# Build editable targets - prefill known defaults, else assume "higher is better".
with st.sidebar:
    st.header("Targets")
    targets = {}
    for m in metrics:
        d = DEFAULT_TARGETS.get(m, {})
        default_val = float(d.get("target", float(df[df["metric"] == m]["value"].max())))
        col1, col2 = st.columns([2, 1])
        tgt = col1.number_input(f"{m} target", value=default_val, key=f"t_{m}")
        direction = col2.selectbox("better", ["up", "down"],
                                   index=0 if d.get("direction", "up") == "up" else 1,
                                   key=f"d_{m}", label_visibility="hidden")
        targets[m] = {"target": tgt, "direction": direction}

rows = build_scorecard(df, targets)
summary = scorecard_summary(rows)

# Health banner first - the headline an exec reads.
b = st.columns(4)
b[0].metric("🟢 On target", summary["green"])
b[1].metric("🟡 At risk", summary["amber"])
b[2].metric("🔴 Off target", summary["red"])
b[3].metric("Metrics tracked", len(rows))

st.subheader("Scorecard")
score_df = pd.DataFrame([
    {
        "": RAG_EMOJI.get(r["rag"], "⚪"),
        "Metric": r["metric"],
        "Latest": r["latest"],
        "Target": r["target"],
        "% of target": r.get("target_pct"),
        "WoW %": r["wow_pct"],
        "MoM %": r["mom_pct"],
        "Trend": {"up": "↑", "down": "↓", "flat": "→"}[r["trend"]],
        "As of": r["asof"],
    }
    for r in rows
])
st.dataframe(score_df, use_container_width=True, hide_index=True)

st.subheader("Trends")
pick = st.selectbox("Metric", metrics)
series = (df[df["metric"] == pick]
          .assign(date=lambda x: pd.to_datetime(x["date"]))
          .sort_values("date")
          .set_index("date")["value"])
st.line_chart(series)
tgt = targets[pick]["target"]
better = "higher" if targets[pick]["direction"] == "up" else "lower"
st.caption(f"Target for **{pick}**: {tgt:,.2f} ({better} is better)")

st.caption("Scoring core lives in `kpi.py` - reusable as a KPI Tracker app on the platform shell.")
