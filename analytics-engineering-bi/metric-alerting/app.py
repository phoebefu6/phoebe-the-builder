from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from alerting import MetricConfig, detect_anomalies, sample_series, summarize

SEV_COLOR = {"critical": "#c0553b", "warning": "#e08a2b", "info": "#3b6fd6"}

st.set_page_config(page_title="Metric Anomaly Alerter", layout="wide")
st.title("Metric Anomaly Alerter")
st.caption("Catch metric drops and spikes the day they happen — not when someone finally notices.")

series = sample_series()

with st.sidebar:
    st.subheader("Alert rules")
    name = st.text_input("Metric name", value=series.name)
    lo = st.number_input("Floor (min)", value=800.0)
    hi = st.number_input("Ceiling (max)", value=1300.0)
    z = st.slider("Z-score threshold", 1.5, 5.0, 2.5, 0.1)
    trend = st.slider("Trend alert (day-over-day %)", 0.05, 1.0, 0.20, 0.05)
    window = st.slider("Baseline window (points)", 3, 14, 7)
    direction = st.selectbox("Direction", ["both", "drop", "spike"])

cfg = MetricConfig(
    name=name, min_value=lo, max_value=hi, z_threshold=z,
    trend_pct=trend, window=window, direction=direction,
)
alerts = detect_anomalies(series, cfg)
summ = summarize(alerts)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alerts", summ["total"])
c2.metric("Critical", summ["by_severity"]["critical"])
c3.metric("Warning", summ["by_severity"]["warning"])
c4.metric("Detectors firing", len(summ["by_kind"]))

# chart
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(range(len(series)), series.values, color="#3b6fd6", lw=1.6, marker="o", ms=3, label=name)
if cfg.min_value is not None:
    ax.axhline(cfg.min_value, color="#c0553b", ls="--", lw=1, alpha=0.5)
if cfg.max_value is not None:
    ax.axhline(cfg.max_value, color="#e08a2b", ls="--", lw=1, alpha=0.5)
dates = list(series.index)
for a in alerts:
    if a.date in dates:
        i = dates.index(a.date)
        ax.scatter([i], [a.value], color=SEV_COLOR[a.severity], s=90, zorder=5, edgecolors="white")
ax.set_title(f"{name} with anomalies flagged", fontsize=12, weight="bold")
ax.set_xticks(range(0, len(series), 3))
ax.set_xticklabels([dates[i][5:] for i in range(0, len(series), 3)], rotation=45, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left")
st.pyplot(fig)

st.subheader("Alerts")
if alerts:
    st.dataframe(
        pd.DataFrame([{"date": a.date, "severity": a.severity, "kind": a.kind, "message": a.message} for a in alerts]),
        use_container_width=True,
    )
else:
    st.success("No anomalies under the current rules.")
