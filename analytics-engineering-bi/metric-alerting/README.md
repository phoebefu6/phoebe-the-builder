# Metric Anomaly Alerter

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/metric-alerting/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/metric-alerting/demo.ipynb)

> We notice metric drops late — this watches a metric and fires the day it moves, with three independent detectors.

## Business Impact
- **Before:** A revenue/DAU drop on Monday gets noticed Thursday. Three days of damage before anyone reacts.
- **After:** Every metric point is checked against a hard SLA, its statistical baseline, and its day-over-day trend — anomalies surface the day they happen, with severity.
- **Estimated ROI:** cuts detection lag from days to hours; earlier response means smaller incidents.

## Tech Stack
Python · numpy/pandas · three detectors (static threshold, rolling z-score, day-over-day trend) · severity + dedupe · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Tune the floor/ceiling, z-threshold, trend %, and window; see the metric charted with anomalies flagged and a full alerts table.

## How it works
1. **threshold** — value outside a business floor/ceiling.
2. **z-score** — value far from its rolling baseline (statistical outlier). The baseline uses the *prior* window only, so a point is never measured against a baseline it has polluted.
3. **trend** — sharp day-over-day % move, even if still in range.
4. **severity + dedupe** — each alert is graded info/warning/critical; duplicates per (date, detector) collapse to the worst.

## Learning Connection
Built while studying **observability & anomaly detection for metrics**.
Applies: ensemble-of-simple-detectors over one opaque model, and the subtle-but-critical "exclude the current point from its own baseline" fix (a −2.4σ miss becomes a −21σ catch).

## Impact Note
- **Who benefits:** analytics, data, and on-call teams watching business or pipeline metrics.
- **Potential risks:** naive detectors have no seasonality — Monday dips or month-end spikes can false-alarm; add day-of-week baselines for periodic metrics. Tune thresholds to your metric's noise, or alert fatigue will train people to ignore it.
