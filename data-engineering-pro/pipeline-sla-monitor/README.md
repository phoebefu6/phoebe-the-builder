# Pipeline SLA Monitor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/pipeline-sla-monitor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/pipeline-sla-monitor/demo.ipynb)

> A pipeline that lands late, runs slow, goes stale, or arrives thin usually breaks a consumer before anyone on the data team notices. This turns four plain-English promises per pipeline into a compliance scorecard you can page on.

## Business Impact
- **Before:** Pipeline health lives in scattered orchestrator logs. You find out `orders_daily` was late - or that `ml_features` stopped running entirely - from a dashboard on fire or an angry Slack ping.
- **After:** Declare an SLA per pipeline, point it at your run history, and get a fleet scorecard plus a breach log naming the exact night and reason. Fresh-run detection catches the scariest failure of all: the job that isn't running.
- **Estimated ROI:** Catches silent staleness and repeated breaches hours-to-days earlier; ~2-3 hrs/incident saved plus the downstream incidents avoided.

## Tech Stack
Python (standard-library core - `dataclasses`, `datetime`), Streamlit, pandas, matplotlib. No warehouse connection, no API keys - runs anywhere.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with the scorecard, breach log, and dashboard chart, or click the Colab/Binder badges above to run it live.

For the Streamlit app (fleet scorecard + filterable breach log):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
An SLA is a promise with four dimensions. A run **breaches** if it misses any of them:

| Dimension | Breach type | Promise |
|-----------|-------------|---------|
| Landing time | `LATE` | run finishes by a wall-clock deadline |
| Duration | `SLOW` | run completes within a time budget |
| Freshness | `STALE` | a *successful* run exists within N hours of now |
| Volume | `LOW_VOLUME` | run lands at least a floor number of rows |
| Any error | `FAILED` | the run itself errored |

1. **Declare** an `SLA(landing_by, max_duration_min, freshness_hours, min_rows)` per pipeline.
2. **Grade** every `Run` in the history against its SLA.
3. **Roll up** to a per-pipeline compliance % (clean run-dates / total runs) and a status:
   🟢 HEALTHY (100%) · 🟡 AT_RISK (90-99%) · 🔴 BREACH (<90% or stale).
4. **Freshness** checks the newest *successful* run against `now` - so a pipeline that quietly stops running is flagged even if its past runs all looked fine.

The key insight the demo makes: **compliance % alone hides the worst failure.** In the sample, `ml_features` scores 90.9% yet is the most dangerous pipeline on the board because it silently stopped - only the freshness check surfaces it.

## Learning Connection
Built while studying data engineering observability patterns (SLAs/SLOs, freshness monitoring, Airflow/dbt run metadata, OpenLineage).
Applies: SLA/SLO modeling, run-history evaluation, freshness/staleness detection, fleet rollups, alert-ready reporting.

## Impact Note
- **Who benefits:** Data / analytics / platform engineers who own scheduled pipelines and answer for their reliability.
- **Potential risks:** It grades the run history you feed it - garbage timestamps in, garbage grades out. Thresholds are promises *you* set; too loose and it never fires, too tight and it cries wolf. It reports on runs, it does not fix them. Validate SLAs with downstream consumers before wiring it to a real pager.
