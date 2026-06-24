# 📊 KPI Tracker

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/kpi-tracker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/kpi-tracker/demo.ipynb)

> Execs ask for the same metrics every Monday - this turns a metric time series into the answer automatically.

## Business Impact
- **Before:** an analyst rebuilds the same metrics pull by hand every Monday - load data, compute deltas vs last week, compare to target, format the email. ~1-2 hours, every week.
- **After:** upload a long-format export once, set targets once, and read the latest value, WoW/MoM deltas, trend, and RAG status on a self-serve dashboard. Execs pull instead of ask.
- **Estimated ROI:** ~1.5 hours/week saved per recurring metrics report, plus fewer "where's the number?" interrupts.

## What it does
- Takes a tidy `[date, metric, value]` frame + a target config per metric
- Computes, per metric: latest value, **week-over-week** and **month-over-month** % change, trend direction, **% of target**, and a **RAG** (red/amber/green) band
- Handles **"lower is better"** metrics (churn, backlog, cost) by flipping the RAG comparison
- Rolls up a one-line health read: *N on target | N at risk | N off target*

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib, Docker, GitHub Actions (ruff lint + smoke test).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with the RAG scorecard + trend charts, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with columns `date, metric, value` (long format), or tick **Use sample data**.

## How scoring works
- `kpi.py` is **UI-free core logic** - shared by the Streamlit app and mountable as a "KPI Tracker" app on the platform shell, or callable from a scheduled report job.
- RAG bands: **green** = at/above target (or at/below, for "down" metrics); **amber** = within 10% of target; **red** = beyond that.

## Learning Connection
Built while studying **Streamlit** + **AWS Cloud Technical Essentials** (Analytics Accelerator month of the FDE roadmap).
Applies: self-serve dashboards, period-over-period metric logic, RAG status design, separating reusable core logic from UI.

## Impact Note
- **Who benefits:** analysts who own recurring exec/metrics reports; leaders who want a self-serve scorecard.
- **Potential risks:** a green RAG is only as good as the target and the data feeding it - a stale or wrong export reads "on target" while reality drifts. Pair with a freshness check on the source.
