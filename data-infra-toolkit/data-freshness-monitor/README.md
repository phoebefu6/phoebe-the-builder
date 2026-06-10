# Data Freshness Monitor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/data-freshness-monitor/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/data-freshness-monitor/demo.ipynb)

> Dashboards show stale data and nobody notices — this tool catches it first.

## Business Impact
- **Before:** Stakeholders make decisions on stale data; nobody knows a pipeline broke until someone complains
- **After:** Every data source has an SLA with automated freshness checks and visual status
- **Estimated ROI:** 3-5 hours/week saved on "is this data up to date?" investigations

## Tech Stack
Python, Streamlit, pandas, matplotlib

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Learning Connection
Built while studying the Data Engineer Career Track (DS365) and Monitoring & Observability.
Applies: data pipeline monitoring, SLA design, observability dashboards.

## Impact Note
- **Who benefits:** Data engineers, analytics teams, anyone consuming warehouse data
- **Potential risks:** False confidence if SLA thresholds are set too loosely; monitor the monitor
