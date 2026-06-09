# API to Warehouse Connector

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/api-warehouse-connector/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/api-warehouse-connector/demo.ipynb)

> Every Monday, someone on the team downloads CSVs from five different SaaS tools. This replaces that.

## The problem (and why it bugged me)

Exporting data from Stripe, HubSpot, GitHub, Slack, and whatever else you use — it's tedious, error-prone, and nobody notices when the dashboards go stale until someone important asks a question.

One `sync_all()` call pulls from all five sources, normalizes everything, and drops it into a queryable SQLite warehouse. Every sync gets logged, so you can always trace what landed and when.

That's roughly 2-3 hours a week back. More importantly, no more "wait, is this data from last Tuesday?" conversations.

## How it works

Pretty standard ETL, nothing fancy:

1. **Extract** — each source has a fetch function. Mock data for the demo, swap in real API calls when you're ready
2. **Transform** — records become DataFrames, stamped with source name and load time
3. **Load** — written to SQLite tables with sync logging
4. **Audit** — every sync gets a row in `_sync_log` with row counts and a data hash

## Tech stack
Python, pandas, SQLite, Streamlit, Matplotlib, Seaborn, Docker

## Try it

**[Run the demo notebook →](demo.ipynb)** — already executed with outputs visible, or click the Colab/Binder badges to run it yourself.

The notebook walks through the whole pipeline: defining connectors, running the sync, querying what landed, and charting data freshness.

For the Streamlit version:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's connected

| Source | Table | What you get |
|--------|-------|-------------|
| Stripe | `raw_stripe_transactions` | Payments — amounts, status, currency |
| HubSpot | `raw_hubspot_contacts` | CRM contacts with lifecycle stage |
| GitHub | `raw_github_events` | Pushes, PRs, issues, releases |
| Slack | `raw_slack_messages` | Channel messages, word counts, attachments |
| Weather | `raw_weather_readings` | Temp, humidity, conditions by city |

## Add your own

```python
def fetch_my_api():
    resp = requests.get("https://api.example.com/data", headers={"Authorization": "Bearer KEY"})
    return resp.json()["results"]

CONNECTORS["my_api"] = {
    "fetch": fetch_my_api,
    "table": "raw_my_api_data",
    "label": "My Custom API",
}

wh.sync("my_api")
```

## Ideas for later
- Scheduling with APScheduler or cron so it runs itself
- Swap SQLite for Postgres or BigQuery when the data outgrows a file
- Incremental loading — only pull records newer than the last sync
- Slack or email alerts when a sync fails

## Worth noting
Built for data teams, analysts, anyone who's sick of the Monday CSV ritual. The demo uses mock data — real connectors need proper auth, rate limit handling, and error recovery.
