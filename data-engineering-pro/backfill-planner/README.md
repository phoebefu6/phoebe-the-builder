# Backfill Planner

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/backfill-planner/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/backfill-planner/demo.ipynb)

> Backfills are error-prone manual scripts — chunk the date range, persist every status, retry the flaky, quarantine the broken, and resume after a crash without repeating a success.

## Business Impact
- **Before:** A 6-month backfill is a shell loop someone babysits; when it dies at 2am nobody knows which dates loaded, so it re-runs from scratch (or worse, double-loads).
- **After:** 27 idempotent chunks with per-chunk status on disk; flaky failures retry automatically, structural failures go dead after 3 strikes with the error attached, and any new run resumes exactly where the last one stopped.
- **Estimated ROI:** Backfills become kill-anytime/resume-anytime — no duplicated loads, no lost progress, no babysitting.

## Tech Stack
Python 3.10+, stdlib (dataclasses + JSON state), Streamlit, matplotlib, pandas. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **`plan_chunks`** — split `[start, end)` into non-overlapping daily/weekly/monthly chunks (weekly Monday-aligned); each chunk is the unit of work and of retry.
2. **`BackfillState`** — JSON-persisted statuses (pending/running/success/failed) with attempt counts, durations, and errors; every change hits disk immediately.
3. **`run_backfill`** — pulls runnable chunks (pending, then failures under the attempt cap), respects a parallelism cap, records outcomes, loops until nothing is runnable; 3 strikes = dead chunk waiting for a human.
4. **Resume** — a new process pointed at the same JSON continues the plan and never re-runs a success. Retries are safe because chunks are idempotent (delete-and-reload per range).

Demo: 6 months weekly under a 25% random failure rate + one structurally corrupt week — retries absorb the flakiness, the corrupt week goes dead with its error, and a post-fix run completes 27/27.

## Learning Connection
Built while studying incremental patterns and orchestration (Month 7: Data Engineering Pro).
Applies: idempotent chunk design, checkpointed state machines, retry-vs-quarantine semantics — the same model behind Airflow catchup and dbt retry.

## Impact Note
- **Who benefits:** Data engineers running historical reloads, migrations, or late-data repairs.
- **Potential risks:** If the per-chunk job is not truly idempotent, retries double-load — the planner makes retries safe only when the job honors chunk boundaries.
