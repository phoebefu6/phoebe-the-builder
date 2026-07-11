# Incremental / CDC Loader

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/incremental-loader/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/incremental-loader/demo.ipynb)

> Full reloads take hours every night — load only what changed with watermark extraction, key upsert, and soft-delete tombstones.

## Business Impact
- **Before:** The nightly job rescans and rewrites the whole table; load time grows with table size and the warehouse bill grows with it.
- **After:** Cycle 1 is a full load, every cycle after scans only the delta — ~11% of the source in the demo, shrinking as the table grows.
- **Estimated ROI:** ~89% of nightly scan work avoided; loads finish in minutes instead of hours.

## Tech Stack
Python 3.10+, pandas, Streamlit, matplotlib. JSON watermark store, CDC-style soft deletes. Same logic ports directly to SQL `MERGE`. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Watermark store** — a JSON file remembers the max `updated_at` per table; each run extracts only rows after it.
2. **Incremental extract** — first run = full load; every later run = delta only.
3. **Upsert** — merge by key: update matches, insert new, apply tombstones (`deleted=True` removes the key).
4. **Verify** — the notebook asserts the incremental target is identical to a full reload: unique keys, matching row counts, current values.
5. **Simulator** — a fake customer table mutates each day (inserts/updates/soft deletes) so the loop is testable end-to-end without a database.

Five simulated days: cycle 1 scans 100%, cycles 2-5 scan 11-13% — 89% of scan work avoided overall.

## Learning Connection
Built while studying incremental patterns and CDC concepts (Month 7: Data Engineering Pro).
Applies: watermark-based extraction, upsert/merge semantics, tombstone handling, and equivalence testing against full reloads.

## Impact Note
- **Who benefits:** Data engineers running nightly batch loads that have outgrown their window.
- **Potential risks:** Late-arriving updates with backdated timestamps slip past the watermark — production systems pair this with a periodic reconciliation full-load.
