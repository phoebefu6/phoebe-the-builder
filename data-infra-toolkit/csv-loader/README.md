# CSV to PostgreSQL Loader

> Loading data into our database is a manual nightmare — copy-paste into SQL, fight with column types, pray nothing breaks.

## Business Impact
- **Before:** Manually writing INSERT statements or using pgAdmin import — 30-60 min per file, error-prone
- **After:** Drag-drop CSV, auto-detect types, one-click load — under 2 minutes
- **Estimated ROI:** ~4 hours/week saved for any team doing regular data imports

## Tech Stack
Python, Streamlit, SQLAlchemy, pandas, psycopg2

## Features
- Auto-detects column types (INTEGER, VARCHAR, TIMESTAMP, etc.)
- Smart type inference based on actual values, not just dtype
- Override any column type before loading
- Preview data and see null/duplicate counts before committing
- Shows the generated CREATE TABLE SQL
- Handles `fail`, `replace`, or `append` modes
- Chunked inserts for large files
- Sanitizes filenames into valid table names

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Impact Note
- **Who benefits:** Data analysts, data engineers, ops teams who regularly load CSVs into Postgres
- **Potential risks:** Credentials entered in the UI — use only on trusted machines, not public deployments
