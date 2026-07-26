# SQL Optimizer Advisor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/query-optimizer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/query-optimizer/demo.ipynb)

> Slow queries, no idea why? This lints SQL for the anti-patterns that quietly force full table scans.

## Business Impact
- **Before:** A query is slow and nobody knows why; people guess, add indexes at random, or just wait.
- **After:** A static linter flags the usual culprits — non-sargable filters, leading-wildcard `LIKE`, comma-joins with no key, `SELECT *` — explains why each is slow, and shows the rewrite. A health score makes "is this query OK?" answerable.
- **Estimated ROI:** faster queries, fewer warehouse-cost surprises, and a teachable checklist for analysts.

## Tech Stack
Python · static SQL heuristics (regex, string-literal aware) · severity-weighted health score · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste a query (or load an example), get a health score and per-issue findings with why-it-hurts and how-to-fix.

## How it works
1. **Strip string literals** so values don't get mistaken for SQL.
2. **Scan** for anti-patterns: `SELECT *`, no `WHERE`, non-sargable functions on columns, leading-wildcard `LIKE`, `OR` chains, comma-joins without a join key, `DISTINCT`, `ORDER BY` without `LIMIT`, correlated subqueries in the `SELECT` list.
3. **Distinguish** a real join key (`a.col = b.col`) from a filter (`YEAR(col) = 2026`) — so cross-joins are caught without false alarms on every filtered query.
4. **Score** 0-100, penalizing by severity.

## Learning Connection
Built while studying **SQL performance & sargability**.
Applies: pattern-based static analysis, string-literal-safe parsing, and encoding DBA rules of thumb as an explainable linter.

## Impact Note
- **Who benefits:** analysts, analytics engineers, anyone writing warehouse SQL.
- **Potential risks:** it's a **heuristic linter, not `EXPLAIN`** — it can't see table sizes, indexes, or the real plan, so it may flag a pattern that's fine on a small table or miss a problem specific to your engine. Use it as a first-pass checklist and confirm with the actual query plan before big changes.
