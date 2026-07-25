# Natural Language to SQL

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/nl-to-sql/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/nl-to-sql/demo.ipynb)

> Non-analysts can't query the warehouse — NL→SQL lets them ask in English, behind guardrails that keep it read-only and in-schema.

## Business Impact
- **Before:** Every "what was revenue in EU last month?" becomes a ticket for the analytics team. Non-analysts wait; analysts get interrupted.
- **After:** Ask in plain English, get an answer — with a guardrail layer that guarantees the query is read-only and scoped to allowed tables/columns.
- **Estimated ROI:** deflects routine data-pull requests; unblocks self-serve without handing out raw SQL access.

## Tech Stack
Python · rule-based NL→SQL translator · **guardrail layer** (SELECT-only, schema whitelist, no multi-statement, auto-LIMIT) · in-memory SQLite executor · Claude API (optional) · Streamlit · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Ask a question, see the generated SQL, watch guardrails block anything unsafe, and get the result on sample data. Set an `ANTHROPIC_API_KEY` to use Claude for translation (output still passes the guardrail).

## How it works
1. **Schema** — a whitelist of the only legal table + columns.
2. **Translate** — English → candidate SQL, via a rule-based translator (count/sum/avg/top-N/group-by/filter) or Claude.
3. **Guard** — `guard_sql` rejects non-SELECT, write/DDL keywords, multiple statements, and unknown tables/columns; injects a `LIMIT`. **The LLM never bypasses this gate.**
4. **Execute** — the vetted query runs against the data via SQLite.

## Learning Connection
Built while studying **LLM-to-SQL patterns & safe tool use** (Anthropic).
Applies: never trust generated code — validate against an allowlist; keep the safety layer independent of (and downstream from) the generator.

## Impact Note
- **Who benefits:** analysts (fewer interruptions), PMs/ops/execs (self-serve answers).
- **Potential risks:** the rule translator handles common shapes only; complex joins/window functions need the LLM path — and even then, a syntactically-safe query can still be *semantically* wrong. Show the SQL, keep results read-only, and pair with a metrics layer so definitions stay correct.
