# Metrics Layer / Semantic Definitions

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/metrics-layer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/metrics-layer/demo.ipynb)

> Every dashboard computes revenue differently — a metrics layer makes one governed definition the single source of truth.

## Business Impact
- **Before:** Finance, growth, and the exec deck each report a different "revenue" because each tool re-implements the calculation. Meetings derail into "whose number is right?"
- **After:** Each metric is defined once in a versioned YAML store — expression, allowed dimensions, filters, owner. Every tool renders the same canonical SQL and gets the same number.
- **Estimated ROI:** kills recurring metric-reconciliation fire drills; new dashboards inherit correct definitions for free.

## Tech Stack
Python · dependency-free YAML metric store · validator (conflict/duplicate/owner checks) · canonical SQL renderer · pandas compute engine · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Edit the YAML store, see live validation, pick a metric, choose a grain, and get canonical SQL + the computed result on sample data.

## How it works
1. **Define** metrics in YAML — `expr` (e.g. `sum(amount)`), dimensions, filters, owner.
2. **Validate** — flag unsupported expressions, missing owners, and (critically) duplicate names with *conflicting* definitions.
3. **Render** each metric to canonical SQL at any requested grain.
4. **Compute** it on a DataFrame — the definition isn't just docs, it runs — with the same filter logic every time.

## Learning Connection
Built while studying **analytics engineering & the semantic layer** (dbt MetricFlow / LookML concepts).
Applies: define-once governance, validation-as-CI-gate, and separating a metric's *definition* from its *computation*.

## Impact Note
- **Who benefits:** analytics engineers, BI teams, finance/ops — anyone tired of reconciling the same metric across tools.
- **Potential risks:** the SQL renderer is intentionally minimal (one dialect, simple filters) — treat it as a reference, not a full MetricFlow. A metrics layer is only trustworthy if the store is the *enforced* source; if teams can still hand-write metric SQL, drift returns.
