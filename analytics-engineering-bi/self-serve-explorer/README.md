# Self-Serve Data Explorer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/self-serve-explorer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/self-serve-explorer/demo.ipynb)

> Analysts get pinged for every number — a self-serve explorer lets anyone pivot, aggregate, and filter without SQL or a ticket.

## Business Impact
- **Before:** Every "what's revenue by region for paid channel?" is a Slack ping and a context switch for an analyst. Routine pulls crowd out real analysis.
- **After:** Business users fill in a form — group by, measure, aggregation, filters — and get the table (and a chart) themselves. Analysts keep the hard problems.
- **Estimated ROI:** deflects a large share of ad-hoc data requests; faster answers for everyone.

## Tech Stack
Python · pandas pivot/groupby engine · auto dimension/measure profiling · query-spec dataclass · Streamlit UI · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV (or use the sample), pick rows/measure/aggregation, add a pivot column and filters, set Top-N — get a table, a chart, and a CSV export.

## How it works
1. **Profile** columns into dimensions (group-by candidates) and measures (aggregatable).
2. **Spec** — an `ExploreQuery` captures rows, measure, agg, optional pivot columns, filters, top-N.
3. **Execute** — the engine maps the spec to pandas `groupby` / `pivot_table` with the chosen aggregation (incl. `count_distinct`).
4. **Suggest** — a couple of auto-generated starter queries so a blank explorer isn't intimidating.

## Learning Connection
Built while studying **self-serve BI & the analyst-bottleneck problem**.
Applies: turning a repetitive analyst task into a parameterized tool, and separating the *query spec* (what the user wants) from *execution* (how it's computed).

## Impact Note
- **Who benefits:** PMs, ops, marketing, execs (self-serve answers); analysts (fewer interruptions).
- **Potential risks:** self-serve numbers can drift from official ones — back the engine with a governed **metrics layer** (Day 101) so an ad-hoc "revenue" matches finance's. Without shared definitions, self-serve can multiply conflicting numbers instead of reducing them.
