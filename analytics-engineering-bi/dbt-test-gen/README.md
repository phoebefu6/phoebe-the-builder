# dbt Test Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/dbt-test-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/dbt-test-gen/demo.ipynb)

> No tests on our models — this profiles a table and auto-generates a paste-ready dbt `schema.yml`.

## Business Impact
- **Before:** Everyone agrees models need tests; nobody writes them because hand-writing `not_null`/`unique`/`accepted_values` across dozens of columns is tedious. Coverage sits at zero.
- **After:** Profile the model, get explainable test suggestions and a valid `schema.yml` in seconds — coverage goes from nothing to nearly complete in one pass.
- **Estimated ROI:** hours saved per model, plus data-quality bugs caught by tests that would otherwise never have been written.

## Tech Stack
Python · pandas profiling · rule-based test inference (`not_null`, `unique`, `accepted_values`, `relationships`) · dbt `schema.yml` renderer · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV (or use the sample), tune the thresholds, and get suggested tests per column (with reasons) plus a downloadable `schema.yml`.

## How it works
1. **Profile** each column — null fraction, uniqueness ratio, cardinality, name pattern.
2. **Infer** tests from the evidence: 0% null → `not_null`; ~all-distinct → `unique`; low-cardinality text → `accepted_values`; `*_id` name → `relationships`.
3. **Restrain** — a near-unique column (email) gets `unique`, *not* `accepted_values`, so tests don't break on every new value.
4. **Render** a valid dbt `schema.yml` and report coverage.

## Learning Connection
Built while studying **analytics engineering & dbt testing**.
Applies: data-profiling → policy, explainable automation (every suggestion carries its reason), and knowing when *not* to fire a test.

## Impact Note
- **Who benefits:** analytics engineers and anyone maintaining a dbt project with thin test coverage.
- **Potential risks:** suggestions come from a *sample* — a column that looks unique/required in the sample may not be in full data, so review before committing. `accepted_values` lists reflect only observed values; a legitimate new category will (correctly) fail the test until the list is updated.
