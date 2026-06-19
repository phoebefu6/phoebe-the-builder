# Great Expectations Config Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/gx-config-generator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/gx-config-generator/demo.ipynb)

> Setting up data validation is tedious - so most teams skip it until something breaks in prod.

Point this at a CSV (or any DataFrame). It auto-profiles every column and emits a ready-to-run
**Great Expectations** expectation suite: not-null rules, key/uniqueness, numeric ranges, type
checks, allowed-value sets, plus table-level row-count and column-set checks.

## Business Impact
- **Before:** an analyst hand-writes 15-20 validation rules per new dataset - slow, skipped under deadline.
- **After:** upload the file, download a suite in seconds, then review and tighten.
- **Estimated ROI:** ~1-2 hours saved per new dataset onboarded.

## Tech Stack
Python · pandas · Streamlit · Great Expectations suite format · Docker

## What it infers

| Signal in the data | Expectation emitted |
|--------------------|---------------------|
| Column present | `expect_column_to_exist` |
| No nulls observed | `expect_column_values_to_not_be_null` |
| Some nulls | same, with `mostly` = observed non-null rate |
| Unique ratio ≥ 0.95 | `expect_column_values_to_be_unique` (key candidate) |
| Numeric column | `expect_column_values_to_be_between` (padded ±10%) + `_to_be_of_type` |
| Low-cardinality text (≤ 20 distinct) | `expect_column_values_to_be_in_set` |
| Whole table | row-count band + `expect_table_columns_to_match_set` |

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV (or tick "Use sample data"), inspect the per-column coverage, and download the suite JSON.

## Use the suite with Great Expectations
```python
import great_expectations as gx
ctx = gx.get_context()
# save the downloaded JSON to great_expectations/expectations/<name>.json
ctx.run_checkpoint(checkpoint_name="my_checkpoint")
```

## Learning Connection
Built while studying the data-quality / data-centric AI track (Great Expectations).
Applies: auto-profiling, expectation-suite design, validation-as-code.

## Impact Note
- **Who benefits:** data engineers and analysts onboarding new datasets.
- **Potential risks:** auto-generated bounds reflect only the sample seen - treat them as a
  starting point, not a final contract. Review before wiring into CI so you don't bake in
  too-loose or too-tight rules.
