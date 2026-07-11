# Data Contract Validator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/data-contract-validator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/data-contract-validator/demo.ipynb)

> Producers break schemas without warning — put a YAML contract in CI so breaking changes block the merge and violating data never lands.

## Business Impact
- **Before:** A producer renames a column or loosens a type; consumers discover it when dashboards go blank; the postmortem takes longer than the fix.
- **After:** Two enforcement points: contract diff in the producer's PR (breaking change = failed build), dataset validation where data lands (violations = failed load).
- **Estimated ROI:** Schema-break incidents caught at merge time instead of production — hours of firefighting per incident avoided, plus restored trust between teams.

## Tech Stack
Python 3.10+, PyYAML, pandas, Streamlit, matplotlib. CI exit-code semantics. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app (two tabs: validate dataset, diff contract versions):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Contract YAML** — per column: `type`, `nullable`, `unique`, `allowed` enum, `min`/`max` bounds; dataset-level `freshness` (newest row younger than N hours).
2. **`validate_dataset`** — data vs contract: missing columns, type mismatches, nulls, dupes, enum drift, bound breaches, staleness, plus warnings for undeclared columns.
3. **`diff_contracts`** — version vs version: removed column / type change / loosened nullability = breaking (error); added optional column = safe (warning); added required column = breaking for old producers.
4. **`exit_code`** — 1 on any error: drop it straight into a CI step or an Airflow task.

Sample run: an orders CSV with five planted problems yields 4 errors + 1 warning (exit 1); a proposed contract v2 gets blocked with two breaking changes.

## Learning Connection
Built while studying data contracts (Month 7: Data Engineering Pro).
Applies: producer-owned data quality (data mesh principle), semantic versioning for schemas, and CI-gated data pipelines.

## Impact Note
- **Who benefits:** Data platform teams tired of consumer-side breakage; producer teams who want a clear definition of "don't break downstream."
- **Potential risks:** Contracts that nobody updates become fiction — pair the validator with ownership (each contract has an owner field) and make the diff check mandatory in CI.
