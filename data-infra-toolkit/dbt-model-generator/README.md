# dbt Model Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/dbt-model-generator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/dbt-model-generator/demo.ipynb)

> Writing dbt staging models from scratch is repetitive boilerplate — paste your SQL DDL and get production-ready dbt artifacts instantly.

## Business Impact
- **Before:** Manually writing dbt source YAML, staging models, and schema tests for every new table — 30-60 min per schema
- **After:** Paste CREATE TABLE SQL, get all dbt files in seconds
- **Estimated ROI:** 2-4 hours/week saved for data teams onboarding new sources

## Tech Stack
- Python 3.11
- Streamlit (interactive UI)
- Regex-based SQL DDL parser
- dbt best practices (CTE pattern, snake_case renaming, auto-tests)
- Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What It Generates

From SQL `CREATE TABLE` statements:
1. **Source YAML** (`__sources.yml`) — declares raw data sources with `unique` and `not_null` tests on PKs
2. **Staging Models** (`stg_*.sql`) — CTE-based models following dbt best practices, with camelCase → snake_case renaming
3. **Model Schema YAML** (`_models.yml`) — column descriptions and schema tests

## Learning Connection
Built while studying the Data Engineer Career Track on DS365/IBM.
Applies: dbt fundamentals, SQL transformation patterns, data modeling best practices.

## Impact Note
- **Who benefits:** Data engineers, analytics engineers onboarding new data sources
- **Potential risks:** Generated models are a starting point — always review column types and add business-specific tests before production use
