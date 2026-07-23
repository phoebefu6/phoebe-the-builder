# Lightweight Data Catalog

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/data-catalog/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/data-catalog/demo.ipynb)

> Nobody knows what tables and columns exist, what they mean, or who owns them - so every analysis starts by pinging three people on Slack.

## Business Impact
- **Before:** A new analyst opens a database, stares at 40 cryptic column names, and burns a morning on Slack asking "what is `cust_ltv`? who owns `orders`? is `status` still used?"
- **After:** Point the catalog at your DataFrames/CSVs; every table is auto-profiled (shape, dtype, null %, distinct count, sample values, inferred semantic type), enriched with owner/description/tags, searchable in one box, and exportable as a Markdown data dictionary.
- **Estimated ROI:** ~2-3 hours saved per new-dataset onboarding, and the tribal knowledge outlives whoever happens to remember it today.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (builds the sample catalog, prints the summary and a search):
```bash
python catalog.py
```

## Learning Connection
Built while studying data governance and metadata management on the Data Quality & Governance arc.
Applies: automated data profiling, semantic-type inference, the data-dictionary / catalog pattern (owner + description + tags per table), and search-driven discovery - the on-ramp to formal cataloguing tools like DataHub or OpenMetadata, at notebook scale.

## Impact Note
- **Who benefits:** new analysts and data scientists onboarding to an unfamiliar warehouse, data stewards documenting what they own, and any team tired of re-answering "where does this field live?"
- **Potential risks:** the semantic types (id / email / date / category / numeric) are INFERRED from column names and sampled values - they are heuristic guesses, not truth, and a human should confirm them before anyone relies on them. Sample values may surface real data, so treat exports the way you treat the underlying tables. This catalog is for discovery and documentation only - it is not access control and grants no permissions.
