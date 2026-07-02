# Data Dictionary Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/data-dict-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/data-dict-gen/demo.ipynb)

> Point it at a table, get a documented data dictionary — types, PII flags, descriptions.

## Business Impact
- **Before:** Undocumented columns — "what does `mrr_usd` mean? is `full_name` PII?" — cost every analyst hours and every audit a scramble.
- **After:** Upload a CSV; get a full data dictionary in one pass — semantic types, null rates, PII flags, plain-English descriptions, exportable to Markdown/CSV.
- **Estimated ROI:** hours per new dataset onboarded; a head start on data governance + PII inventory.

## Tech Stack
Python · pandas profiling · semantic typing (name hints + value inspection) · PII detection · Claude API (optional richer descriptions) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV (or use the built-in sample), generate the dictionary, and export to Markdown or CSV. Set an `ANTHROPIC_API_KEY` for Claude-written business descriptions.

## How it works
1. **Profile** each column: dtype, null %, cardinality, sample values.
2. **Semantic type** it (id / email / date / currency / boolean / category / numeric / text) from name hints + value checks.
3. **Flag PII** (email, phone, name/address/SSN-style text).
4. **Describe** each column in plain English from the profile; **Claude mode** upgrades to business context ("Monthly recurring revenue per customer in USD").

## Learning Connection
Built while studying **schema analysis & auto-documentation** (Anthropic Prompt Engineering).
Applies: profile-before-prompt, grounding descriptions in observed data, and keeping a deterministic fallback that needs no API key.

## Impact Note
- **Who benefits:** data engineers, analysts, governance/privacy teams standing up a catalog.
- **Potential risks:** semantic typing and PII detection are heuristic — they can miss PII in oddly-named columns or misclassify sparse data. Treat the PII flags as a first-pass inventory, not a compliance guarantee, and have a human confirm before publishing.
