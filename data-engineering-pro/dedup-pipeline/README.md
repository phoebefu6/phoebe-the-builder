# Dedup & Survivorship Pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/dedup-pipeline/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/dedup-pipeline/demo.ipynb)

> Duplicate records everywhere — normalize the variants, cluster by match keys, and merge each cluster into a golden record with field-level survivorship and full provenance.

## Business Impact
- **Before:** The same customer exists 3 times across CRM, web, and imports; row-level "keep newest" dedup throws away the CRM's better email and the web's fresher phone.
- **After:** One golden record per entity where *each field* took its best value by an explicit rule — and a provenance table answers "where did this value come from?" for every merge.
- **Estimated ROI:** Deduped counts stakeholders trust, no more double-mailed customers, and auditable merges instead of black-box ones.

## Tech Stack
Python 3.10+, pandas, Streamlit, matplotlib. Normalizers (Gmail dot/plus stripping, phone digit extraction), pluggable survivorship rules. Runs fully offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app (rules configurable in the sidebar):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Normalize** — different spellings of the same entity collide: emails lowercased with Gmail dots/plus-tags stripped, phones reduced to last 10 digits. Normalized keys match; golden records keep real values.
2. **Cluster** — rows with matching normalized `(email, phone)` form one entity.
3. **Survivorship per field** — `most_recent`, `source_priority(crm>web>import)`, `longest_value`, `most_complete_record`; default rule for unlisted fields.
4. **Provenance** — every contested field logs winner record, source, and rule; conservation asserted (`clusters + merged = input rows`).

Sample: 99 messy rows from 3 systems → 78 golden records, 21 duplicates absorbed, every contested field auditable.

## Learning Connection
Built while studying data contracts and MDM patterns (Month 7: Data Engineering Pro).
Applies: entity resolution, survivorship-rule design, and audit-first merging.

## Impact Note
- **Who benefits:** Anyone consolidating customer/product/vendor records across systems — the daily grind of CDPs and MDM projects.
- **Potential risks:** Exact-key clustering misses fuzzy duplicates (typo'd emails, no shared keys) — production entity resolution adds blocking + similarity scoring on top of exactly this survivorship layer.
