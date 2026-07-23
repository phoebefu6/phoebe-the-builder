# Dataset Snapshot Diff

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/data-diff/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/data-diff/demo.ipynb)

> Between yesterday's snapshot and today's, what actually changed? Nobody can answer without eyeballing two files - this diffs the same dataset over time on a key column and tells you exactly what moved.

## Business Impact
- **Before:** Someone opens two exports side by side and squints, trying to spot which rows are new, which vanished, and which values quietly changed - slow, error-prone, and undocumented.
- **After:** One diff names the added rows, removed rows, and modified rows (down to the column and old -> new value), flags schema drift, and reports how much of the dataset churned - in seconds, with a downloadable audit trail.
- **Estimated ROI:** ~2-3 hours/week saved per analyst on manual snapshot comparison, plus caught silent changes (a price flip, a dropped column) before they surprise a downstream report.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the change summary + a sample of modified rows for the built-in sample):
```bash
python differ.py
```

## Learning Connection
Built while studying change data capture and snapshot auditing on the Data Quality & Governance arc.
Applies: temporal diffing on a stable key, schema-drift detection, and change-velocity metrics - the "what changed since last run?" question that CDC pipelines, slowly-changing-dimension loads, and data-contract monitoring all depend on. This is change-over-time within one system, distinct from cross-system reconciliation.

## Impact Note
- **Who benefits:** data stewards, analytics engineers, and anyone who owns a table that gets refreshed on a schedule and needs to explain what moved between runs.
- **Potential risks:** a diff shows WHAT changed, not WHY. A large diff might be a legitimate bulk update or a broken pipeline - the tool cannot tell them apart, so big or surprising diffs always need a human to judge intent. Treat the output as an audit trail and review queue, never as an approval to promote data downstream.

_Month 10 capstone (Day 100) of the FDE portfolio - the Data Quality & Governance Suite._
