# Source-to-Target Reconciliation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/reconciliation-checker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/reconciliation-checker/demo.ipynb)

> We copied a table from the source system into the warehouse and can't prove it matched - row counts drift and values silently differ.

## Business Impact
- **Before:** After a migration or nightly load, someone spot-checks a few rows, eyeballs a count, and hopes the copy is faithful. Silent drift - a dropped row, a re-rounded amount, a flipped status - surfaces weeks later in a wrong report.
- **After:** Point the checker at the source and the target on a chosen key. In seconds you get a pass/fail verdict with row-count delta, keys missing in the target, keys extra in the target, every cell-level value mismatch (which column, source value vs target value), and an aggregate sum cross-check - each finding explainable enough to defend in a ticket.
- **Estimated ROI:** ~2-3 hours saved per migration / load validation, plus corrupted-copy errors caught before they reach a dashboard instead of after.

## What it checks
- **Row-count delta** - did the same number of rows land?
- **Keys missing in target** - rows in the source that never made the trip.
- **Keys extra in target** - stray rows that should not exist.
- **Cell-level value mismatches** - on the shared keys, column by column, tolerant of harmless float re-rounds and whitespace so only real drift is flagged.
- **Aggregate checks** - a numeric column's sum must agree within tolerance, a cheap independent proof that nothing drifted at scale.
- **Match rate + verdict** - share of source rows that arrived AND matched on every cell, and an overall PASS only if every dimension agrees.

This is cross-system correctness at ONE point in time ("did the copy land?"), not change-over-time drift.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook -&gt;](demo.ipynb)** - pre-rendered with outputs, or click the Colab / Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (reconciles the built-in sample and prints the report):
```bash
python reconciler.py
```

## Learning Connection
Built while working through the data migration validation / ETL reconciliation arc of the Data Quality &amp; Governance suite. Applies: source-of-record thinking, set logic for key membership (missing vs extra), tolerance-aware cell comparison, and aggregate cross-checks as a scale-level control - the same pattern used to sign off a warehouse load or a system migration.

## Impact Note
- **Who benefits:** data engineers running migrations and nightly loads, analytics engineers who own warehouse tables, and stewards asked to certify that a copy is faithful.
- **Potential risks:** a match is evidence, not proof - reconciling on these checks does not certify every value is business-correct, only that source and target agree within the rules you set. Tolerances are a judgment call: too loose and real drift slips through, too tight and harmless float re-rounds cry wolf. Treat the verdict as a decision aid, and never delete "extra" target rows on the tool's word alone without confirming they are truly spurious.
