# DQ Rules Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/dq-rules-engine/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/dq-rules-engine/demo.ipynb)

> Our data-quality rules live in a wiki nobody enforces - by the time a bad batch is caught, it's already in a report.

## Business Impact
- **Before:** Quality rules are prose on a wiki page. Nobody runs them on every batch, so a broken primary key or an orphaned foreign key is discovered only when a downstream report looks wrong - after it has shipped.
- **After:** The same rules are plain dicts (or YAML) that run against the DataFrame on every batch. Each rule returns pass/fail, a violation count, sample offending values, and a plain-English message - with a rollup verdict that gates on real breaks (error) while still logging softer smells (warn).
- **Estimated ROI:** ~4-6 hours/week of firefighting avoided per data team, plus the bad batches caught at the gate instead of in a board deck.

## What it checks
Declare checks as plain dicts - no DSL to learn:

| Rule type | What it enforces |
|-----------|------------------|
| `not_null` | a column has no missing values |
| `unique` | no duplicate values (primary-key style) |
| `in_range` | numeric values sit within `[min, max]` (non-numeric values also fail) |
| `allowed_values` | every value is in an approved set |
| `regex_match` | values match a required pattern (e.g. email) |
| `foreign_key` | every value exists in a reference set (referential integrity) |
| `row_count` | the batch has an expected number of rows (catches empty deliveries) |
| `expression` | any custom cross-column pandas expression - the escape hatch |

Each rule carries a **severity**: `error` (a real stop that fails the batch) or `warn` (a smell that is logged but does not block). A rule that points at a missing column returns a graceful failing result - the run still completes and tells you which rule is misconfigured, rather than crashing.

## Tech Stack
Python, pandas, numpy, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints the summary table + failing-rule detail for the built-in sample):
```bash
python engine.py
```

## Learning Connection
Built while studying data quality and governance on the Data Quality & Governance arc.
Applies: declarative rule design, referential integrity and constraint checks, severity-based gating (error vs warn), and steward-first, explainable results - the same "rules as enforceable code, not wiki prose" idea behind contract testing and expectation frameworks.

## Impact Note
- **Who benefits:** data stewards, analytics engineers, and pipeline owners who need a batch validated before it feeds a report or model.
- **Potential risks:** rules are heuristics, not truth. A flagged value can be a legitimate exception (a genuinely large order), and a rule set only catches what it was told to check. Treat results as a prioritised review queue and keep a human owning the exceptions - never auto-reject or auto-delete rows on the engine's word alone.
