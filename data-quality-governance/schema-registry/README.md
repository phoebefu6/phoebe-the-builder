[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-quality-governance/schema-registry/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-quality-governance/schema-registry/demo.ipynb)

# Schema Registry

> A producer changes a column type and three downstream jobs break silently - we find out from the dashboard, not before. Register every schema version and check a proposed change for compatibility BEFORE it ships.

## Business Impact
- **Before:** A producer quietly narrows a type or makes a field required. Nothing warns anyone; the break surfaces hours later as a wrong number on a dashboard, and someone spends the afternoon bisecting jobs to find it.
- **After:** Every proposed schema is diffed against the latest registered version. The engine returns a compatibility verdict - BACKWARD, FORWARD, FULL, or BREAKING - with the specific field change that drives it, so the break is caught at review time, not at read time.
- **Estimated ROI:** ~1 prevented incident per producer per month, plus 2-3 hours/week of "why did this job fail" investigation avoided across the downstream team.

## Compatibility model (explainable heuristic)
- **Add optional/nullable field** -> backward-compatible (old data has no value, readers tolerate it).
- **Add required field, no default** -> forward-breaking (old data / old producers omit it).
- **Remove a field** -> backward-breaking (consumers still expecting it lose data), forward-safe.
- **Narrow a type** (long -> int) or change it incompatibly (string -> int) -> BREAKING both ways. Widening (int -> long -> double) is safe.
- **Nullable -> required (non-null)** -> BREAKING (backward): new consumers reject the nulls in old data.
- **Required -> nullable** -> safe.

Verdict roll-up: no direction broken = FULL; only forward broken = BACKWARD; only backward broken = FORWARD; both broken = BREAKING. First-ever schema (no prior version) = "initial, compatible". The rules live at the top of `registry.py` (`_WIDENS_TO` lattice) and every `Change` carries the WHY.

## Tech Stack
Python, pandas, Streamlit, matplotlib. Fully offline - no API keys, no external services.

## Demo

**[Run the interactive demo notebook ->](demo.ipynb)** - pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

CLI quick look (prints version history + a compatibility report for the sample proposals):
```bash
python registry.py
```

## Learning Connection
Built while studying schema evolution and data contracts on the Data Quality & Governance arc.
Applies: Confluent/Avro-style compatibility modes (BACKWARD / FORWARD / FULL), type-promotion lattices, and treating a schema as a contract between producer and consumer rather than an implementation detail.

## Impact Note
- **Who benefits:** data producers who need a safe way to evolve a table, and every downstream consumer who reads it.
- **Potential risks:** the compatibility rules are a heuristic, not a proof. A change flagged BREAKING may be intentional and coordinated; one flagged safe could still surprise a consumer relying on undocumented behavior. A human confirms intent before a breaking change ships - treat the verdict as a review gate, not an auto-approve.
