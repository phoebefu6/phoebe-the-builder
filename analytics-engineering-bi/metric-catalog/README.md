# Metric Catalog & Ownership

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/metric-catalog/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/metric-catalog/demo.ipynb)

> Which metrics exist, and who owns them? A registry with tiers, dependencies, and automated governance that flags the metrics nobody is minding.

**Finale of Month 11 — Analytics Engineering & BI.** Day 101 defined metrics once; this is where they live and get governed.

## Business Impact
- **Before:** Nobody can answer "how many metrics do we have?" or "who owns this number?" Definitions drift, owners leave, and dependencies break silently.
- **After:** A searchable catalog with an owner, tier, and dependencies per metric — plus automated governance that flags unowned, stale, and broken-dependency metrics before they cause an incident.
- **Estimated ROI:** clear accountability, faster incident response, and a board-ready governance health number.

## Tech Stack
Python · metric registry (dataclasses) · governance rule engine (ownership / staleness / dependency checks) · JSON persistence · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Browse and filter the catalog, see governance issues, and track ownership % and tier distribution.

## How it works
1. **Register** metrics with definition, owner, team, tier (1 = board-critical), dependencies, and last-reviewed date.
2. **Search** by text, team, tier, or status.
3. **Govern** automatically — flag active metrics with no owner, stale/never-reviewed metrics, and (high severity) metrics depending on missing or deprecated metrics.
4. **Report** health — ownership %, open issues by severity, tier distribution — and persist to JSON.

## Learning Connection
Built while studying **data governance & metric ownership** (data catalog / semantic-layer governance).
Applies: turning governance from a wiki page into automated checks, tiering by criticality, and closing the Month-11 loop (define → self-serve → govern).

## Impact Note
- **Who benefits:** data leaders, analytics engineers, governance/BI teams.
- **Potential risks:** a catalog is only as honest as its entries — if `last_reviewed` isn't kept current or owners aren't real, the governance signal is noise. Wire it to your metrics-layer source (Day 101) so entries can't silently drift from the actual definitions.
