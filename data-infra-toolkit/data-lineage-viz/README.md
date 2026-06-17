# Data Lineage Visualizer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/data-lineage-viz/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/data-lineage-viz/demo.ipynb)

> Paste SQL, see what breaks when you change a table.

## Business Impact
- **Before:** Engineers change or drop a table and find out it broke a dashboard from an angry Slack message. Manual tracing through dozens of SQL files.
- **After:** Paste the SQL, get a dependency graph + instant upstream/downstream impact for any table.
- **Estimated ROI:** ~2-3 hours/week saved on impact tracing + avoided production breakage incidents.

## Tech Stack
Python · networkx · matplotlib · Streamlit · Docker

## What It Does
- Parses `CREATE TABLE AS`, `CREATE VIEW AS`, and `INSERT INTO ... SELECT` statements
- Extracts `(source -> target)` table dependency edges (handles comments, schema prefixes, JOINs, self-refs)
- Builds a directed lineage graph
- **Impact analysis:** for any table, lists upstream sources and downstream consumers (what breaks if you change it)
- Highlights impact on the graph: red = changed, yellow = breaks downstream, green = upstream sources

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Learning Connection
Built while studying the **Data Engineer Career Track (DS365)** and **Advanced SQL**.
Applies: data lineage, dependency graphs, and impact analysis — the manual version of what OpenMetadata / DataHub automate.

## Impact Note
- **Who benefits:** data engineers, analytics engineers, anyone owning a warehouse who fears silent breakage.
- **Potential risks:** regex-based parsing covers common patterns, not every SQL dialect (CTEs and subqueries are simplified). Treat output as a fast first-pass map, not a guaranteed-complete lineage — verify before destructive changes.
