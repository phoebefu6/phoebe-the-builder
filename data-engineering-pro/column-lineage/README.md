# Column-Level Lineage Parser

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/column-lineage/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/column-lineage/demo.ipynb)

> Table-level lineage tells you *model B reads model A*. It can't tell you that renaming `orders.amount` will silently break `total_revenue` three hops away. This parses **column-level** lineage from plain SQL - no database, no `sqlparse`.

## Business Impact
- **Before:** Before touching a column, an engineer greps the dbt repo by hand, guesses the blast radius, and finds out what broke from a dashboard on fire.
- **After:** Point it at your `CREATE ... AS SELECT` models, get every column-to-column edge and the transitive blast radius of any change in seconds.
- **Estimated ROI:** ~2-3 hrs saved per schema change, plus the broken-dashboard incidents it prevents.

## Tech Stack
Python (standard library only for the core), Streamlit, pandas, matplotlib. Pure string parsing - runs anywhere, no warehouse connection.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** - pre-rendered with outputs and the lineage graph, or click the Colab/Binder badges above to run it live.

For the Streamlit app (editable SQL boxes + live impact selector):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works
1. **Find the target** - `CREATE TABLE/VIEW x AS SELECT ...` → `x` (bare `SELECT` uses the model name).
2. **Resolve aliases** - map `FROM`/`JOIN` aliases back to real table names.
3. **Split the select list** at the top level (commas inside `sum(a, b)` are ignored).
4. **For each output column**, collect its source columns and resolve `alias.col` to `table.col`.
5. **Answer questions** - `upstream_of`, `downstream_of`, and transitive `impact` (full blast radius).

Handles: qualified refs, JOINs, expressions over multiple columns, aggregates, aliasing. Flags `SELECT *` and ambiguous unqualified columns as warnings instead of guessing silently.

## Learning Connection
Built while studying data engineering lineage/observability patterns (OpenLineage, dbt column-level lineage, DataHub).
Applies: SQL parsing, dependency-graph construction, transitive impact analysis, DAG layout.

## Impact Note
- **Who benefits:** Data / analytics engineers doing schema changes, refactors, and audits.
- **Potential risks:** It's a lightweight parser, not a full SQL engine - CTEs, subqueries, and window functions resolve at a coarse grain, and `SELECT *` is approximate. Treat the output as a strong hint for review, not a guarantee. Validate against your warehouse before deleting anything.
