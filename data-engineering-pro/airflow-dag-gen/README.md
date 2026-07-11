# Airflow DAG Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-engineering-pro/airflow-dag-gen/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-engineering-pro/airflow-dag-gen/demo.ipynb)

> Writing DAG boilerplate is repetitive — describe the pipeline in ~30 lines of YAML and get a validated, ready-to-deploy Airflow DAG.

## Business Impact
- **Before:** Every new pipeline means copy-pasting operator boilerplate; retries, owners, and tags drift across DAGs; cycles and typo'd dependencies surface only at deploy time.
- **After:** Teams write *what* the pipeline does in YAML; the generator owns *how* Airflow wants it said — with cycle detection and schema checks failing the build before code exists.
- **Estimated ROI:** New pipeline scaffold in minutes; org-wide DAG conventions enforced by construction.

## Tech Stack
Python 3.10+, Jinja2, PyYAML, Streamlit, matplotlib. Supports bash/python/sql task types. Runs fully offline (Airflow not required to generate).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **YAML schema** — `dag_id`, `schedule`, `start_date`, retry defaults, and a `tasks` list (`id`, `type`: bash|python|sql, payload, optional `depends_on`).
2. **CI-style validation first** — duplicate/invalid ids, unknown types, missing payloads, ghost dependencies, and dependency cycles (DFS coloring) block generation with specific errors.
3. **Jinja render** — operators plus `a >> b` edges; imports only for operator types actually used; generated file carries a "do not edit by hand" header.
4. **Graph view** — topological levels draw the DAG left-to-right, the Airflow-UI view before you deploy.

Sample: a 5-task nightly sales pipeline (parallel extracts → transform → SQL load → dashboard refresh) renders to 54 lines of `ast.parse`-clean Python.

## Learning Connection
Built while studying Airflow patterns (Month 7: Data Engineering Pro).
Applies: config-driven code generation, DAG cycle detection, and template-enforced platform conventions.

## Impact Note
- **Who benefits:** Data engineering teams standardizing many similar pipelines; analysts who can describe a pipeline but not write operator code.
- **Potential risks:** Generated code that's edited by hand forks from its config — regeneration overwrites edits; keep YAML as the single source of truth.
