# ERD Generator from SQL

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-infra-toolkit/erd-generator/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-infra-toolkit/erd-generator/demo.ipynb)

> "We don't have database diagrams" — paste your CREATE TABLE statements and get an instant ER diagram.

## Business Impact
- **Before:** No database documentation; new team members reverse-engineer schema by reading SQL dumps
- **After:** Paste SQL, get a visual ERD + Mermaid code for README/wiki in seconds
- **Estimated ROI:** 2-4 hours saved per schema review or onboarding session

## Tech Stack
Python, Streamlit, Regex-based SQL parsing, Mermaid syntax generation, Matplotlib

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Parses standard `CREATE TABLE` syntax (MySQL, PostgreSQL, SQLite)
- Detects primary keys, foreign keys (inline and standalone), NOT NULL constraints
- Generates Mermaid `erDiagram` syntax (GitHub-native rendering)
- Generates Graphviz DOT syntax
- Matplotlib visualization for notebooks
- Handles self-referencing foreign keys (e.g., employee → manager)

## Learning Connection
Built while studying the Data Engineer Career Track on DS365.
Applies: SQL DDL parsing, data governance documentation, schema visualization (DrawDB/erdantic concepts)

## Impact Note
- **Who benefits:** Data engineers, DBAs, new team members onboarding to a codebase
- **Potential risks:** Parser handles standard SQL; non-standard DDL extensions may need manual review
