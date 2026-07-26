# Dashboard Spec Generator

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/dashboard-spec/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/dashboard-spec/demo.ipynb)

> Dashboards built from vague asks — this reads the data's shape and recommends the right charts, with the reasoning.

## Business Impact
- **Before:** "Build me a dashboard" with no spec produces a pie chart of 30 categories and three redundant tables.
- **After:** The tool classifies columns, applies encoding best practices, and returns a justified dashboard spec (chart type + encoding + rationale) — rendered as a real dashboard.
- **Estimated ROI:** faster, better-designed first-draft dashboards; fewer rebuild cycles from vague requirements.

## Tech Stack
Python · column role detection (temporal / categorical / measure) · encoding-rule chart recommender · JSON dashboard spec · matplotlib rendering · Streamlit · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Upload a CSV (or use the sample); see detected column roles, a rendered recommended dashboard, and a downloadable `spec.json`.

## How it works
1. **Classify** each column: temporal (name pattern or parseable dates), measure (numeric, non-id), categorical (low-cardinality text), or high-card id.
2. **Recommend** panels from encoding rules: temporal + measure → line; category + measure → **bar (not pie)**; two categories + measure → grouped bar; two measures → scatter; one measure → histogram; key measures → KPI cards.
3. **Emit** a portable JSON spec and render the charts as proof.

## Learning Connection
Built while studying **data visualization & chart selection** (encoding theory — Cleveland/Mackinlay effectiveness).
Applies: letting data structure drive chart choice, and the discipline of attaching a *rationale* to every visual decision.

## Impact Note
- **Who benefits:** analysts, PMs, and anyone handed "make a dashboard" without a spec.
- **Potential risks:** recommendations come from column *shape*, not business intent — the right chart for the data isn't always the right chart for the question. Treat the spec as a strong first draft, and pair titles/units with a governed metrics layer so panels are labeled correctly.
