# Cohort Analysis Tool

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/cohort-analysis/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/cohort-analysis/demo.ipynb)

> "Retention analysis takes our analyst 2 days" — turn a raw event log into a cohort retention heatmap in seconds.

## Business Impact
- **Before:** Analyst manually pivots signup dates against activity logs in spreadsheets, ~2 days per cycle.
- **After:** Upload a CSV (`user_id`, `event_date`) and get a retention heatmap instantly.
- **Estimated ROI:** ~16 hours/month saved per analyst running monthly retention reviews.

## Tech Stack
Python, pandas, Streamlit, matplotlib, seaborn

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Learning Connection
Built while studying Streamlit and AWS Cloud Technical Essentials.
Applies: cohorting logic with pandas `groupby`/`pivot`, retention-rate calculation, heatmap visualization.

## Impact Note
- **Who benefits:** Growth/product analysts, founders tracking retention, customer success teams.
- **Potential risks:** Small cohorts (low user counts) produce noisy retention percentages — flag sample size before trusting month-over-month comparisons.
