# Data Quality Scorecard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/data-quality-scorecard/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/data-quality-scorecard/demo.ipynb)

> Stop guessing how bad your data is — get a 0-100 score, a letter grade, and a ranked list of what to fix first.

## Business Impact
- **Before:** "Is our data good enough?" gets a shrug. No baseline, no way to track whether quality is improving.
- **After:** Every dataset gets a defensible score, a per-dimension breakdown, and a prioritized fix list — the start of a data-quality SLA.
- **Estimated ROI:** Catches the missing values, duplicates, and broken categories that silently corrupt every downstream dashboard and model.

## How it works
Each column is checked against six Great-Expectations-inspired dimensions, then rolled up into a single weighted score:

| Dimension | Asks | Weight |
|-----------|------|--------|
| **Completeness** | How few missing values? | 25% |
| **Validity** | Right type / range / format (emails, negatives, finiteness)? | 25% |
| **Uniqueness** | How few duplicate rows? | 15% |
| **Consistency** | Categories not fragmented by case/whitespace (`USA` vs `usa`)? | 15% |
| **Timeliness** | Dates not stale or implausibly future-dated? | 10% |
| **Distribution** | Numeric columns not dominated by outliers (IQR)? | 10% |

Score → grade: A (≥90), B (≥80), C (≥70), D (≥60), F (<60). The worst-scoring checks are surfaced as a ranked "fix these first" list.

Native pandas + numpy — no heavy `great_expectations` dependency, no API keys, runs standalone in a notebook or CI.

## Tech Stack
Python · pandas · numpy · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use messy sample data** in the sidebar to try it — a dataset with planted problems (invalid emails, fragmented categories, missing values, outliers, duplicates) that scores **~87.9 / grade B**.

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: data profiling, validation rule design, weighted scoring/rollups, and translating raw checks into a prioritized remediation list.

## Impact Note
- **Who benefits:** data engineers, analysts, and governance teams who need a quality baseline and a fix backlog.
- **Potential risks:** the score is a heuristic with opinionated weights — a high grade doesn't guarantee the data is *correct*, only that it passes structural checks (a perfectly-formatted wrong number still scores well). Tune the weights and thresholds to your domain, and treat the grade as a screening signal, not a certification.
