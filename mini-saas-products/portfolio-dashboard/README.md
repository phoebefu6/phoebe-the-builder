# Full Portfolio Dashboard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/portfolio-dashboard/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/portfolio-dashboard/demo.ipynb)

> Showcase all 60 projects in one view — the portfolio's final build is the view of the portfolio itself.

## Business Impact
- **Before:** 60 builds spread across six folders; anyone evaluating the portfolio has to click through directories to see the scope.
- **After:** One dashboard with the burn-up chart, pace stats, product-line breakdown, and a searchable catalog linking every build.
- **Estimated ROI:** The whole 6-month arc is legible in 10 seconds — for recruiters, clients, and future-Phoebe.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Regex parser over TRACKER.md (the portfolio's own source of truth) — no API keys, runs offline.

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app (reads the live TRACKER.md):
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Parse** — two regexes turn TRACKER.md into structured builds: month headings give the product line and folder; checklist lines give day, slug, title, status, and completion date. Each build carries its GitHub URL.
2. **Stats** — completed count, calendar span, builds-per-day pace, active build days, busiest day, longest gap, per-product-line totals.
3. **Burn-up** — cumulative completions against the ideal 1-build-per-day line, showing the real rhythm (sprints of 11 builds in a day, gaps of 4).
4. **Catalog** — searchable table of all 60 builds with clickable links into the repo.

Final numbers: **60/60 builds in 35 calendar days** (2026-06-07 → 2026-07-11), six product lines at 10/10, busiest day 2026-07-07 with 11 builds.

## Learning Connection
Built as the capstone of Month 6 (Digital Product Management, Agile Essentials).
Applies: parsing your own project artifacts as data, burn-up/velocity visualization, and dashboard design.

## Impact Note
- **Who benefits:** Anyone evaluating the portfolio — and any builder who tracks work in a Markdown checklist and wants it to visualize itself.
- **Potential risks:** Stats are only as honest as the tracker; backfilled dates would silently inflate streaks. The parser trusts the file.
