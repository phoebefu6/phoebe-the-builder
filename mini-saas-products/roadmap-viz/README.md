# Product Roadmap Visualizer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/roadmap-viz/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/roadmap-viz/demo.ipynb)

> Roadmaps are PowerPoints that go stale — generate the roadmap from data instead: a Gantt-style timeline, grouped by lane, colored by status, with a "today" line.

## Business Impact
- **Before:** The roadmap lives in a slide that's out of date the day after the offsite; updating it means fiddling with shapes.
- **After:** The roadmap is a table of items. Edit a row, re-render, and the timeline is always current — with status and schedule visible in one view.
- **Estimated ROI:** Kills the recurring "update the roadmap slide" chore and gives every standup a single accurate picture.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Pure-Python timeline rendering (runs offline, no API keys).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Items as data** — each initiative is `name, lane, start, end, status`.
2. **Timeline render** — one horizontal bar per item on a real date axis, grouped by lane, colored by status (done / in-progress / planned / at-risk).
3. **Today marker** — a dashed vertical line so anyone can instantly read what's late, active, or upcoming.
4. **Flags** — at-risk items and behind-schedule bars surface as the standup conversation.

The sample roadmap spans three lanes (Platform, Growth, Data) across 2026; with the today line at April 1, the at-risk referral program stands out in red as the thing to discuss.

## Learning Connection
Built while studying product roadmapping and timeline visualization (Month 6: Mini SaaS Products).
Applies: data-driven Gantt rendering, matplotlib date axes, and status/schedule dual-encoding.

## Impact Note
- **Who benefits:** PMs and team leads who maintain roadmaps and are tired of stale slides.
- **Potential risks:** A tidy timeline can imply false precision — planned end dates are estimates, not commitments. Keep statuses honest and re-render often; a beautiful roadmap built on optimistic dates is still wrong.
