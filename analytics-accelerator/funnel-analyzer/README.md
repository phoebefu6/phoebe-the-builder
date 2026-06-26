# Funnel Analyzer

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/funnel-analyzer/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/funnel-analyzer/demo.ipynb)

> See exactly where users drop off — biggest leak called out, step-by-step conversion, and a segment comparison.

## Business Impact
- **Before:** "We're losing users somewhere" — no visibility into *which* step bleeds, so fixes are guesswork.
- **After:** Every funnel pinpoints the biggest drop-off step and shows how conversion differs by segment, turning a vague worry into a targeted fix.
- **Estimated ROI:** Focuses scarce eng/design effort on the one step that's actually leaking, instead of redesigning the whole flow.

## How it works
1. **Strict ordered funnel** — a user only counts at step *k* if they completed every earlier step, so the numbers are monotonically non-increasing and honest.
2. **Step metrics** — users at each step, conversion vs the top and vs the previous step, and users lost.
3. **Biggest leak** — the single step with the largest % drop-off is flagged: fix here first.
4. **Segment comparison** — overall conversion split by any column (device, plan, channel) to find *who* drops off.

Core is pure pandas — no API keys, runs standalone in a notebook or CI. The Streamlit app layers an interactive **Plotly** funnel on top.

## Tech Stack
Python · pandas · Plotly (interactive funnel) · Streamlit · matplotlib (notebook charts) · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app (interactive Plotly funnel):
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample event log** in the sidebar to try it — a 5-step e-commerce funnel (2,000 users) that converts ~10% end-to-end, with the biggest leak planted at **checkout** and a **mobile** disadvantage.

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: event-log modeling, ordered conversion funnels, drop-off / step-conversion math, segment comparison, and interactive viz with Plotly.

## Impact Note
- **Who benefits:** product, growth, and analytics teams diagnosing conversion problems.
- **Potential risks:** this funnel uses set-membership ("did the user ever do step X?"), not strict event timestamps, so it won't catch out-of-order or looping journeys — fine for most conversion funnels, but validate the step order matches your real user flow before acting on it.
