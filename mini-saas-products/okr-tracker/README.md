# OKR Tracker & Advisor

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/mini-saas-products/okr-tracker/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=mini-saas-products/okr-tracker/demo.ipynb)

> OKRs get set and forgotten — this tracks progress against pace and flags which key results are behind while there's still time to act.

## Business Impact
- **Before:** OKRs are written at the start of the quarter, then nobody looks until the end — when it's too late to change the outcome.
- **After:** Every KR is scored against how much of the period has elapsed. Behind-pace KRs get flagged as at-risk or off-track, with a clear "push vs. escalate" recommendation.
- **Estimated ROI:** Turns OKRs from a set-and-forget ritual into a weekly steering tool — misses get caught mid-quarter, not at the retro.

## Tech Stack
Python 3.10+, Streamlit, pandas, matplotlib. Pure-Python progress + pace logic (runs offline, no API keys).

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How It Works
1. **Progress** — fraction of the way from `start` to `target`, computed against the signed span so "lower is better" KRs (churn, latency) work correctly.
2. **Pace-based status** — compare progress to the fraction of the period elapsed. Gap ≥ 10% → at-risk; ≥ 25% → off-track. This is the whole idea: 40% done is fine at week 4 and a crisis at week 10.
3. **Objective roll-up** — an objective's status is its worst KR (off-track beats at-risk beats on-track).
4. **Advisor** — lists behind-pace KRs worst-first, with the gap and an action: "add focus this week" (at-risk) vs. "escalate and re-plan" (off-track).

In the sample (60% through the quarter), MRR and NPS are on pace, but enterprise logos, churn, and features-GA are all off-track — so the advisor says escalate those three now.

## Learning Connection
Built while studying goal-setting frameworks and progress tracking (Month 6: Mini SaaS Products).
Applies: normalized progress math, pace-vs-time status, and prescriptive (not just descriptive) reporting.

## Impact Note
- **Who benefits:** Founders, team leads, and anyone running quarterly OKRs.
- **Potential risks:** Linear pace assumes even progress, which some KRs don't follow (a launch lands all at once). Treat off-track flags as a prompt to check in, not an automatic failure; annotate KRs with non-linear expectations where needed.
