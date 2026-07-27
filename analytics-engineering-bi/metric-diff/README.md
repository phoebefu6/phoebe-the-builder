# Metric Diff

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/metric-diff/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/metric-diff/demo.ipynb)

> "Conversion is up 5% this week!" — but is it a real shift or just noise? Metric Diff pairs a period-over-period delta with a significance test so you only act on changes that are real.

**Day 121 — Analytics Engineering & BI.** A raw delta lies; this puts a p-value and a confidence interval next to every change.

## Business Impact
- **Before:** Every week someone reacts to a metric that "moved" — reallocating budget, paging a team, changing a roadmap — off a swing that was pure sampling noise. And genuine shifts get dismissed as "probably nothing."
- **After:** One call returns the delta, a p-value, a 95% confidence interval, and a plain-English verdict. Green means act; grey means wait for more data.
- **Estimated ROI:** fewer false-alarm fire drills and fewer missed real movements — hours of misdirected analyst and eng time saved every week.

## What it does
Handles the two shapes almost every dashboard metric takes:

| Kind | Example | Test |
|------|---------|------|
| **Mean** — continuous, row-level samples | Avg Order Value, session length, revenue/user | Welch's t-test (unequal variance) + 95% CI |
| **Rate** — a proportion, successes / trials | conversion, click-through, churn | Two-proportion z-test + 95% CI |

The included sample data makes the point in one screen: Avg Order Value `+4.7%` is **real** (p≈0.004), while Checkout Conversion `+4.9%` is **noise** (p≈0.43). Same headline size, opposite decisions.

## Tech Stack
Python · numpy · scipy (`ttest_ind`, `norm`) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs and the delta-with-confidence-interval chart, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste your own two periods (or use the sample), tune the alpha level in the sidebar, and read the verdict.

## Learning Connection
Built while studying analytics engineering and applied statistics (the DS365 / stat-testing arc).
Applies: two-sample hypothesis testing, Welch vs pooled variance, two-proportion z-tests, and confidence intervals as a guard against noise-chasing.

## Impact Note
- **Who benefits:** analysts, PMs, and growth teams doing weekly/monthly metric reviews or reading A/B results.
- **Potential risks:** a significance test is not a causal claim — a "real" change still needs a cause. Small samples yield wide intervals; don't over-read a single week. p-values assume roughly independent observations, so seasonality and repeated peeking still need care.
