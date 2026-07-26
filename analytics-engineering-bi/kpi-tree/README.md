# KPI Tree / Driver Decomposition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-engineering-bi/kpi-tree/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-engineering-bi/kpi-tree/demo.ipynb)

> "Why did revenue move?" — this splits a KPI's change across its drivers exactly, with no residual.

## Business Impact
- **Before:** A metric moves and the room guesses which driver caused it. Naive attribution leaves an unexplained residual nobody trusts.
- **After:** Model the KPI as a product of drivers and get an **exact** additive attribution — this much from users, this much from conversion, this much from price — that sums to the total change.
- **Estimated ROI:** turns metric post-mortems from debate into arithmetic; points action at the real lever.

## Tech Stack
Python · LMDI-I (log-mean divisia index) decomposition · waterfall visualization · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Enter before/after values for each driver; get the exact contribution of each, a one-line story, and a driver waterfall.

## How it works
1. **Model** the KPI as a product of drivers (`Revenue = Active users × Conversion × ARPU`).
2. **Decompose** the period-over-period change with LMDI-I — each driver's contribution is its log-change weighted by the log mean of the KPI.
3. **Guarantee** the contributions sum to the *exact* total change (residual ≈ 0), unlike naive "% change × base" which leaves interaction terms unexplained.
4. **Narrate + visualize** — biggest mover first, plus a start→drivers→end waterfall.

## Learning Connection
Built while studying **metric analytics & index decomposition** (LMDI from energy/economics, applied to product KPIs).
Applies: why multiplicative KPIs need log-based decomposition, and delivering an exact, additive answer instead of a hand-wavy one.

## Impact Note
- **Who benefits:** analysts, PMs, finance, execs doing metric post-mortems.
- **Potential risks:** decomposition explains *arithmetic* contribution, not *causation* — a driver can contribute mathematically while the real cause is upstream (a pricing change that also shifted conversion). Use it to focus the investigation, not end it; and non-positive driver values fall back to a simple mean, which is only approximate.
