# Sales Forecast Dashboard

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/analytics-accelerator/sales-forecast/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=analytics-accelerator/sales-forecast/demo.ipynb)

> Get sales forecasts out of someone's spreadsheet — a reproducible forecast with a confidence band and an honest accuracy check.

## Business Impact
- **Before:** The forecast is a tab in one person's spreadsheet — unreproducible, undocumented, and impossible to trust or update.
- **After:** Upload the history, get a forecast with a 95% confidence band and a backtest MAPE, re-runnable the moment data refreshes.
- **Estimated ROI:** Replaces a recurring manual forecasting ritual; the accuracy score turns "trust me" into a defensible number.

## How it works
1. **Detect columns** — finds the date and numeric value columns and infers the frequency (daily / weekly / monthly).
2. **Fit** — Holt-Winters exponential smoothing with additive trend and seasonality (added automatically when there's ≥ 2 full cycles of history).
3. **Backtest** — trains on all but a held-out tail, predicts it, and reports MAPE so you know how much to trust the forecast.
4. **Forecast** — projects N periods ahead with a 95% confidence band, charted and downloadable.

The primary engine is **statsmodels Holt-Winters** — light, reliable, always available. If you install the optional `prophet` package, pass `engine="prophet"` to swap it in.

Pure statsmodels + pandas — no API keys, runs standalone in a notebook or CI.

## Tech Stack
Python · statsmodels (Holt-Winters / Exponential Smoothing) · pandas · Streamlit · matplotlib · Docker · *(optional: Prophet)*

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Use sample sales data** in the sidebar to try it (36 months, trend + yearly seasonality). The sample backtests at **MAPE ≈ 2.6%**.

## Learning Connection
Built while studying **Streamlit** and **AWS Cloud Technical Essentials** (Month 3 of the FDE track).
Applies: time-series forecasting, seasonality and trend decomposition, train/test backtesting (MAPE), confidence intervals, and pluggable model backends.

## Impact Note
- **Who benefits:** sales ops, finance, and revenue teams who need a defensible forward number.
- **Potential risks:** forecasts assume the future resembles the past — a price change, a new competitor, or a demand shock will break them. The confidence band reflects in-sample noise, not regime change. Treat the forecast as a baseline to adjust with business judgment, not a commitment, and always read the MAPE before trusting the number.
