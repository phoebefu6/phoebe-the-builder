# General Time Series Forecaster

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/data-science-cookbook/ts-forecaster/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=data-science-cookbook/ts-forecaster/demo.ipynb)

> "Forecast any metric" — Holt-Winters (trend + seasonality) in pure numpy, with honest backtest accuracy.

## Business Impact
- **Before:** Forecasts live in a spreadsheet as a straight-line guess, or need a heavy forecasting library nobody maintains.
- **After:** A dependency-light Holt-Winters forecaster captures trend and seasonality, produces uncertainty bands, and reports backtested accuracy.
- **Estimated ROI:** trustworthy short-horizon forecasts for planning, with a known error bar.

## Tech Stack
Python · numpy (Holt-Winters triple exponential smoothing, hand-rolled) · backtest (MAPE/RMSE) · Streamlit · matplotlib

## Demo
**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered, or use the badges above.
```bash
pip install -r requirements.txt
streamlit run app.py
```
Tune α/β/γ, season, and horizon; see history, fit, forecast with a 95% band, and backtest MAPE.

## How it works
1. **Decompose** the series into level (α), trend (β), and seasonal (γ) components via exponential smoothing.
2. **Forecast** forward, projecting all three; the uncertainty band widens with the horizon.
3. **Fallback** to Holt's linear method when there's no seasonal period.
4. **Backtest** — hold out the last points and score MAPE/RMSE for honest accuracy.

## Learning Connection
Built while studying **time series forecasting**. Applies: exponential smoothing intuition, always backtesting a forecast, and knowing when a light classical model beats a heavy one.

## Impact Note
- **Who benefits:** analysts, finance, ops doing metric planning.
- **Potential risks:** exponential smoothing extrapolates existing patterns — it **can't foresee regime changes, launches, or shocks**, and the fixed α/β/γ aren't auto-tuned here. Trust short horizons over long ones, always read the backtest error, and widen bands for volatile metrics.
