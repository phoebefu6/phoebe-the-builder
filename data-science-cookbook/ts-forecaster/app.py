from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from forecaster import backtest, holt_winters_add, sample_series

st.set_page_config(page_title="Time Series Forecaster", layout="wide")
st.title("General Time Series Forecaster")
st.caption('"Forecast any metric" — Holt-Winters (trend + seasonality) in pure numpy, with honest backtest accuracy.')

series = sample_series()

with st.sidebar:
    uploaded = st.file_uploader("Upload CSV (date,value) or use sample", type=["csv"])
    season = st.slider("Seasonal period", 1, 24, 12)
    horizon = st.slider("Forecast horizon", 3, 24, 12)
    alpha = st.slider("α (level)", 0.05, 0.95, 0.4)
    beta = st.slider("β (trend)", 0.0, 0.9, 0.1)
    gamma = st.slider("γ (season)", 0.0, 0.9, 0.3)

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    series = pd.Series(raw.iloc[:, 1].values, index=pd.to_datetime(raw.iloc[:, 0]), name=raw.columns[1])

y = series.values.astype(float)
fc = holt_winters_add(y, season, horizon, alpha=alpha, beta=beta, gamma=gamma)

st.caption(f"Method: **{fc.method}**")

# backtest if enough data
if len(y) > season + horizon:
    bt = backtest(y, season, min(horizon, max(3, len(y) // 4)), alpha=alpha, beta=beta, gamma=gamma)
    c1, c2 = st.columns(2)
    c1.metric("Backtest MAPE", f"{bt['mape']:.1f}%")
    c2.metric("Backtest RMSE", f"{bt['rmse']:.1f}")

# plot
fig, ax = plt.subplots(figsize=(11, 4.6))
hist_x = np.arange(len(y))
fut_x = np.arange(len(y), len(y) + horizon)
ax.plot(hist_x, y, color="#3b6fd6", label="history", marker="o", ms=3)
ax.plot(hist_x, fc.fitted, color="#888", ls="--", lw=1, label="fitted")
ax.plot(fut_x, fc.forecast, color="#c0553b", label="forecast", marker="o", ms=3)
ax.fill_between(fut_x, fc.lower, fc.upper, color="#c0553b", alpha=0.15, label="95% band")
ax.axvline(len(y) - 0.5, color="#aaa", ls=":", lw=1)
ax.set_title(f"{series.name}: {horizon}-step forecast", fontsize=13, weight="bold")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Forecast values")
future_idx = pd.date_range(series.index[-1], periods=horizon + 1, freq="MS")[1:] if hasattr(series.index, "freq") or True else range(horizon)
try:
    fc_df = pd.DataFrame({"forecast": np.round(fc.forecast, 1),
                          "lower": np.round(fc.lower, 1),
                          "upper": np.round(fc.upper, 1)}, index=future_idx)
except Exception:
    fc_df = pd.DataFrame({"forecast": np.round(fc.forecast, 1)})
st.dataframe(fc_df, use_container_width=True)
