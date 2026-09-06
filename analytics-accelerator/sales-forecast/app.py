"""Sales Forecast Dashboard — Streamlit UI.

Upload a sales history CSV (date + value) or use the built-in sample, pick a
horizon, and get a forecast with a confidence band, a backtest accuracy score,
and a downloadable forecast table — no spreadsheet required.
"""
from __future__ import annotations

import io
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from forecast import forecast_sales

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Sales Forecast Dashboard", page_icon="📈", layout="wide")

st.title("📈 Sales Forecast Dashboard")
st.caption(
    "Get sales forecasts out of someone's spreadsheet — upload history, pick a "
    "horizon, and get a forecast with a confidence band and an honest accuracy check."
)

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Sales history CSV (date + value)", type=["csv"])
    use_sample = st.button("Use sample sales data")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    from forecast import sample_sales

    df = sample_sales()
else:
    st.info(
        "Upload a CSV with a date column and a numeric sales column, or click "
        "**Use sample sales data** to try it (36 months)."
    )
    st.stop()

st.subheader("Raw data")
st.dataframe(df.head(20), use_container_width=True)

# Let the user confirm/override the detected columns
with st.sidebar:
    st.header("Forecast")
    date_col = st.selectbox("Date column", df.columns, index=0)
    num_cols = [c for c in df.columns if c != date_col]
    value_col = st.selectbox("Value column", num_cols, index=0)
    periods = st.slider("Periods to forecast", 3, 36, 12)

try:
    res = forecast_sales(df, periods=periods, date_col=date_col, value_col=value_col)
except Exception as e:
    st.error(f"Could not forecast: {e}")
    st.stop()

# --- Headline metrics ------------------------------------------------------
m = res.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Engine", res.engine)
c2.metric("Frequency", res.freq)
mape = m.get("mape_pct")
c3.metric("Backtest MAPE", f"{mape:.1f}%" if mape is not None else "n/a")
c4.metric("Forecast avg", f"{m['forecast_mean']:,.0f}")
st.caption(
    "MAPE = mean absolute % error on a held-out tail (train on the past, test on "
    "what came next). Lower is better; under ~10% is strong for sales data."
)

# --- Forecast chart --------------------------------------------------------
st.subheader("Forecast")
fig, ax = plt.subplots(figsize=(11, 5))
hist = res.history
ax.plot(hist["ds"], hist["y"], color="#1F2937", label="History", linewidth=2)
ax.plot(res.forecast["ds"], res.forecast["yhat"], color="#4F46E5",
        label="Forecast", linewidth=2, marker="o", markersize=3)
ax.fill_between(
    res.forecast["ds"],
    res.forecast["yhat_lower"],
    res.forecast["yhat_upper"],
    color="#4F46E5",
    alpha=0.18,
    label="95% interval",
)
ax.axvline(hist["ds"].iloc[-1], color="#9CA3AF", linestyle="--", linewidth=1)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# --- Forecast table + download ---------------------------------------------
st.subheader("Forecast table")
st.dataframe(res.forecast, use_container_width=True)

buf = io.StringIO()
res.forecast.to_csv(buf, index=False)
st.download_button(
    "⬇️ Download forecast CSV",
    buf.getvalue(),
    file_name="sales_forecast.csv",
    mime="text/csv",
)
