from __future__ import annotations

"""Streamlit UI: drop in a CSV, get an instant EDA - overview, per-column
profiles, distributions, correlations, and quality flags.
"""

import numpy as np
import pandas as pd
import streamlit as st
from profiler import numeric_correlations, profile_dataframe

st.set_page_config(page_title="Auto-EDA Dashboard", page_icon="🔎", layout="wide")

SEV_EMOJI = {"error": "🔴", "warning": "🟡", "info": "🔵"}

st.title("🔎 Auto-EDA Dashboard")
st.caption("Stop hand-profiling every new dataset. Upload a CSV and get shape, missingness, dtypes, distributions, correlations, and quality flags in seconds.")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    use_sample = st.checkbox("Use sample data", value=uploaded is None)


def _sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": rng.integers(18, 75, n),
        "plan": rng.choice(["free", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1]),
        "monthly_spend": np.round(rng.gamma(2, 30, n), 2),
        "region": rng.choice(["NA", "EU", "APAC"], n),
        "signup_source": rng.choice(["organic", "ads", "referral", None], n, p=[0.4, 0.3, 0.2, 0.1]),
        "is_active": rng.choice([True, False], n, p=[0.7, 0.3]),
        "notes": [None] * n,  # an all-empty column to trip a quality flag
    })


if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = _sample()
else:
    st.info("Upload a CSV or tick **Use sample data** to begin.")
    st.stop()

prof = profile_dataframe(df)
ov = prof["overview"]

c = st.columns(5)
c[0].metric("Rows", f"{ov['rows']:,}")
c[1].metric("Columns", ov["columns"])
c[2].metric("Missing cells", f"{ov['missing_cells_pct']}%")
c[3].metric("Duplicate rows", ov["duplicate_rows"])
c[4].metric("Memory", f"{ov['memory_kb']} KB")

# Quality flags first - the actionable part.
if prof["flags"]:
    st.subheader("Quality flags")
    for f in prof["flags"]:
        st.write(f"{SEV_EMOJI.get(f['severity'], '•')} **{f['column']}** - {f['message']}")
else:
    st.success("No quality flags - data looks clean.")

st.subheader("Column profiles")
col_table = pd.DataFrame([
    {"column": p["name"], "kind": p["kind"], "dtype": p["dtype"],
     "missing %": p["missing_pct"], "unique": p["unique"], "unique %": p["unique_pct"]}
    for p in prof["columns"]
])
st.dataframe(col_table, use_container_width=True, hide_index=True)

st.subheader("Distributions")
cols = st.columns(2)
numeric_cols = [p["name"] for p in prof["columns"] if p["kind"] == "numeric"]
cat_cols = [p["name"] for p in prof["columns"] if p["kind"] in {"categorical", "boolean"}]

with cols[0]:
    if numeric_cols:
        pick = st.selectbox("Numeric column", numeric_cols)
        st.bar_chart(np.histogram(df[pick].dropna(), bins=20)[0])
        st.caption(f"Histogram of {pick} (20 bins)")
with cols[1]:
    if cat_cols:
        pickc = st.selectbox("Categorical column", cat_cols)
        st.bar_chart(df[pickc].value_counts().head(15))

corr = numeric_correlations(df)
if not corr.empty:
    st.subheader("Numeric correlations")
    st.dataframe(corr.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1),
                 use_container_width=True)

st.caption("Profiling core lives in `profiler.py` - reusable as an Auto-EDA app on the platform shell.")
