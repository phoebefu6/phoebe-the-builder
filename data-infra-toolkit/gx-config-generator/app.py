from __future__ import annotations

"""Streamlit UI: upload a CSV, auto-profile it, download a GX expectation suite."""

import io

import pandas as pd
import streamlit as st

from profiler import generate_suite, suite_to_json, summarize

st.set_page_config(page_title="GX Config Generator", page_icon="✅", layout="wide")

st.title("✅ Great Expectations Config Generator")
st.caption("Point at a CSV → auto-profile every column → download a ready-to-run expectation suite. No more hand-writing validation configs.")

with st.sidebar:
    st.header("Settings")
    suite_name = st.text_input("Suite name", value="auto_generated_suite")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    use_sample = st.checkbox("Use sample data", value=uploaded is None)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": range(1001, 1011),
            "customer_email": [f"user{i}@example.com" for i in range(10)],
            "amount": [12.5, 99.0, 5.25, 250.0, 18.75, 42.0, 7.5, 310.0, 21.0, 64.5],
            "status": ["paid", "paid", "refunded", "paid", "pending",
                       "paid", "refunded", "paid", "pending", "paid"],
            "region": ["NA", "EU", "NA", "APAC", "EU", "NA", "EU", "APAC", "NA", "EU"],
            "discount_pct": [0, 10, 0, 5, None, 0, 15, 0, None, 5],
        }
    )


df: pd.DataFrame | None = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = _sample_frame()

if df is None:
    st.info("Upload a CSV or tick **Use sample data** to begin.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)
st.caption(f"{len(df):,} rows × {len(df.columns)} columns")

suite = generate_suite(df, suite_name=suite_name)

left, right = st.columns([1, 2])
with left:
    st.subheader("Coverage")
    st.dataframe(summarize(suite), use_container_width=True, hide_index=True)
    st.metric("Total expectations", len(suite["expectations"]))

with right:
    st.subheader("Generated suite")
    suite_json = suite_to_json(suite)
    st.code(suite_json, language="json")
    st.download_button(
        "⬇ Download expectation suite (JSON)",
        data=suite_json,
        file_name=f"{suite_name}.json",
        mime="application/json",
    )

with st.expander("How to use this with Great Expectations"):
    st.markdown(
        """
1. Save the downloaded JSON into your GX project under
   `great_expectations/expectations/<suite_name>.json`.
2. Run a checkpoint against it:
   ```python
   import great_expectations as gx
   ctx = gx.get_context()
   ctx.run_checkpoint(checkpoint_name="my_checkpoint")
   ```
3. Review and tighten the auto-generated bounds - this is a **starting point**,
   not a final contract.
"""
    )
