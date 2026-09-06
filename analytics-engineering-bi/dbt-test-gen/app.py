from __future__ import annotations

import pandas as pd
import streamlit as st
from testgen import (
    coverage,
    generate_model_tests,
    sample_dataframe,
    to_schema_yml,
)

st.set_page_config(page_title="dbt Test Generator", layout="wide")
st.title("dbt Test Generator")
st.caption("Point it at a model, get a paste-ready schema.yml — so your models finally have tests.")

with st.sidebar:
    st.subheader("Settings")
    model_name = st.text_input("Model name", value="orders")
    description = st.text_input("Model description", value="Order line items")
    uploaded = st.file_uploader("Upload a CSV (or use sample)", type=["csv"])
    unique_threshold = st.slider("Uniqueness threshold for `unique`", 0.80, 1.0, 0.99)
    category_max = st.slider("Max distinct values for `accepted_values`", 3, 25, 12)

df = pd.read_csv(uploaded) if uploaded is not None else sample_dataframe()
st.caption(f"Profiling {df.shape[0]} rows × {df.shape[1]} columns.")

cts = generate_model_tests(df, unique_threshold=unique_threshold, category_max=category_max)
cov = coverage(cts)

c1, c2, c3 = st.columns(3)
c1.metric("Test coverage", f"{cov['pct']}%")
c2.metric("Columns tested", f"{cov['tested']}/{cov['total']}")
c3.metric("Total tests", sum(cov["by_test"].values()))

st.subheader("Suggested tests")
rows = []
for c in cts:
    rows.append(
        {
            "column": c.name,
            "tests": ", ".join(c.tests) or "—",
            "why": "; ".join(f"{k}: {v}" for k, v in c.reason.items()),
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Generated schema.yml")
yml = to_schema_yml(model_name, cts, description)
st.code(yml, language="yaml")
st.download_button("Download schema.yml", data=yml, file_name="schema.yml", mime="text/yaml")

with st.expander("Sample data"):
    st.dataframe(df, use_container_width=True)
