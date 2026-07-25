from __future__ import annotations

import streamlit as st

from metrics import (
    SAMPLE_YAML,
    compute,
    parse_metric_yaml,
    sample_dataframe,
    to_sql,
    validate_metrics,
)

st.set_page_config(page_title="Metrics Layer", layout="wide")
st.title("Metrics Layer / Semantic Definitions")
st.caption("One governed definition per metric — so every dashboard computes revenue the same way.")

col_def, col_out = st.columns([1, 1])

with col_def:
    st.subheader("Metric store (YAML)")
    yaml_text = st.text_area("Definitions", value=SAMPLE_YAML, height=380)

metrics = parse_metric_yaml(yaml_text)
issues = validate_metrics(metrics)

with col_out:
    st.subheader("Validation")
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Metrics", len(metrics))
    c2.metric("Errors", len(errors))
    c3.metric("Warnings", len(warnings))
    if errors:
        for i in errors:
            st.error(f"`{i.metric}`: {i.message}")
    if warnings:
        for i in warnings:
            st.warning(f"`{i.metric}`: {i.message}")
    if not issues:
        st.success("All metrics valid and consistent.")

if metrics:
    st.divider()
    st.subheader("Inspect a metric")
    names = [m.name for m in metrics]
    pick = st.selectbox("Metric", names)
    metric = next(m for m in metrics if m.name == pick)

    st.markdown(f"**{metric.label}** — {metric.description or '_no description_'}  ·  owner: `{metric.owner or 'unassigned'}`")
    grain = st.multiselect("Group by (grain)", metric.dimensions)

    st.markdown("**Canonical SQL**")
    st.code(to_sql(metric, grain), language="sql")

    st.markdown("**Computed on sample data**")
    df = sample_dataframe()
    try:
        result = compute(metric, df, grain)
        st.dataframe(result, use_container_width=True)
    except Exception as e:
        st.error(f"Cannot compute: {e}")

    with st.expander("Sample source data"):
        st.dataframe(df, use_container_width=True)
