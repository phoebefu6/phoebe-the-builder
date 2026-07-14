from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from factory import fit_transform, infer_plan, make_sample_data

ROLE_BADGE = {
    "numeric": "🔢",
    "categorical": "🏷️",
    "binary": "⚪",
    "datetime": "📅",
    "drop": "🚫",
}

st.set_page_config(page_title="Feature Factory", page_icon="🏭", layout="wide")
st.title("🏭 Feature Factory")
st.caption("Point it at a table. It infers each column's role, builds a leakage-safe "
           "sklearn ColumnTransformer, and hands you the model-ready feature matrix - "
           "no hand-written impute/scale/one-hot boilerplate.")

# --- Load data -------------------------------------------------------------
upload = st.file_uploader("Upload a CSV (or use the sample messy customer table)", type="csv")
if upload is not None:
    df = pd.read_csv(upload)
    st.success(f"Loaded {upload.name}: {len(df)} rows x {df.shape[1]} columns")
else:
    df = make_sample_data()
    st.info("Using the built-in sample: a messy mixed-type customer table with missing "
            "values, an id column, a datetime, and a high-cardinality text column.")

with st.expander("Preview raw data", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

target = st.selectbox("Target column (dropped from features)", ["(none)"] + list(df.columns),
                      index=(list(df.columns).index("churned") + 1) if "churned" in df.columns else 0)
target = None if target == "(none)" else target

# --- Inferred plan ---------------------------------------------------------
plan = infer_plan(df, target=target)
st.subheader("Inferred feature plan")
plan_rows = [{"": ROLE_BADGE.get(p.role, ""), "Column": p.column,
              "Role": p.role, "Why": p.reason} for p in plan.plans]
st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

roles = pd.Series([p.role for p in plan.plans]).value_counts()
cols = st.columns(len(ROLE_BADGE))
for c, role in zip(cols, ROLE_BADGE):
    c.metric(f"{ROLE_BADGE[role]} {role}", int(roles.get(role, 0)))

# --- Build + transform -----------------------------------------------------
st.divider()
if st.button("🏭 Build ColumnTransformer & transform", type="primary"):
    try:
        features, ct = fit_transform(df, plan)
    except Exception as exc:  # noqa: BLE001 - surface any sklearn build error to the user
        st.error(f"Could not build the pipeline: {exc}")
        st.stop()

    st.subheader("Model-ready feature matrix")
    st.write(f"**{df.shape[1]} raw columns → {features.shape[1]} numeric features**, "
             f"0 missing values, all scaled/encoded.")
    st.dataframe(features.head(10), use_container_width=True)

    csv = features.to_csv(index=False).encode()
    st.download_button("Download feature matrix (CSV)", csv,
                       "features.csv", "text/csv")

    with st.expander("Generated pipeline (paste into your project)"):
        st.code(repr(ct), language="python")
else:
    st.caption("Click build to fit the transformer and see the expanded feature matrix.")
