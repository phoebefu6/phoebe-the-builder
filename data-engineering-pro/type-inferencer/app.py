"""Streamlit UI: paste or upload a CSV, get DDL you can defend plus the findings behind it."""

from __future__ import annotations

import csv
import io
from typing import Dict, List

import pandas as pd
import streamlit as st
from type_infer import (
    BUCKETS,
    ColumnType,
    Policy,
    all_findings,
    demo_rows,
    emit_ddl,
    grade,
    infer_table,
    is_lossy,
    naive_infer,
)

st.set_page_config(page_title="Type Inferencer", layout="wide")

SEVERITY_ICON = {"block": "🛑", "warn": "⚠️", "info": "ℹ️"}

st.title("Type Inferencer")
st.caption(
    "Everything imported as text. This infers the SQL types - but only casts where the "
    "cast is reversible, and says out loud where it refused."
)

with st.sidebar:
    st.header("Policy")
    st.caption("Every threshold the inference leans on. Nothing is hidden in the code.")
    policy = Policy(
        min_rows_for_not_null=st.slider("Min rows before NOT NULL", 1, 200, 25),
        min_rows_for_narrowing=st.slider("Min rows before narrowing", 1, 200, 25),
        varchar_block=st.select_slider("VARCHAR rounding block", [1, 4, 8, 16, 32], value=8),
        varchar_max=st.slider("VARCHAR -> TEXT above", 32, 1024, 255, step=32),
        enum_max_distinct=st.slider("Enum: max distinct values", 2, 40, 12),
        enum_max_ratio=st.slider("Enum: max distinct / rows", 0.01, 1.0, 0.15),
    )
    dialect = st.radio("Dialect", ["postgres", "duckdb", "sqlite"], horizontal=False)
    table_name = st.text_input("Table name", "orders")

source = st.radio(
    "Input", ["Demo corpus (240 order rows)", "Upload CSV", "Paste CSV"], horizontal=True
)

rows: List[Dict[str, str]] = []
if source.startswith("Demo"):
    rows = demo_rows()
elif source == "Upload CSV":
    up = st.file_uploader("CSV file", type=["csv", "tsv", "txt"])
    if up is not None:
        text = up.getvalue().decode("utf-8-sig", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
else:
    pasted = st.text_area("Paste CSV (with a header row)", height=180)
    if pasted.strip():
        rows = list(csv.DictReader(io.StringIO(pasted)))

if not rows:
    st.info("Pick the demo corpus or supply a CSV to begin.")
    st.stop()

# Everything below treats every cell as text - which is exactly the situation being fixed.
rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in rows]
results = infer_table(rows, policy)
findings = all_findings(results)

st.subheader(f"{len(results)} columns · {len(rows)} rows")

table = pd.DataFrame([
    {
        "column": r.name,
        "inferred type": r.type.sql(dialect),
        "null": "NULL" if r.type.nullable else "NOT NULL",
        "nulls": f"{r.profile.n_null}/{r.profile.n_rows}",
        "distinct": r.profile.distinct,
        "len": f"{r.profile.min_len}-{r.profile.max_len}",
        "abstained": "yes" if r.abstained else "",
    }
    for r in results
])
st.dataframe(table, use_container_width=True, hide_index=True)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("DDL")
    ddl = emit_ddl(table_name, results, dialect)
    st.code(ddl, language="sql")
    st.download_button("Download .sql", ddl, file_name=f"{table_name}.sql", mime="text/plain")

with right:
    st.subheader("Findings")
    blocks = sum(1 for f in findings if f.severity == "block")
    warns = sum(1 for f in findings if f.severity == "warn")
    if blocks:
        st.error(f"{blocks} blocking · {warns} warnings - do not ship this DDL unreviewed.")
    elif warns:
        st.warning(f"{warns} warnings - read them before shipping.")
    else:
        st.success("No warnings. Every column had enough evidence for its type.")
    for f in findings:
        with st.expander(f"{SEVERITY_ICON[f.severity]} `{f.column}` - {f.code}", expanded=f.severity != "info"):
            st.write(f.message)

st.divider()
st.subheader("Against the two things people actually ship")
st.caption(
    "`lossy` is measured, not asserted: the proposed type is replayed against every row in "
    "the full column and fails if any value would be corrupted or rejected."
)

cols = list(rows[0].keys())
strategies = {
    "all-TEXT": {c: ColumnType("TEXT") for c in cols},
    "naive cast (200-row sample)": {c: naive_infer(c, [r[c] for r in rows]) for c in cols},
    "lossless (this tool)": {r.name: r.type for r in results},
}

if source.startswith("Demo"):
    bench = pd.DataFrame([
        {"strategy": name, **{
            b: sum(1 for v in grade(rows, proposed, policy).values() if v == b)
            for b in BUCKETS
        }}
        for name, proposed in strategies.items()
    ])
    st.dataframe(bench, use_container_width=True, hide_index=True)
    st.caption(
        "**exact** = matches the hand-written answer · **lossy** = would corrupt at least one "
        "real value · **unsafe** = round-trips but is semantically unproven (an ambiguous date "
        "direction) · **untyped** = text where a safe narrower type existed · **wide** = safe and "
        "right in kind, just wider than necessary - not a defect."
    )
else:
    # No hand-written answer key for arbitrary uploads - report the one thing still measurable.
    damage = pd.DataFrame([
        {
            "strategy": name,
            "columns that would corrupt data": sum(
                1 for c, t in proposed.items() if is_lossy(t, [r[c] for r in rows], policy)
            ),
            "columns left as text": sum(1 for t in proposed.values() if t.kind in ("TEXT", "VARCHAR")),
        }
        for name, proposed in strategies.items()
    ])
    st.dataframe(damage, use_container_width=True, hide_index=True)
    st.caption(
        "No answer key exists for an uploaded file, so only the measurable half is shown: "
        "how many columns each strategy would silently corrupt."
    )
