from __future__ import annotations

# Streamlit UI for the JSON Flattener. Paste nested JSON, pick how arrays should
# be handled, and get flat rows, ragged-path and type-conflict reports, and the
# CREATE TABLE for the flattened shape. Fully offline.
import json

import pandas as pd
import streamlit as st
from flatten import (
    SAMPLE_ORDERS,
    flatten,
    infer_schema,
    to_dataframe,
    to_ddl,
)

st.set_page_config(page_title="JSON Flattener", page_icon="🧱", layout="wide")

st.title("🧱 JSON Flattener")
st.caption(
    "Every API response is nested. Every warehouse table is flat. This closes the gap "
    "with dot-path columns - and takes an explicit position on the two decisions that "
    "actually break pipelines: what to do with arrays, and what to do with missing keys."
)

with st.sidebar:
    st.header("Array handling")
    array_mode = st.radio(
        "Mode",
        ["explode", "index", "json"],
        index=0,
        help="explode = one row per element (fact shape) · index = one column per "
             "element · json = keep the array whole as a string",
    )
    pin_grain = False
    grain = ""
    if array_mode == "explode":
        pin_grain = st.checkbox("Pin the grain (explode only one path)", value=True)
        grain = st.text_input("Path to explode", "items", disabled=not pin_grain)
    max_array_cols = st.number_input("Max array columns (index mode)", 1, 100, 20)
    st.header("Table")
    table_name = st.text_input("Table name for DDL", "orders_flat")

st.subheader("Input")
use_sample = st.checkbox("Use the bundled ragged sample (3 orders)", value=True)
if use_sample:
    raw = json.dumps(SAMPLE_ORDERS, indent=2)
    st.caption(
        "Deliberately awkward: one record has no `address.zip` and no `coupon` key, "
        "another adds a `channel` field nobody else has, and `total` arrives as a "
        "string in the third - a type conflict that would reject the batch at load time."
    )
else:
    raw = "[]"
text = st.text_area("Nested JSON (object or array of objects)", raw, height=220)

try:
    parsed = json.loads(text) if text.strip() else []
except json.JSONDecodeError as e:
    st.error(f"Invalid JSON: {e}")
    st.stop()

records = parsed if isinstance(parsed, list) else [parsed]
records = [r for r in records if isinstance(r, dict)]
if not records:
    st.info("Paste an object, or an array of objects, to flatten.")
    st.stop()

explode_paths = [g.strip() for g in grain.split(",") if g.strip()] if pin_grain else None
rows, stats = flatten(
    records,
    array_mode=array_mode,
    max_array_cols=int(max_array_cols),
    explode_paths=explode_paths,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Records in", stats.input_records)
c2.metric("Rows out", stats.output_rows, f"{stats.row_multiplier}x")
c3.metric("Columns", stats.columns)
c4.metric("Max depth", stats.max_depth)
c5.metric("Type conflicts", len(stats.type_conflicts))

if stats.fanout_warning:
    st.error(f"**Fan-out:** {stats.fanout_warning}")
elif array_mode == "explode" and stats.exploded_paths:
    st.success(f"Grain pinned to: {', '.join(stats.exploded_paths)} - measures are safe to SUM.")

tab_rows, tab_schema, tab_ragged, tab_ddl = st.tabs(
    ["📋 Flat rows", "🧬 Schema", "🕳️ Ragged & conflicts", "🗄️ DDL"]
)

df = to_dataframe(rows)


def display_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type columns to string for the grid only.

    Arrow can't serialise a column holding both 140.0 and "0.00" - and a
    mixed-type column is exactly what this tool exists to surface, so it will
    happen on real input. Stringify for display; the inferred schema and DDL
    still report the true underlying types.
    """
    out = frame.copy()
    for col in out.columns:
        if col in stats.type_conflicts:
            out[col] = out[col].map(lambda v: "" if v is None else str(v))
    return out


with tab_rows:
    st.dataframe(display_safe(df), width="stretch", hide_index=True)
    if stats.row_multiplier > 1:
        st.info(
            f"{stats.input_records} records became {stats.output_rows} rows "
            f"({stats.row_multiplier}x). Every scalar from the parent repeats on each "
            "row - correct for a fact table, wrong if you then SUM the parent's columns."
        )

with tab_schema:
    sch = pd.DataFrame(infer_schema(rows))
    st.dataframe(sch, width="stretch", hide_index=True)
    st.caption(
        "`fill_rate` is the share of rows where the path was present. A low fill rate "
        "on a column you thought was mandatory is the interesting signal here."
    )
    low = sch[sch["fill_rate"] < 0.6]["column"].tolist() if len(sch) else []
    if low:
        st.warning("Under 60% populated: " + ", ".join(low[:8]))

with tab_ragged:
    st.write("**Ragged paths** - present in some records, absent in others (filled with null).")
    if stats.ragged_paths:
        st.dataframe(
            pd.DataFrame(
                sorted(stats.ragged_paths.items(), key=lambda kv: -kv[1]),
                columns=["path", "rows missing it"],
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.success("Every record had every path - rectangular input.")

    st.write("**Type conflicts** - the same path with different scalar types across records.")
    if stats.type_conflicts:
        st.dataframe(
            pd.DataFrame(
                [{"path": k, "types": " + ".join(v)} for k, v in stats.type_conflicts.items()]
            ),
            width="stretch", hide_index=True,
        )
        st.error(
            "This is the failure that shows up at *load* time, not flatten time: a column "
            "that is numeric in 9,000 records and text in 3 rejects the whole batch. "
            "The inferred DDL widens these to VARCHAR rather than dropping rows."
        )
    else:
        st.success("No type conflicts.")

with tab_ddl:
    st.code(to_ddl(rows, table_name or "flattened"), language="sql")
    st.caption(
        "Dot paths are quoted so they survive as column names. Mixed-type columns are "
        "widened to VARCHAR and annotated, so the load succeeds and the conflict stays visible."
    )

st.divider()
st.caption(
    "Day 128 of Phoebe's daily FDE build - Data Engineering Pro line. "
    "Pairs with Day 98 (schema registry), Day 63 (type inference), Day 2 (schema diff)."
)
