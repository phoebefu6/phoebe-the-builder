from __future__ import annotations

# Streamlit UI for the Lightweight Data Catalog. Upload one or more CSVs (or use
# the built-in sample tables), auto-profile each into a catalog entry, attach
# human metadata (owner / description / tags), browse per-column profiles, and
# search across table + column names, descriptions, and tags. Fully offline.

import pandas as pd
import streamlit as st

from catalog import Catalog, make_sample_tables, profile_dataframe

st.set_page_config(page_title="Lightweight Data Catalog", page_icon="📖", layout="wide")

st.title("📖 Lightweight Data Catalog")
st.caption(
    "Nobody remembers what tables and columns exist, what they mean, or who owns "
    "them - so every analysis starts by pinging three people on Slack. Point this "
    "at your data: it auto-profiles each table (dtype, null %, distinct count, "
    "sample values, an inferred semantic type), you add owner/description/tags, "
    "and everyone can search it. Offline, no API keys."
)


def _build_catalog() -> Catalog:
    """Read whichever source is active into a fresh Catalog. Uploaded CSVs get
    'unassigned' owners so a steward knows metadata is still owed."""
    cat = Catalog()
    ups = st.session_state.get("uploads")
    if ups:
        for up in ups:
            df = pd.read_csv(up)
            name = up.name.rsplit(".", 1)[0]
            cat.add(profile_dataframe(df, name))
        return cat
    # Sample fallback carries realistic human metadata so search has something
    # to hit beyond column names.
    tables = make_sample_tables()
    cat.add(profile_dataframe(
        tables["customers"], "customers", owner="growth-team",
        description="One row per registered customer with contact + value.",
        tags=["crm", "pii", "core"],
    ))
    cat.add(profile_dataframe(
        tables["orders"], "orders", owner="commerce-team",
        description="One row per order placed, linked to customers by customer_id.",
        tags=["transactions", "core"],
    ))
    return cat


with st.sidebar:
    st.header("Data")
    ups = st.file_uploader("Upload CSV(s)", type=["csv"], accept_multiple_files=True)
    st.session_state["uploads"] = ups
    if not ups:
        st.info("No upload yet - showing sample **customers** + **orders** tables.")
    st.divider()
    st.markdown(
        "**Semantic types (inferred)**\n\n"
        "`id` `email` `date` `category` `numeric` `text` `boolean`\n\n"
        "These are heuristic guesses from the data - confirm before trusting."
    )

cat = _build_catalog()

query = st.text_input("🔍 Search tables, columns, descriptions, tags", "")
entries = cat.search(query) if query else list(cat.entries.values())

c1, c2 = st.columns(2)
c1.metric("Tables catalogued", len(cat.entries))
c2.metric("Matching search", len(entries))

if not entries:
    st.warning("No tables match that search.")
else:
    for e in entries:
        with st.expander(
            f"📋 {e.name}  -  {e.n_rows:,} rows x {e.n_cols} cols  "
            f"(owner: {e.owner})",
            expanded=bool(query),
        ):
            tags = ", ".join(e.tags) if e.tags else "-"
            st.markdown(f"**Description:** {e.description or '-'}")
            st.markdown(f"**Tags:** {tags}")
            prof = pd.DataFrame([{
                "column": c.name,
                "dtype": c.dtype,
                "semantic_type": c.semantic_type,
                "null_%": c.null_pct,
                "distinct": c.distinct,
                "samples": ", ".join(c.samples) if c.samples else "-",
            } for c in e.columns])
            st.dataframe(prof, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Export")
md = cat.to_markdown()
st.download_button(
    "⬇️ Download catalog as Markdown",
    md.encode(),
    file_name="data_catalog.md",
    mime="text/markdown",
)
with st.expander("Preview Markdown export"):
    st.code(md, language="markdown")

st.caption(
    "Semantic types are inferred heuristics - a human should confirm them. This "
    "catalog is for discovery, not access control."
)
