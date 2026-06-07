"""
CSV to PostgreSQL Loader
Upload a CSV, preview it, configure column types, and load it into any PostgreSQL database.
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from io import StringIO
import re


def infer_sql_type(dtype, sample_values):
    """Map pandas dtype to PostgreSQL type with smart inference."""
    dtype_str = str(dtype)
    if "int" in dtype_str:
        max_val = sample_values.dropna().abs().max() if len(sample_values.dropna()) > 0 else 0
        if max_val < 32767:
            return "SMALLINT"
        elif max_val < 2147483647:
            return "INTEGER"
        return "BIGINT"
    elif "float" in dtype_str:
        return "NUMERIC"
    elif "datetime" in dtype_str:
        return "TIMESTAMP"
    elif "bool" in dtype_str:
        return "BOOLEAN"
    else:
        max_len = sample_values.astype(str).str.len().max() if len(sample_values) > 0 else 255
        if max_len and max_len < 256:
            return f"VARCHAR({int(max_len * 1.5)})"
        return "TEXT"


def sanitize_table_name(name):
    """Clean filename into a valid SQL table name."""
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def generate_create_table_sql(table_name, df, type_overrides):
    """Generate CREATE TABLE statement."""
    columns = []
    for col in df.columns:
        sql_type = type_overrides.get(col, infer_sql_type(df[col].dtype, df[col]))
        safe_col = f'"{col}"'
        columns.append(f"  {safe_col} {sql_type}")
    cols_sql = ",\n".join(columns)
    return f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n{cols_sql}\n);'


def load_to_postgres(connection_string, table_name, df, if_exists):
    """Load DataFrame into PostgreSQL."""
    engine = create_engine(connection_string)
    with engine.connect() as conn:
        # Test connection
        conn.execute(text("SELECT 1"))

    df.to_sql(table_name, engine, if_exists=if_exists, index=False, method="multi", chunksize=1000)
    return len(df)


# --- UI ---
st.set_page_config(page_title="CSV to PostgreSQL Loader", page_icon="🐘", layout="wide")
st.title("🐘 CSV to PostgreSQL Loader")
st.caption("Upload a CSV. Preview it. Load it into Postgres. Done.")

# Sidebar: connection config
with st.sidebar:
    st.header("Database Connection")
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="5432")
    database = st.text_input("Database", value="mydb")
    username = st.text_input("Username", value="postgres")
    password = st.text_input("Password", type="password")
    conn_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"

# File upload
uploaded_file = st.file_uploader("Drop a CSV here", type=["csv", "tsv", "txt"])

if uploaded_file:
    # Detect separator
    sep = st.selectbox("Delimiter", [",", "\t", "|", ";"], index=0)

    try:
        df = pd.read_csv(uploaded_file, sep=sep, nrows=10000)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    # Preview
    st.subheader(f"Preview — {len(df):,} rows, {len(df.columns)} columns")
    st.dataframe(df.head(20), use_container_width=True)

    # Data profile
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Nulls", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    # Table config
    st.subheader("Load Configuration")
    default_name = sanitize_table_name(uploaded_file.name.rsplit(".", 1)[0])
    table_name = st.text_input("Table name", value=default_name)
    if_exists = st.radio("If table exists", ["fail", "replace", "append"], index=1, horizontal=True)

    # Column type overrides
    with st.expander("Column types (auto-detected, click to override)"):
        type_overrides = {}
        pg_types = ["VARCHAR(255)", "TEXT", "INTEGER", "BIGINT", "SMALLINT",
                     "NUMERIC", "FLOAT", "BOOLEAN", "DATE", "TIMESTAMP"]
        for col in df.columns:
            detected = infer_sql_type(df[col].dtype, df[col])
            selected = st.selectbox(f"{col}", pg_types,
                                    index=pg_types.index(detected) if detected in pg_types else 0,
                                    key=f"type_{col}")
            type_overrides[col] = selected

    # Show generated SQL
    create_sql = generate_create_table_sql(table_name, df, type_overrides)
    with st.expander("Generated SQL"):
        st.code(create_sql, language="sql")

    # Load button
    if st.button("Load to PostgreSQL", type="primary", use_container_width=True):
        if not password:
            st.warning("Enter your database password in the sidebar.")
        else:
            with st.spinner(f"Loading {len(df):,} rows into `{table_name}`..."):
                try:
                    rows = load_to_postgres(conn_string, table_name, df, if_exists)
                    st.success(f"Loaded {rows:,} rows into `{table_name}`")
                    st.balloons()
                except Exception as e:
                    st.error(f"Load failed: {e}")
else:
    st.info("Upload a CSV file to get started.")
