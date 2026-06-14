"""dbt Model Generator — auto-generate dbt staging models, source YAML, and docs from SQL DDL."""
from __future__ import annotations

import re
import streamlit as st
from typing import Optional, List, Dict, Tuple


def _split_columns(raw: str) -> List[str]:
    """Split column definitions by commas, ignoring commas inside parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in raw:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _extract_create_body(sql: str, start: int) -> Optional[str]:
    """Extract the parenthesized body of a CREATE TABLE, handling nested parens."""
    depth = 0
    for i in range(start, len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
            if depth == 0:
                return sql[start + 1 : i]
    return None


def parse_create_table(sql: str) -> List[Dict]:
    """Parse SQL CREATE TABLE statements into structured table metadata."""
    tables = []
    header_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?:(?P<schema>\w+)\.)?(?P<table>\w+)\s*\(",
        re.IGNORECASE,
    )

    for match in header_pattern.finditer(sql):
        schema = match.group("schema") or "public"
        table_name = match.group("table")
        paren_start = match.end() - 1
        columns_raw = _extract_create_body(sql, paren_start)
        if columns_raw is None:
            continue

        columns = []
        for line in _split_columns(columns_raw):
            line = line.strip()
            if not line:
                continue
            # Skip constraints
            if re.match(r"^\s*(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT|INDEX)", line, re.IGNORECASE):
                continue

            parts = line.split()
            if len(parts) >= 2:
                col_name = parts[0].strip('"').strip("`").strip("[]")
                col_type = parts[1].upper()
                is_nullable = "NOT NULL" not in line.upper()
                is_pk = "PRIMARY KEY" in line.upper()
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "nullable": is_nullable,
                    "is_pk": is_pk,
                })

        tables.append({
            "schema": schema,
            "table": table_name,
            "columns": columns,
        })

    return tables


def map_sql_type_to_dbt(sql_type: str) -> str:
    """Map SQL types to dbt-friendly cast types."""
    sql_type = sql_type.upper()
    if sql_type in ("INT", "INTEGER", "BIGINT", "SMALLINT", "TINYINT"):
        return "integer"
    if sql_type in ("FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "NUMBER"):
        return "float"
    if sql_type in ("BOOL", "BOOLEAN"):
        return "boolean"
    if sql_type in ("DATE",):
        return "date"
    if sql_type in ("TIMESTAMP", "TIMESTAMPTZ", "DATETIME", "TIMESTAMP_NTZ"):
        return "timestamp"
    return "string"


def generate_source_yaml(tables: List[Dict], source_name: str) -> str:
    """Generate dbt source YAML config."""
    lines = [
        "version: 2",
        "",
        "sources:",
        f"  - name: {source_name}",
        f'    description: "Auto-generated source for {source_name}"',
        "    tables:",
    ]

    for t in tables:
        lines.append(f"      - name: {t['table']}")
        lines.append(f'        description: "Source table {t["schema"]}.{t["table"]}"')
        lines.append("        columns:")
        for col in t["columns"]:
            lines.append(f"          - name: {col['name']}")
            lines.append(f'            description: ""')
            if col["is_pk"]:
                lines.append("            tests:")
                lines.append("              - unique")
                lines.append("              - not_null")
            elif not col["nullable"]:
                lines.append("            tests:")
                lines.append("              - not_null")

    return "\n".join(lines)


def generate_staging_model(table: Dict, source_name: str) -> str:
    """Generate a dbt staging model SQL file."""
    table_name = table["table"]
    model_name = f"stg_{source_name}__{table_name}"

    lines = [
        f"-- {model_name}.sql",
        f"-- Staging model for {source_name}.{table_name}",
        "",
        "with source as (",
        "",
        f"    select * from {{{{ source('{source_name}', '{table_name}') }}}}",
        "",
        "),",
        "",
        "renamed as (",
        "",
        "    select",
    ]

    col_lines = []
    for col in table["columns"]:
        col_name = col["name"]
        clean_name = re.sub(r"([A-Z])", r"_\1", col_name).lower().strip("_")
        dbt_type = map_sql_type_to_dbt(col["type"])

        if clean_name != col_name.lower():
            col_lines.append(f"        {col_name} as {clean_name}")
        else:
            col_lines.append(f"        {col_name}")

    lines.append(",\n".join(col_lines))

    lines.extend([
        "",
        "    from source",
        "",
        ")",
        "",
        "select * from renamed",
    ])

    return "\n".join(lines)


def generate_model_yaml(table: Dict, source_name: str) -> str:
    """Generate dbt model YAML (schema test file)."""
    table_name = table["table"]
    model_name = f"stg_{source_name}__{table_name}"

    lines = [
        "version: 2",
        "",
        "models:",
        f"  - name: {model_name}",
        f'    description: "Staging model for {table_name}"',
        "    columns:",
    ]

    for col in table["columns"]:
        clean_name = re.sub(r"([A-Z])", r"_\1", col["name"]).lower().strip("_")
        lines.append(f"      - name: {clean_name}")
        lines.append(f'        description: ""')

        tests = []
        if col["is_pk"]:
            tests.extend(["unique", "not_null"])
        elif not col["nullable"]:
            tests.append("not_null")

        if tests:
            lines.append("        tests:")
            for test in tests:
                lines.append(f"          - {test}")

    return "\n".join(lines)


def generate_all(sql: str, source_name: str) -> Dict[str, str]:
    """Parse SQL and generate all dbt artifacts."""
    tables = parse_create_table(sql)
    if not tables:
        return {}

    files = {}
    files[f"models/staging/{source_name}/__{source_name}__sources.yml"] = generate_source_yaml(tables, source_name)

    for table in tables:
        table_name = table["table"]
        model_name = f"stg_{source_name}__{table_name}"
        files[f"models/staging/{source_name}/{model_name}.sql"] = generate_staging_model(table, source_name)
        files[f"models/staging/{source_name}/_{source_name}__models.yml"] = generate_model_yaml(table, source_name)

    return files


SAMPLE_SQL = """
CREATE TABLE raw.customers (
    customer_id INTEGER PRIMARY KEY,
    firstName VARCHAR(100) NOT NULL,
    lastName VARCHAR(100) NOT NULL,
    email VARCHAR(255),
    signup_date TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true,
    lifetime_value DECIMAL(10,2)
);

CREATE TABLE raw.orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TIMESTAMP NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(50) NOT NULL,
    shipping_address TEXT,
    FOREIGN KEY (customer_id) REFERENCES raw.customers(customer_id)
);

CREATE TABLE raw.order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    discount_pct FLOAT DEFAULT 0,
    FOREIGN KEY (order_id) REFERENCES raw.orders(order_id)
);
"""


def main():
    st.set_page_config(page_title="dbt Model Generator", page_icon="🔧", layout="wide")
    st.title("dbt Model Generator")
    st.markdown("> Paste SQL `CREATE TABLE` statements, get production-ready dbt staging models instantly.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        source_name = st.text_input("Source name", value="raw_db", help="Used in source() and model naming")
        sql_input = st.text_area(
            "Paste CREATE TABLE SQL",
            value=SAMPLE_SQL,
            height=400,
        )

        generate_btn = st.button("Generate dbt Models", type="primary", use_container_width=True)

    with col2:
        st.subheader("Generated Output")

        if generate_btn and sql_input.strip():
            files = generate_all(sql_input, source_name)

            if not files:
                st.error("No CREATE TABLE statements found. Check your SQL syntax.")
            else:
                st.success(f"Generated {len(files)} files")

                for filepath, content in files.items():
                    lang = "yaml" if filepath.endswith(".yml") else "sql"
                    with st.expander(f"📄 {filepath}", expanded=True):
                        st.code(content, language=lang)

                st.divider()
                st.subheader("Copy-paste ready")
                all_content = []
                for filepath, content in files.items():
                    all_content.append(f"-- FILE: {filepath}\n{content}")
                st.code("\n\n".join(all_content), language="sql")


if __name__ == "__main__":
    main()
