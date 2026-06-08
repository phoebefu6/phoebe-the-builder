from __future__ import annotations

import re
from typing import Optional
import streamlit as st
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Schema parser
# ---------------------------------------------------------------------------

@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True
    default: Optional[str] = None
    constraints: list[str] = field(default_factory=list)


@dataclass
class Table:
    name: str
    columns: dict[str, Column] = field(default_factory=dict)
    primary_key: list[str] = field(default_factory=list)


def parse_sql_schema(sql: str) -> dict[str, Table]:
    """Parse CREATE TABLE statements from raw SQL into structured Table objects."""
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

    tables: dict[str, Table] = {}
    table_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )

    for match in table_pattern.finditer(sql):
        table_name = match.group(1).lower()
        body = match.group(2)
        table = Table(name=table_name)

        for line in body.split(","):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("PRIMARY KEY"):
                pk_cols = re.findall(r"[`\"\[]?(\w+)[`\"\]]?", line.split("(", 1)[1].split(")")[0])
                table.primary_key = [c.lower() for c in pk_cols]
                continue
            if any(upper.startswith(kw) for kw in ("UNIQUE", "CHECK", "FOREIGN KEY", "CONSTRAINT", "INDEX")):
                continue

            col_match = re.match(
                r"[`\"\[]?(\w+)[`\"\]]?\s+([\w\s()]+?)(?:\s+(NOT\s+NULL|NULL))?(?:\s+DEFAULT\s+(.+?))?(?:\s+(PRIMARY\s+KEY|UNIQUE))?$",
                line.strip().rstrip(","),
                re.IGNORECASE,
            )
            if col_match:
                col_name = col_match.group(1).lower()
                col_type = re.sub(r"\s+", " ", col_match.group(2).strip()).upper()
                nullable = True
                if col_match.group(3) and "NOT" in col_match.group(3).upper():
                    nullable = False
                default = col_match.group(4).strip() if col_match.group(4) else None
                constraints = []
                if col_match.group(5):
                    constraints.append(col_match.group(5).upper().replace("  ", " "))
                    if "PRIMARY" in constraints[0]:
                        table.primary_key.append(col_name)

                table.columns[col_name] = Column(
                    name=col_name,
                    data_type=col_type,
                    nullable=nullable,
                    default=default,
                    constraints=constraints,
                )

        tables[table_name] = table

    return tables


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

@dataclass
class Change:
    severity: str  # "breaking", "warning", "info"
    category: str
    table: str
    detail: str


def diff_schemas(old: dict[str, Table], new: dict[str, Table]) -> list[Change]:
    changes: list[Change] = []

    old_names = set(old.keys())
    new_names = set(new.keys())

    for t in sorted(old_names - new_names):
        changes.append(Change("breaking", "table_dropped", t, f"Table `{t}` was dropped"))

    for t in sorted(new_names - old_names):
        changes.append(Change("info", "table_added", t, f"Table `{t}` was added"))

    for t in sorted(old_names & new_names):
        old_cols = old[t].columns
        new_cols = new[t].columns

        for c in sorted(set(old_cols) - set(new_cols)):
            changes.append(Change("breaking", "column_dropped", t, f"Column `{t}.{c}` was dropped"))

        for c in sorted(set(new_cols) - set(old_cols)):
            col = new_cols[c]
            sev = "warning" if not col.nullable and col.default is None else "info"
            changes.append(Change(sev, "column_added", t, f"Column `{t}.{c}` added ({col.data_type}, {'NOT NULL' if not col.nullable else 'nullable'})"))

        for c in sorted(set(old_cols) & set(new_cols)):
            oc, nc = old_cols[c], new_cols[c]
            if oc.data_type != nc.data_type:
                changes.append(Change("warning", "type_changed", t, f"`{t}.{c}` type changed: {oc.data_type} -> {nc.data_type}"))
            if oc.nullable and not nc.nullable:
                changes.append(Change("warning", "nullable_changed", t, f"`{t}.{c}` changed from nullable to NOT NULL"))
            elif not oc.nullable and nc.nullable:
                changes.append(Change("info", "nullable_changed", t, f"`{t}.{c}` changed from NOT NULL to nullable"))
            if oc.default != nc.default:
                changes.append(Change("info", "default_changed", t, f"`{t}.{c}` default changed: {oc.default} -> {nc.default}"))

        if old[t].primary_key != new[t].primary_key:
            changes.append(Change("breaking", "pk_changed", t, f"`{t}` primary key changed: {old[t].primary_key} -> {new[t].primary_key}"))

    return changes


# ---------------------------------------------------------------------------
# Sample schemas
# ---------------------------------------------------------------------------

SAMPLE_OLD = """CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    sku VARCHAR(50)
);"""

SAMPLE_NEW = """CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(500) NOT NULL,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    role VARCHAR(20) DEFAULT 'user'
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    total DECIMAL(12,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE payments (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    method VARCHAR(50)
);"""


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {"breaking": "❗", "warning": "⚠️", "info": "ℹ️"}
SEVERITY_COLOR = {"breaking": "red", "warning": "orange", "info": "blue"}


def main():
    st.set_page_config(page_title="Schema Diff Tool", page_icon="\U0001f504", layout="wide")
    st.title("\U0001f504 Schema Diff Tool")
    st.caption("Paste two SQL schemas and instantly see what changed — and what might break.")

    use_sample = st.checkbox("Load sample schemas", value=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Old Schema (before)")
        old_sql = st.text_area("old_sql", value=SAMPLE_OLD if use_sample else "", height=300, label_visibility="collapsed")
    with col2:
        st.subheader("New Schema (after)")
        new_sql = st.text_area("new_sql", value=SAMPLE_NEW if use_sample else "", height=300, label_visibility="collapsed")

    if st.button("\U0001f50d Compare Schemas", type="primary", use_container_width=True):
        if not old_sql.strip() or not new_sql.strip():
            st.error("Please paste SQL in both panels.")
            return

        old_tables = parse_sql_schema(old_sql)
        new_tables = parse_sql_schema(new_sql)

        if not old_tables and not new_tables:
            st.warning("No CREATE TABLE statements found. Check your SQL syntax.")
            return

        changes = diff_schemas(old_tables, new_tables)

        st.divider()

        # Summary metrics
        breaking = sum(1 for c in changes if c.severity == "breaking")
        warnings = sum(1 for c in changes if c.severity == "warning")
        info = sum(1 for c in changes if c.severity == "info")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Changes", len(changes))
        m2.metric("❗ Breaking", breaking)
        m3.metric("⚠️ Warnings", warnings)
        m4.metric("ℹ️ Info", info)

        if not changes:
            st.success("Schemas are identical — no changes detected.")
            return

        # Migration risk
        if breaking > 0:
            st.error(f"**HIGH RISK** — {breaking} breaking change(s) detected. Review carefully before migrating.")
        elif warnings > 0:
            st.warning(f"**MEDIUM RISK** — {warnings} warning(s). Test with production-like data before deploying.")
        else:
            st.success("**LOW RISK** — Only additive/informational changes.")

        # Change list
        st.subheader("Changes")
        for c in sorted(changes, key=lambda x: ("breaking", "warning", "info").index(x.severity)):
            emoji = SEVERITY_EMOJI[c.severity]
            color = SEVERITY_COLOR[c.severity]
            st.markdown(f":{color}[{emoji} **{c.severity.upper()}**] — {c.detail}")

        # Table summary
        st.subheader("Table Overview")
        all_tables = sorted(set(list(old_tables.keys()) + list(new_tables.keys())))
        for t in all_tables:
            in_old = t in old_tables
            in_new = t in new_tables
            if in_old and in_new:
                table_changes = [c for c in changes if c.table == t]
                if table_changes:
                    st.markdown(f"**`{t}`** — {len(table_changes)} change(s)")
                else:
                    st.markdown(f"**`{t}`** — no changes")
            elif in_new:
                st.markdown(f"**`{t}`** — :green[NEW]")
            else:
                st.markdown(f"**`{t}`** — :red[DROPPED]")


if __name__ == "__main__":
    main()
