from __future__ import annotations

import re
from typing import Dict, List

import streamlit as st


def parse_sql(sql: str) -> Dict[str, Dict]:
    """Parse CREATE TABLE statements into a structured schema dict."""
    tables: Dict[str, Dict] = {}
    sql_clean = re.sub(r'--[^\n]*', '', sql)
    sql_clean = re.sub(r'/\*.*?\*/', '', sql_clean, flags=re.DOTALL)

    pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\']?(\w+)[`"\']?\s*\((.*?)\)\s*;',
        re.IGNORECASE | re.DOTALL,
    )

    for match in pattern.finditer(sql_clean):
        table_name = match.group(1).lower()
        body = match.group(2)
        columns: List[Dict] = []
        foreign_keys: List[Dict] = []

        for line in body.split(','):
            line = line.strip()
            if not line:
                continue

            fk_match = re.match(
                r'FOREIGN\s+KEY\s*\(\s*[`"\']?(\w+)[`"\']?\s*\)\s*REFERENCES\s+[`"\']?(\w+)[`"\']?\s*\(\s*[`"\']?(\w+)[`"\']?\s*\)',
                line, re.IGNORECASE,
            )
            if fk_match:
                foreign_keys.append({
                    'column': fk_match.group(1).lower(),
                    'ref_table': fk_match.group(2).lower(),
                    'ref_column': fk_match.group(3).lower(),
                })
                continue

            if re.match(r'(PRIMARY\s+KEY|UNIQUE|INDEX|CHECK|CONSTRAINT)', line, re.IGNORECASE):
                pk_match = re.match(r'PRIMARY\s+KEY\s*\(\s*([^)]+)\)', line, re.IGNORECASE)
                if pk_match:
                    pk_cols = [c.strip().strip('`"\'').lower() for c in pk_match.group(1).split(',')]
                    for col in columns:
                        if col['name'] in pk_cols:
                            col['primary_key'] = True
                continue

            col_match = re.match(
                r'[`"\']?(\w+)[`"\']?\s+(\w+(?:\s*\([^)]*\))?)',
                line, re.IGNORECASE,
            )
            if col_match:
                col_name = col_match.group(1).lower()
                col_type = col_match.group(2).upper()
                is_pk = bool(re.search(r'PRIMARY\s+KEY', line, re.IGNORECASE))
                is_nn = bool(re.search(r'NOT\s+NULL', line, re.IGNORECASE))

                inline_ref = re.search(
                    r'REFERENCES\s+[`"\']?(\w+)[`"\']?\s*\(\s*[`"\']?(\w+)[`"\']?\s*\)',
                    line, re.IGNORECASE,
                )
                if inline_ref:
                    foreign_keys.append({
                        'column': col_name,
                        'ref_table': inline_ref.group(1).lower(),
                        'ref_column': inline_ref.group(2).lower(),
                    })

                columns.append({
                    'name': col_name,
                    'type': col_type,
                    'primary_key': is_pk,
                    'not_null': is_nn,
                })

        tables[table_name] = {'columns': columns, 'foreign_keys': foreign_keys}

    return tables


def generate_mermaid(tables: Dict[str, Dict]) -> str:
    """Generate Mermaid erDiagram syntax from parsed schema."""
    lines = ['erDiagram']

    for table_name, info in tables.items():
        lines.append(f'    {table_name} {{')
        for col in info['columns']:
            pk_marker = ' PK' if col['primary_key'] else ''
            fk_marker = ''
            for fk in info['foreign_keys']:
                if fk['column'] == col['name']:
                    fk_marker = ' FK'
                    break
            col_type = col['type'].replace(' ', '_')
            lines.append(f'        {col_type} {col["name"]}{pk_marker}{fk_marker}')
        lines.append('    }')

    for table_name, info in tables.items():
        for fk in info['foreign_keys']:
            ref = fk['ref_table']
            lines.append(f'    {ref} ||--o{{ {table_name} : "{fk["column"]}"')

    return '\n'.join(lines)


def generate_dot(tables: Dict[str, Dict]) -> str:
    """Generate Graphviz DOT syntax for rendering with matplotlib."""
    lines = [
        'digraph ERD {',
        '    rankdir=LR;',
        '    node [shape=plaintext];',
    ]

    for table_name, info in tables.items():
        label_rows = [f'<TR><TD COLSPAN="2" BGCOLOR="#4A90D9"><FONT COLOR="white"><B>{table_name}</B></FONT></TD></TR>']
        for col in info['columns']:
            pk = "PK " if col['primary_key'] else ""
            fk = "FK " if any(f['column'] == col['name'] for f in info['foreign_keys']) else ""
            prefix = pk or fk
            label_rows.append(
                f'<TR><TD ALIGN="LEFT"><FONT FACE="monospace">{prefix}{col["name"]}</FONT></TD>'
                f'<TD ALIGN="LEFT"><FONT COLOR="#666666">{col["type"]}</FONT></TD></TR>'
            )
        label = ''.join(label_rows)
        lines.append(
            f'    {table_name} [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0">{label}</TABLE>>];'
        )

    for table_name, info in tables.items():
        for fk in info['foreign_keys']:
            lines.append(f'    {fk["ref_table"]} -> {table_name} [label="{fk["column"]}"];')

    lines.append('}')
    return '\n'.join(lines)


SAMPLE_SQL = """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(200)
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    department_id INTEGER NOT NULL,
    manager_id INTEGER,
    hire_date DATE NOT NULL,
    salary DECIMAL(10,2),
    FOREIGN KEY (department_id) REFERENCES departments(id),
    FOREIGN KEY (manager_id) REFERENCES employees(id)
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12,2),
    department_id INTEGER,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE assignments (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    role VARCHAR(50),
    hours_allocated INTEGER,
    FOREIGN KEY (employee_id) REFERENCES employees(id),
    FOREIGN KEY (project_id) REFERENCES projects(id)
);
"""


def main() -> None:
    st.set_page_config(page_title="ERD Generator", page_icon="🗂️", layout="wide")
    st.title("ERD Generator from SQL")
    st.caption("Paste your CREATE TABLE statements and get an instant ER diagram")

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("SQL Input")
        sql_input = st.text_area(
            "Paste CREATE TABLE statements:",
            value=SAMPLE_SQL,
            height=400,
        )

        if st.button("Generate ERD", type="primary", use_container_width=True):
            st.session_state['generate'] = True

    with col_output:
        if sql_input and st.session_state.get('generate', False):
            tables = parse_sql(sql_input)

            if not tables:
                st.error("No valid CREATE TABLE statements found. Check your SQL syntax.")
                return

            st.subheader(f"Schema: {len(tables)} tables found")

            for tname, info in tables.items():
                with st.expander(f"**{tname}** ({len(info['columns'])} columns)", expanded=False):
                    for col in info['columns']:
                        flags = []
                        if col['primary_key']:
                            flags.append("PK")
                        if col['not_null']:
                            flags.append("NOT NULL")
                        if any(f['column'] == col['name'] for f in info['foreign_keys']):
                            flags.append("FK")
                        flag_str = f" `{'|'.join(flags)}`" if flags else ""
                        st.markdown(f"- **{col['name']}** `{col['type']}`{flag_str}")

            st.subheader("Mermaid ER Diagram")
            mermaid_code = generate_mermaid(tables)
            st.code(mermaid_code, language="text")

            st.caption("Copy the code above into [mermaid.live](https://mermaid.live) to render, or use GitHub markdown.")

            st.subheader("Graphviz DOT")
            dot_code = generate_dot(tables)
            st.code(dot_code, language="dot")

            st.download_button(
                "Download Mermaid (.md)",
                f"```mermaid\n{mermaid_code}\n```",
                file_name="erd.md",
                mime="text/markdown",
            )


if __name__ == '__main__':
    main()
