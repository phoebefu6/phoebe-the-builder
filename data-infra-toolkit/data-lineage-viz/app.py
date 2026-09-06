from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# ---------------------------------------------------------------------------
# Core logic: parse SQL into table-to-table lineage edges
# ---------------------------------------------------------------------------

# Statements that DEFINE a table fed from a query: target <- sources
_TARGET_PATTERNS = [
    re.compile(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+[`"\']?([\w.]+)[`"\']?\s+AS\b', re.IGNORECASE),
    re.compile(r'CREATE\s+(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\']?([\w.]+)[`"\']?\s+AS\b', re.IGNORECASE),
    re.compile(r'INSERT\s+INTO\s+[`"\']?([\w.]+)[`"\']?', re.IGNORECASE),
]

# Source tables referenced inside the query body
_SOURCE_PATTERN = re.compile(r'\b(?:FROM|JOIN)\s+[`"\']?([\w.]+)[`"\']?', re.IGNORECASE)

# SQL keywords that can follow FROM/JOIN but are not real tables
_NOISE = {"select", "where", "on", "using", "as", "lateral", "unnest"}


def _strip_comments(sql: str) -> str:
    sql = re.sub(r'--[^\n]*', '', sql)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    return sql


def _norm(name: str) -> str:
    """Drop schema prefix and lowercase: analytics.Orders -> orders."""
    return name.split('.')[-1].lower()


def parse_lineage(sql: str) -> List[Tuple[str, str]]:
    """Return list of (source_table, target_table) dependency edges."""
    sql_clean = _strip_comments(sql)
    edges: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    # Split on semicolons so each statement maps to one target
    for stmt in sql_clean.split(';'):
        stmt = stmt.strip()
        if not stmt:
            continue

        target = None
        for pat in _TARGET_PATTERNS:
            m = pat.search(stmt)
            if m:
                target = _norm(m.group(1))
                break
        if not target:
            continue  # plain SELECT / DDL with no lineage target

        for sm in _SOURCE_PATTERN.finditer(stmt):
            source = _norm(sm.group(1))
            if source in _NOISE or source == target:
                continue
            edge = (source, target)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return edges


def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def impact_analysis(g: nx.DiGraph, table: str) -> Dict[str, List[str]]:
    """Upstream = what feeds `table`; downstream = what breaks if `table` changes."""
    table = _norm(table)
    if table not in g:
        return {"upstream": [], "downstream": [], "exists": False}
    return {
        "upstream": sorted(nx.ancestors(g, table)),
        "downstream": sorted(nx.descendants(g, table)),
        "exists": True,
    }


def draw_graph(g: nx.DiGraph, highlight: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No lineage found", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(g, seed=42, k=1.2)
    colors = []
    hl = _norm(highlight) if highlight else None
    for n in g.nodes():
        if n == hl:
            colors.append("#ff6b6b")
        elif hl and n in nx.descendants(g, hl):
            colors.append("#ffd166")  # downstream = impacted
        elif hl and n in nx.ancestors(g, hl):
            colors.append("#06d6a0")  # upstream = sources
        else:
            colors.append("#118ab2")

    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=2200, ax=ax)
    nx.draw_networkx_edges(g, pos, edge_color="#555", arrows=True,
                           arrowsize=20, node_size=2200, ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, font_color="white", ax=ax)
    ax.set_title("Data Lineage Graph", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

SAMPLE_SQL = """\
CREATE TABLE stg_orders AS SELECT * FROM raw_orders;
CREATE TABLE stg_customers AS SELECT * FROM raw_customers;

CREATE VIEW dim_customers AS
SELECT * FROM stg_customers;

CREATE TABLE fct_orders AS
SELECT o.id, o.amount, c.region
FROM stg_orders o
JOIN dim_customers c ON o.customer_id = c.id;

INSERT INTO mart_revenue
SELECT region, SUM(amount) FROM fct_orders GROUP BY region;
"""


def main() -> None:
    st.set_page_config(page_title="Data Lineage Visualizer", page_icon="🔗", layout="wide")
    st.title("🔗 Data Lineage Visualizer")
    st.caption("Paste SQL. See what breaks when you change a table.")

    sql = st.text_area("SQL (CREATE TABLE AS / CREATE VIEW / INSERT INTO)",
                       value=SAMPLE_SQL, height=280)

    edges = parse_lineage(sql)
    g = build_graph(edges)

    if g.number_of_nodes() == 0:
        st.warning("No lineage edges found. Need CREATE ... AS, CREATE VIEW, or INSERT INTO statements.")
        return

    tables = sorted(g.nodes())
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Tables", g.number_of_nodes())
        st.metric("Dependencies", g.number_of_edges())
        focus = st.selectbox("Impact analysis for table:", ["(none)"] + tables)

    focus_tbl = None if focus == "(none)" else focus
    with col2:
        st.pyplot(draw_graph(g, focus_tbl))

    if focus_tbl:
        res = impact_analysis(g, focus_tbl)
        st.subheader(f"Impact of changing `{focus_tbl}`")
        c1, c2 = st.columns(2)
        c1.markdown("**⬆️ Upstream (sources it depends on):**")
        c1.write(res["upstream"] or "_none — this is a root source_")
        c2.markdown("**⬇️ Downstream (breaks if you change it):**")
        c2.write(res["downstream"] or "_none — nothing depends on it_")


if __name__ == "__main__":
    main()
