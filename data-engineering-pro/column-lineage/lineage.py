from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Column-Level Lineage Parser
# ---------------------------------------------------------------------------
# Given a set of SQL model definitions (CREATE TABLE/VIEW ... AS SELECT ...),
# work out which *upstream columns* feed each *downstream column*.
#
# Table-level lineage ("model B reads from model A") is easy and most tools
# stop there. The expensive question in a real warehouse is column-level:
# "if I change orders.amount, which downstream columns break?" This parser
# answers that with pure-Python string parsing - no database, no sqlparse.
# It handles the 80% of dbt-style SELECTs teams actually write: aliases,
# qualified refs, JOINs, expressions over multiple columns, and SELECT *.
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "select", "from", "where", "join", "left", "right", "inner", "outer",
    "full", "cross", "on", "as", "and", "or", "not", "in", "is", "null",
    "group", "by", "order", "having", "limit", "distinct", "case", "when",
    "then", "else", "end", "over", "partition", "asc", "desc", "with",
    "union", "all", "using", "true", "false", "cast", "between", "like",
}


@dataclass
class Edge:
    """One column-level dependency: source_col feeds target_col."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str

    @property
    def source(self) -> str:
        return f"{self.source_table}.{self.source_column}"

    @property
    def target(self) -> str:
        return f"{self.target_table}.{self.target_column}"


@dataclass
class Lineage:
    edges: List[Edge] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def upstream_of(self, column: str) -> List[str]:
        """All source columns that feed a given target column (table.col)."""
        return sorted({e.source for e in self.edges if e.target == column})

    def downstream_of(self, column: str) -> List[str]:
        """All target columns that a given source column feeds (blast radius)."""
        return sorted({e.target for e in self.edges if e.source == column})

    def impact(self, column: str) -> List[str]:
        """Transitively resolve every column downstream of `column`."""
        seen: set[str] = set()
        stack = [column]
        while stack:
            cur = stack.pop()
            for nxt in self.downstream_of(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return sorted(seen)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split on `sep` but ignore separators nested inside parentheses."""
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _target_name(script: str, fallback: str) -> Tuple[str, str]:
    """Return (target_table, select_body) from a CREATE ... AS SELECT script."""
    m = re.search(
        r"create\s+(?:or\s+replace\s+)?(?:table|view|materialized\s+view)\s+"
        r"([\w.\"`]+)\s+as\s+(select\b.*)",
        script, flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip('"`'), m.group(2)
    # Bare SELECT - use the dict key as the model name.
    m = re.search(r"(select\b.*)", script, flags=re.IGNORECASE | re.DOTALL)
    return fallback, (m.group(1) if m else script)


def _parse_sources(select_body: str) -> Dict[str, str]:
    """Map alias -> real table name from FROM and JOIN clauses."""
    sources: Dict[str, str] = {}
    pattern = re.compile(
        r"(?:from|join)\s+([\w.\"`]+)(?:\s+(?:as\s+)?([\w\"`]+))?",
        flags=re.IGNORECASE,
    )
    for tbl, alias in pattern.findall(select_body):
        tbl = tbl.strip('"`')
        alias = (alias or "").strip('"`')
        if alias and alias.lower() not in _KEYWORDS:
            sources[alias] = tbl
        sources[tbl] = tbl  # allow referencing by full name too
    return sources


def _select_list(select_body: str) -> str:
    """Everything between SELECT and the first top-level FROM."""
    body = re.sub(r"^\s*select\s+(distinct\s+)?", "", select_body,
                  flags=re.IGNORECASE)
    depth = 0
    for i in range(len(body) - 4):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
        elif depth == 0 and re.match(r"from\b", body[i:], flags=re.IGNORECASE):
            return body[:i]
    return body


def _output_name(expr: str) -> Optional[str]:
    """Derive the output column name of a select-list item."""
    m = re.search(r"\bas\s+([\w\"`]+)\s*$", expr, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip('"`')
    tail = expr.strip().split()[-1] if expr.strip() else ""
    if re.fullmatch(r"[\w.\"`]+", tail):
        col = tail.split(".")[-1].strip('"`')
        return col if col != "*" else "*"
    return None  # unnamed expression (e.g. `sum(x) + 1` with no alias)


def _referenced_columns(expr: str) -> List[Tuple[Optional[str], str]]:
    """Pull (alias_or_none, column) references out of an expression."""
    # Drop the trailing `AS name` so the alias isn't mistaken for a column.
    expr = re.sub(r"\bas\s+[\w\"`]+\s*$", "", expr, flags=re.IGNORECASE)
    refs: List[Tuple[Optional[str], str]] = []
    for tok in re.findall(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w*]*)?", expr):
        if "." in tok:
            prefix, col = tok.split(".", 1)
            refs.append((prefix, col))
        else:
            if tok.lower() in _KEYWORDS:
                continue
            # A bare word directly followed by "(" is a function name, skip it.
            if re.search(re.escape(tok) + r"\s*\(", expr):
                continue
            refs.append((None, tok))
    return refs


def parse_models(sql_scripts: Dict[str, str]) -> Lineage:
    """Parse a dict of {model_name: sql} into column-level lineage edges."""
    lin = Lineage()
    for name, script in sql_scripts.items():
        script = _strip_comments(script)
        target, body = _target_name(script, name)
        sources = _parse_sources(body)
        default_src = None
        real_tables = {v for v in sources.values()}
        if len(real_tables) == 1:
            default_src = next(iter(real_tables))

        for item in _split_top_level(_select_list(body)):
            out_col = _output_name(item)
            refs = _referenced_columns(item)

            if item.strip() == "*" or (out_col == "*"):
                lin.warnings.append(
                    f"{target}: SELECT * detected - column lineage is approximate "
                    "until columns are named explicitly."
                )
                for tbl in real_tables:
                    lin.edges.append(Edge(tbl, "*", target, "*"))
                continue

            if out_col is None:
                lin.warnings.append(
                    f"{target}: could not name an output column in `{item.strip()}` "
                    "- add an explicit AS alias."
                )
                continue

            for prefix, col in refs:
                if prefix and prefix in sources:
                    src_tbl = sources[prefix]
                elif prefix:  # qualified with an unknown alias
                    src_tbl = prefix
                elif default_src is not None:
                    src_tbl = default_src
                else:
                    lin.warnings.append(
                        f"{target}.{out_col}: column `{col}` is unqualified and "
                        "multiple sources exist - source table is ambiguous."
                    )
                    src_tbl = "?"
                lin.edges.append(Edge(src_tbl, col, target, out_col))

    # De-duplicate edges while preserving order.
    seen: set[Tuple[str, str, str, str]] = set()
    unique: List[Edge] = []
    for e in lin.edges:
        key = (e.source_table, e.source_column, e.target_table, e.target_column)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    lin.edges = unique
    return lin


SAMPLE_MODELS: Dict[str, str] = {
    "stg_orders": """
        CREATE TABLE stg_orders AS
        SELECT
            o.id            AS order_id,
            o.customer_id   AS customer_id,
            o.amount        AS amount,
            o.created_at    AS created_at
        FROM raw.orders o
    """,
    "stg_customers": """
        CREATE TABLE stg_customers AS
        SELECT
            c.id      AS customer_id,
            c.name    AS customer_name,
            c.region  AS region
        FROM raw.customers c
    """,
    "fct_revenue": """
        CREATE VIEW fct_revenue AS
        SELECT
            o.order_id                       AS order_id,
            c.region                         AS region,
            c.customer_name                  AS customer_name,
            o.amount * 1.1                   AS amount_with_tax,
            o.amount                         AS amount
        FROM stg_orders o
        JOIN stg_customers c ON o.customer_id = c.customer_id
    """,
    "rpt_region_revenue": """
        CREATE TABLE rpt_region_revenue AS
        SELECT
            region                AS region,
            sum(amount_with_tax)  AS total_revenue
        FROM fct_revenue
        GROUP BY region
    """,
}
