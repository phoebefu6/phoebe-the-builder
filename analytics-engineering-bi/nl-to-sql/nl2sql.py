from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Schema:
    """A whitelisted schema. Only these tables/columns may appear in generated SQL."""

    table: str
    columns: dict  # name -> type ('num' | 'text' | 'date')

    def column_names(self) -> list[str]:
        return list(self.columns)


@dataclass
class GuardResult:
    ok: bool
    sql: str
    issues: list[str] = field(default_factory=list)


# ------------------------------- guardrails -------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|attach|pragma|"
    r"replace|merge|vacuum)\b",
    re.I,
)
_MULTI_STMT = re.compile(r";\s*\S")


def guard_sql(sql: str, schema: Schema, max_limit: int = 1000) -> GuardResult:
    """Enforce read-only, schema-scoped SQL. This is the safety net around any generator - rule or LLM.

    Rejects: non-SELECT, DDL/DML keywords, multiple statements, unknown tables/columns.
    Injects a LIMIT if missing so a runaway query can't scan everything.
    """
    issues: list[str] = []
    clean = sql.strip().rstrip(";").strip()

    if not re.match(r"(?is)^\s*select\b", clean):
        issues.append("Only SELECT queries are allowed")
    if _FORBIDDEN.search(clean):
        issues.append("Contains a forbidden write/DDL keyword")
    if _MULTI_STMT.search(sql.strip()):
        issues.append("Multiple statements are not allowed")

    # table check
    tables = set(re.findall(r"\bfrom\s+([a-zA-Z_]\w*)", clean, re.I)) | set(
        re.findall(r"\bjoin\s+([a-zA-Z_]\w*)", clean, re.I)
    )
    for t in tables:
        if t.lower() != schema.table.lower():
            issues.append(f"Unknown table: {t}")

    # column check - identifiers that aren't columns, sql keywords, or the table.
    # strip string literals first so a value like 'pro' isn't mistaken for an identifier.
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", " ", clean)
    allowed = {c.lower() for c in schema.column_names()} | {schema.table.lower()}
    keywords = _SQL_KEYWORDS
    idents = set(re.findall(r"\b([a-zA-Z_]\w*)\b", without_strings))
    aliases = {a.lower() for a in re.findall(r"\bas\s+([a-zA-Z_]\w*)", without_strings, re.I)}
    for ident in idents:
        low = ident.lower()
        if low in keywords or low in allowed or low in aliases:
            continue
        if re.match(r"^\d", ident):
            continue
        issues.append(f"Unknown identifier: {ident}")

    # inject LIMIT
    if issues:
        return GuardResult(ok=False, sql=clean, issues=_dedupe(issues))
    if not re.search(r"\blimit\b", clean, re.I):
        clean = f"{clean}\nLIMIT {max_limit}"
    return GuardResult(ok=True, sql=clean, issues=[])


def _dedupe(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit", "as", "and", "or",
    "not", "in", "like", "between", "on", "join", "inner", "left", "right", "outer", "count",
    "sum", "avg", "min", "max", "distinct", "desc", "asc", "is", "null", "case", "when", "then",
    "else", "end", "offset",
}


# --------------------------- heuristic NL -> SQL ---------------------------

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}


def heuristic_translate(question: str, schema: Schema) -> str:
    """A tiny rule-based translator for common analytic questions. No LLM.

    Handles count / sum / average / top-N / group-by / simple equality filters over one table.
    """
    q = question.lower().strip()
    cols = {c.lower(): c for c in schema.column_names()}
    num_cols = [c for c, t in schema.columns.items() if t == "num"]

    # aggregate
    agg, agg_col = None, None
    if re.search(r"\b(how many|number of|count)\b", q):
        agg = "COUNT(*)"
    elif re.search(r"\b(total|sum of|sum)\b", q):
        col = _find_col(q, num_cols)
        agg, agg_col = (f"SUM({col})", col) if col else ("COUNT(*)", None)
    elif re.search(r"\b(average|avg|mean)\b", q):
        col = _find_col(q, num_cols)
        agg, agg_col = (f"AVG({col})", col) if col else (None, None)

    # group by - "by region", and also "top N regions" (plural dimension before 'by')
    group = None
    m = re.search(r"\b(?:by|per|for each)\s+([a-zA-Z_]+)", q)
    if m and _match_col(m.group(1), cols):
        group = _match_col(m.group(1), cols)
    else:
        mtop_dim = re.search(r"\btop\s+(?:\d+|\w+)\s+([a-zA-Z_]+)", q)
        if mtop_dim and _match_col(mtop_dim.group(1), cols):
            group = _match_col(mtop_dim.group(1), cols)

    # filter (col = value)
    where = _extract_filter(q, schema)

    # top N / order
    order, limit = None, None
    mtop = re.search(r"\btop\s+(\d+|\w+)\b", q)
    if mtop:
        n = mtop.group(1)
        limit = int(n) if n.isdigit() else _NUM_WORDS.get(n, 5)
        if agg:
            order = f"{_agg_alias(agg)} DESC"

    # assemble
    select_parts = []
    if group:
        select_parts.append(group)
    if agg:
        select_parts.append(f"{agg} AS {_agg_alias(agg)}")
    if not select_parts:
        select_parts = ["*"]

    sql = f"SELECT {', '.join(select_parts)}\nFROM {schema.table}"
    if where:
        sql += f"\nWHERE {where}"
    if group:
        sql += f"\nGROUP BY {group}"
    if order:
        sql += f"\nORDER BY {order}"
    if limit:
        sql += f"\nLIMIT {limit}"
    return sql


def _match_col(word: str, cols: dict) -> Optional[str]:
    """Match a word to a column, tolerating a trailing plural 's' (regions -> region)."""
    w = word.lower()
    if w in cols:
        return cols[w]
    if w.endswith("s") and w[:-1] in cols:
        return cols[w[:-1]]
    return None


def _agg_alias(agg: str) -> str:
    return {"COUNT(*)": "n"}.get(agg, re.sub(r"[^a-z]", "_", agg.lower()).strip("_"))


def _find_col(q: str, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c.lower() in q:
            return c
    return cols[0] if cols else None


def _extract_filter(q: str, schema: Schema) -> Optional[str]:
    cols = {c.lower(): c for c in schema.column_names()}
    # "where region is us", "for plan pro", "in region eu"
    m = re.search(r"\b(?:where|for|in|with)\s+([a-zA-Z_]+)\s+(?:is|=|of|equal to|equals)?\s*['\"]?([a-zA-Z0-9]+)['\"]?", q)
    if m and m.group(1) in cols and schema.columns.get(cols[m.group(1)]) == "text":
        return f"{cols[m.group(1)]} = '{m.group(2)}'"
    return None


# ------------------------------ execution ------------------------------

def run_sql(sql: str, df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Execute vetted SQL against a DataFrame via in-memory SQLite."""
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table, conn, index=False, if_exists="replace")
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def translate(question: str, schema: Schema, api_key: Optional[str] = None) -> GuardResult:
    """NL -> guarded SQL. Uses Claude if a key is present, else the heuristic translator.

    In BOTH cases the output passes through guard_sql - the LLM never gets to bypass the safety net.
    """
    import os

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        raw = _llm_translate(question, schema, api_key)
    else:
        raw = heuristic_translate(question, schema)
    return guard_sql(raw, schema)


def _llm_translate(question: str, schema: Schema, api_key: str) -> str:
    import anthropic

    cols = ", ".join(f"{c} ({t})" for c, t in schema.columns.items())
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Translate the question to a single read-only SQLite SELECT over table "
                    f"'{schema.table}' with columns: {cols}. Use ONLY these columns. No writes, "
                    "no other tables, no comments. Reply with ONLY the SQL.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    return resp.content[0].text.strip().strip("`").replace("sql\n", "")


SCHEMA = Schema(
    table="orders",
    columns={"order_id": "num", "amount": "num", "region": "text", "plan": "text", "status": "text"},
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "amount": [100, 50, 0, 200, 30, 80, 0, 120],
            "region": ["US", "US", "EU", "EU", "US", "APAC", "EU", "US"],
            "plan": ["pro", "pro", "free", "team", "pro", "team", "free", "pro"],
            "status": ["paid", "paid", "refunded", "paid", "paid", "paid", "refunded", "paid"],
        }
    )


SAMPLE_QUESTIONS = [
    "How many orders are there?",
    "What is the total amount by region?",
    "Show the average amount for plan pro",
    "Top 3 regions by total amount",
]
