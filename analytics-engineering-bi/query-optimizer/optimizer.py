from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Finding:
    """One detected SQL anti-pattern, with why it hurts and how to fix it."""

    rule: str
    severity: str  # high | medium | low
    issue: str
    why: str
    suggestion: str


@dataclass
class OptimizerReport:
    findings: list[Finding] = field(default_factory=list)

    def score(self) -> int:
        """0-100 query-health score - penalize by severity so one full scan outweighs a nit."""
        w = {"high": 25, "medium": 12, "low": 5}
        penalty = sum(w[f.severity] for f in self.findings)
        return max(0, 100 - penalty)


def _strip_strings(sql: str) -> str:
    return re.sub(r"'[^']*'|\"[^\"]*\"", "''", sql)


def analyze_query(sql: str) -> OptimizerReport:
    """Heuristically scan a SELECT for common performance anti-patterns. Static, no DB needed.

    Not a cost-based optimizer - a linter that catches the mistakes that make analysts' queries slow.
    """
    findings: list[Finding] = []
    s = _strip_strings(sql)
    low = s.lower()

    def has(pat: str) -> bool:
        return re.search(pat, low, re.I | re.S) is not None

    # SELECT *
    if re.search(r"select\s+\*", low):
        findings.append(Finding(
            "select-star", "medium", "SELECT * pulls every column",
            "Reads and transfers columns you don't use; blocks covering indexes.",
            "List only the columns you need."))

    # leading-wildcard LIKE
    if re.search(r"like\s+''", low) and re.search(r"like\s+'%", sql, re.I):
        pass  # strings were stripped; check original below
    if re.search(r"like\s+'%", sql, re.I):
        findings.append(Finding(
            "leading-wildcard", "high", "LIKE '%...' can't use an index",
            "A leading wildcard forces a full scan - the index is useless.",
            "Anchor the pattern ('abc%'), or use full-text search / a trigram index."))

    # function on column in WHERE (non-sargable)
    if re.search(r"where[^)]*\b(date|upper|lower|trim|cast|year|month|coalesce)\s*\([a-z_]", low):
        findings.append(Finding(
            "non-sargable", "high", "Function wrapped around a column in WHERE",
            "Wrapping a column in a function stops the index being used (non-sargable).",
            "Rewrite so the column is bare, e.g. col >= '2026-01-01' instead of YEAR(col)=2026."))

    # no WHERE clause
    if has(r"\bfrom\b") and not has(r"\bwhere\b") and not has(r"\bgroup by\b"):
        findings.append(Finding(
            "no-filter", "high", "No WHERE clause",
            "Scans the whole table; fine for tiny tables, dangerous at scale.",
            "Add a selective WHERE filter, or confirm the table is small."))

    # SELECT ... no LIMIT (exploratory)
    if not has(r"\blimit\b") and not has(r"\bgroup by\b") and not has(r"count\s*\("):
        findings.append(Finding(
            "no-limit", "low", "No LIMIT on a row-returning query",
            "Exploratory queries without LIMIT can return millions of rows.",
            "Add LIMIT while exploring."))

    # OR in WHERE
    if re.search(r"where[^;]*\bor\b", low):
        findings.append(Finding(
            "or-conditions", "medium", "OR in the WHERE clause",
            "OR across different columns often defeats index use.",
            "Use IN (...) for one column, or split into UNION ALL of index-friendly queries."))

    # implicit/cross join (comma-join without join predicate)
    from_clause = re.search(r"from\s+(.+?)(where|group by|order by|$)", low, re.S)
    if from_clause and "," in from_clause.group(1) and not has(r"\bjoin\b"):
        # a real join predicate looks like `a.col = b.col`; `col = literal` doesn't count.
        if not re.search(r"\b\w+\.\w+\s*=\s*\w+\.\w+", low):
            findings.append(Finding(
                "cross-join", "high", "Comma join without a join condition",
                "Missing join predicate = Cartesian product (rows explode).",
                "Use explicit JOIN ... ON, or add the join keys in WHERE."))

    # SELECT DISTINCT
    if re.search(r"select\s+distinct", low):
        findings.append(Finding(
            "distinct", "low", "SELECT DISTINCT",
            "DISTINCT often hides a join that fans out rows; it also sorts/hashes everything.",
            "Check whether a JOIN is duplicating rows; dedupe at the source or use GROUP BY."))

    # ORDER BY without LIMIT
    if has(r"\border by\b") and not has(r"\blimit\b"):
        findings.append(Finding(
            "sort-no-limit", "low", "ORDER BY without LIMIT",
            "Sorting the full result set is wasted work if you don't need it ordered/topped.",
            "Add LIMIT if you only need the top rows, or drop ORDER BY."))

    # correlated subquery in SELECT
    if re.search(r"select[^;]*\(\s*select\b", low):
        findings.append(Finding(
            "subquery-in-select", "medium", "Subquery inside the SELECT list",
            "A per-row correlated subquery runs once per output row - O(n) round trips.",
            "Rewrite as a JOIN or a window function."))

    return OptimizerReport(findings=findings)


SAMPLE_QUERIES = {
    "Slow dashboard query": (
        "SELECT *\n"
        "FROM orders o, customers c\n"
        "WHERE YEAR(o.created_at) = 2026\n"
        "  AND c.name LIKE '%acme%'\n"
        "ORDER BY o.created_at"
    ),
    "Clean query": (
        "SELECT o.id, o.amount, c.region\n"
        "FROM orders o\n"
        "JOIN customers c ON c.id = o.customer_id\n"
        "WHERE o.created_at >= '2026-01-01'\n"
        "LIMIT 100"
    ),
    "OR + distinct": (
        "SELECT DISTINCT region\n"
        "FROM orders\n"
        "WHERE status = 'paid' OR status = 'pending'"
    ),
}
