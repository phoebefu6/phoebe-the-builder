from __future__ import annotations

"""Core logic: format (prettify) SQL and run lightweight safety/style lints.

Unreadable SQL hides bugs and slows reviews. This module reformats a query into a
consistent house style (keyword case, indentation, one clause per line) using the
well-tested `sqlparse` library, then layers our own **lint rules** on top - things a
formatter won't catch, like `SELECT *` or a `DELETE` with no `WHERE`.

Pure functions, no web framework - reused by the FastAPI service AND mountable as a
governed "SQL tools" app on the platform shell.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

import sqlparse


def format_sql(sql: str, *, keyword_case: str = "upper", indent_width: int = 2,
               reindent: bool = True) -> str:
    """Return prettified SQL in a consistent house style."""
    if not sql or not sql.strip():
        return ""
    return sqlparse.format(
        sql,
        keyword_case=keyword_case,        # UPPER-CASE keywords by default
        identifier_case=None,             # leave table/column names as written
        reindent=reindent,                # clause-aware line breaks + indentation
        indent_width=indent_width,
        strip_comments=False,
        use_space_around_operators=True,
    ).strip()


# ── Lint rules ────────────────────────────────────────────────────────────────
# Each rule scans the (normalized) SQL and may emit an issue. Severity:
#   "error"   - likely dangerous (e.g. unscoped DELETE)
#   "warning" - probably unintended (e.g. SELECT *)
#   "style"   - readability nit

@dataclass
class LintIssue:
    rule: str
    severity: str
    message: str


def _has_clause(sql_upper: str, clause: str) -> bool:
    return re.search(rf"\b{clause}\b", sql_upper) is not None


def lint_sql(sql: str) -> List[LintIssue]:
    """Run safety + style checks. Returns a list of issues (empty == clean)."""
    issues: List[LintIssue] = []
    if not sql or not sql.strip():
        return issues

    # Normalize: collapse whitespace, upper copy for keyword scans.
    flat = re.sub(r"\s+", " ", sql).strip()
    upper = flat.upper()

    # 1. SELECT * - hurts performance and breaks on schema change.
    if re.search(r"\bSELECT\s+\*", upper):
        issues.append(LintIssue("select_star", "warning",
                                "Avoid SELECT * - list columns explicitly for stability and performance."))

    # 2. DELETE / UPDATE without WHERE - can wipe a whole table.
    if _has_clause(upper, "DELETE") and not _has_clause(upper, "WHERE"):
        issues.append(LintIssue("delete_no_where", "error",
                                "DELETE without a WHERE clause affects every row. Add a WHERE."))
    if _has_clause(upper, "UPDATE") and not _has_clause(upper, "WHERE"):
        issues.append(LintIssue("update_no_where", "error",
                                "UPDATE without a WHERE clause changes every row. Add a WHERE."))

    # 3. Implicit join (comma in FROM) instead of explicit JOIN.
    if re.search(r"\bFROM\b[^;]*,[^;]*\bWHERE\b", upper):
        issues.append(LintIssue("implicit_join", "style",
                                "Comma-style join detected - prefer explicit JOIN ... ON for clarity."))

    # 4. Missing semicolon terminator (style for multi-statement safety).
    if not flat.endswith(";"):
        issues.append(LintIssue("no_semicolon", "style",
                                "Statement is not terminated with a semicolon."))

    # 5. Trailing whitespace on any original line.
    if any(line != line.rstrip() for line in sql.splitlines()):
        issues.append(LintIssue("trailing_whitespace", "style", "Trailing whitespace on one or more lines."))

    return issues


def analyze(sql: str, **fmt_opts) -> Dict[str, object]:
    """Format + lint in one call - the shape the API and UI both consume."""
    formatted = format_sql(sql, **fmt_opts)
    issues = lint_sql(sql)
    return {
        "formatted": formatted,
        "issues": [issue.__dict__ for issue in issues],
        "issue_count": len(issues),
        "error_count": sum(1 for i in issues if i.severity == "error"),
    }
