from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Metric:
    """One governed metric definition - the single source of truth for a number.

    expr is a pandas/SQL-style aggregation over a base column, e.g. 'sum(amount)' or
    'count_distinct(user_id)'. filters narrow the rows; dimensions are the allowed group-bys.
    """

    name: str
    label: str
    expr: str
    dimensions: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    owner: str = ""
    description: str = ""


# supported aggregations mapped to how we compute them on a pandas Series/GroupBy
_AGG_RE = re.compile(r"^\s*(sum|avg|mean|count|count_distinct|min|max)\s*\(\s*([a-zA-Z_][\w]*)\s*\)\s*$", re.I)


def parse_metric_yaml(text: str) -> list[Metric]:
    """Parse a minimal YAML metric store without a yaml dependency.

    Format (one metric per '- name:' block):
        - name: revenue
          label: Revenue
          expr: sum(amount)
          dimensions: [region, plan]
          filters: [status == 'paid']
          owner: finance
          description: Net paid revenue
    """
    metrics: list[Metric] = []
    current: dict = {}

    def flush() -> None:
        if current.get("name"):
            metrics.append(
                Metric(
                    name=current.get("name", ""),
                    label=current.get("label", current.get("name", "")),
                    expr=current.get("expr", ""),
                    dimensions=_as_list(current.get("dimensions", "")),
                    filters=_as_list(current.get("filters", "")),
                    owner=current.get("owner", ""),
                    description=current.get("description", ""),
                )
            )

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = re.match(r"^\s*-\s*name:\s*(.*)$", line)
        if m:
            flush()
            current = {"name": _clean(m.group(1))}
            continue
        kv = re.match(r"^\s*([a-zA-Z_]+):\s*(.*)$", line)
        if kv:
            current[kv.group(1)] = _clean(kv.group(2))
    flush()
    return metrics


def _clean(v: str) -> str:
    return v.strip().strip('"').strip("'")


def _as_list(v: str) -> list[str]:
    v = v.strip()
    if not v:
        return []
    v = v.strip("[]")
    # split on commas not inside quotes (filters may contain quoted strings)
    parts = re.split(r",(?![^']*')", v)
    return [_unwrap(p.strip()) for p in parts if p.strip()]


def _unwrap(s: str) -> str:
    """Strip surrounding quotes only when both ends match - never break an internal quote."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


@dataclass
class ValidationIssue:
    metric: str
    severity: str  # error | warning
    message: str


def validate_metrics(metrics: list[Metric]) -> list[ValidationIssue]:
    """Catch the failures that make dashboards disagree: dup names, clashing exprs, bad syntax."""
    issues: list[ValidationIssue] = []
    by_name: dict[str, list[Metric]] = {}
    exprs_by_name: dict[str, set] = {}

    for mt in metrics:
        by_name.setdefault(mt.name, []).append(mt)
        if not _AGG_RE.match(mt.expr):
            issues.append(ValidationIssue(mt.name, "error", f"Unsupported expr '{mt.expr}'"))
        if not mt.owner:
            issues.append(ValidationIssue(mt.name, "warning", "No owner assigned"))
        exprs_by_name.setdefault(mt.name, set()).add(mt.expr)

    for name, defs in by_name.items():
        if len(defs) > 1:
            if len(exprs_by_name[name]) > 1:
                issues.append(
                    ValidationIssue(name, "error", f"Conflicting definitions: {sorted(exprs_by_name[name])}")
                )
            else:
                issues.append(ValidationIssue(name, "warning", f"Defined {len(defs)} times (duplicate)"))
    return issues


def to_sql(metric: Metric, grain: Optional[list[str]] = None) -> str:
    """Render a metric to a canonical SQL string - so every tool computes it the same way."""
    m = _AGG_RE.match(metric.expr)
    if not m:
        return f"-- invalid expr for {metric.name}"
    agg, col = m.group(1).lower(), m.group(2)
    select_agg = {
        "count_distinct": f"COUNT(DISTINCT {col})",
        "avg": f"AVG({col})",
        "mean": f"AVG({col})",
    }.get(agg, f"{agg.upper()}({col})")

    grain = grain or []
    dims = [d for d in grain if d in metric.dimensions] or metric.dimensions[: len(grain)] if grain else []
    select_cols = ", ".join(dims + [f"{select_agg} AS {metric.name}"]) if dims else f"{select_agg} AS {metric.name}"
    sql = f"SELECT {select_cols}\nFROM events"
    if metric.filters:
        sql += "\nWHERE " + " AND ".join(metric.filters)
    if dims:
        sql += "\nGROUP BY " + ", ".join(dims)
    return sql


def compute(metric: Metric, df: pd.DataFrame, grain: Optional[list[str]] = None) -> pd.DataFrame:
    """Actually compute the metric on a DataFrame - the runtime proof the definition works."""
    m = _AGG_RE.match(metric.expr)
    if not m:
        raise ValueError(f"Invalid expr: {metric.expr}")
    agg, col = m.group(1).lower(), m.group(2)

    work = df.copy()
    for f in metric.filters:
        work = _apply_filter(work, f)

    grain = [g for g in (grain or []) if g in metric.dimensions and g in work.columns]

    def _agg(series: pd.Series):
        if agg == "count_distinct":
            return series.nunique()
        if agg == "count":
            return series.count()
        if agg in ("avg", "mean"):
            return series.mean()
        return getattr(series, agg)()

    if grain:
        out = work.groupby(grain)[col].apply(_agg).reset_index(name=metric.name)
        return out
    return pd.DataFrame({metric.name: [_agg(work[col])]})


def _apply_filter(df: pd.DataFrame, expr: str) -> pd.DataFrame:
    """Apply a simple 'col == value' / 'col > value' filter safely (no eval of arbitrary code)."""
    m = re.match(r"^\s*([a-zA-Z_]\w*)\s*(==|!=|>=|<=|>|<)\s*(.+)\s*$", expr)
    if not m:
        return df
    col, op, val = m.group(1), m.group(2), _clean(m.group(3))
    if col not in df.columns:
        return df
    typed: object = val
    try:
        typed = float(val) if re.match(r"^-?\d+(\.\d+)?$", val) else val
    except ValueError:
        typed = val
    ops = {
        "==": df[col] == typed, "!=": df[col] != typed,
        ">": df[col] > typed, "<": df[col] < typed,
        ">=": df[col] >= typed, "<=": df[col] <= typed,
    }
    return df[ops[op]]


SAMPLE_YAML = """\
# Metric store - the single source of truth for every dashboard
- name: revenue
  label: Net Revenue
  expr: sum(amount)
  dimensions: [region, plan]
  filters: [status == 'paid']
  owner: finance
  description: Sum of paid transaction amounts

- name: active_users
  label: Active Users
  expr: count_distinct(user_id)
  dimensions: [region, plan]
  filters: []
  owner: growth
  description: Distinct users with an event

- name: avg_order_value
  label: Average Order Value
  expr: avg(amount)
  dimensions: [region]
  filters: [status == 'paid']
  owner: finance
  description: Mean paid transaction amount
"""


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 3, 4, 5, 5],
            "amount": [100, 50, 0, 200, 30, 80, 0, 120],
            "status": ["paid", "paid", "refunded", "paid", "paid", "paid", "refunded", "paid"],
            "region": ["US", "US", "EU", "EU", "US", "APAC", "EU", "US"],
            "plan": ["pro", "pro", "free", "team", "pro", "team", "free", "pro"],
        }
    )
