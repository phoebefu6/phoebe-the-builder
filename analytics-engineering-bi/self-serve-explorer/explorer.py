from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExploreQuery:
    """A self-serve query spec: what to group by, what to measure, how to aggregate, filters.

    This is the whole 'no-SQL' contract - a business user fills these in, the engine does the rest.
    """

    rows: list[str] = field(default_factory=list)        # group-by dimensions
    measure: str = ""                                     # column to aggregate
    agg: str = "sum"                                      # sum | mean | count | count_distinct | min | max
    columns: Optional[str] = None                         # optional pivot dimension
    filters: dict = field(default_factory=dict)           # {col: [allowed values]}
    top_n: Optional[int] = None                           # keep top N rows by the measure
    sort_desc: bool = True


_AGGS = {
    "sum": "sum",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
}


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df
    for col, allowed in filters.items():
        if col in out.columns and allowed:
            out = out[out[col].isin(allowed)]
    return out


def run_query(df: pd.DataFrame, q: ExploreQuery) -> pd.DataFrame:
    """Execute a pivot/aggregate query - the engine that replaces 'can you pull this number?'."""
    work = apply_filters(df, q.filters)
    if work.empty:
        return work

    if not q.measure:
        raise ValueError("A measure column is required.")

    # count_distinct needs custom handling; others map to pandas agg
    if q.agg == "count_distinct":
        aggfunc = pd.Series.nunique
    else:
        if q.agg not in _AGGS:
            raise ValueError(f"Unsupported agg: {q.agg}")
        aggfunc = _AGGS[q.agg]

    if q.columns:
        result = pd.pivot_table(
            work,
            index=q.rows or None,
            columns=q.columns,
            values=q.measure,
            aggfunc=aggfunc,
            fill_value=0,
            observed=True,
        )
        result = result.reset_index() if q.rows else result
        return result

    if not q.rows:
        # scalar aggregate
        val = aggfunc(work[q.measure]) if q.agg == "count_distinct" else work[q.measure].agg(aggfunc)
        return pd.DataFrame({f"{q.agg}_{q.measure}": [val]})

    grouped = work.groupby(q.rows, observed=True)[q.measure].agg(aggfunc).reset_index()
    grouped = grouped.rename(columns={q.measure: f"{q.agg}_{q.measure}"})
    measure_col = f"{q.agg}_{q.measure}"
    grouped = grouped.sort_values(measure_col, ascending=not q.sort_desc)
    if q.top_n:
        grouped = grouped.head(q.top_n)
    return grouped.reset_index(drop=True)


def profile_columns(df: pd.DataFrame) -> dict:
    """Split columns into dimensions (group-by candidates) and measures (aggregatable)."""
    dims, measures = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 8:
            measures.append(col)
        else:
            dims.append(col)
    # numeric id-like columns can still be measures; keep any numeric as a measure option too
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return {"dimensions": dims, "measures": measures or numeric, "numeric": numeric}


def suggest_queries(df: pd.DataFrame) -> list[ExploreQuery]:
    """A few sensible starting questions so a new user isn't staring at a blank explorer."""
    prof = profile_columns(df)
    dims, measures = prof["dimensions"], prof["measures"]
    out = []
    if dims and measures:
        out.append(ExploreQuery(rows=[dims[0]], measure=measures[0], agg="sum", top_n=10))
    if len(dims) >= 2 and measures:
        out.append(ExploreQuery(rows=[dims[0]], measure=measures[0], agg="mean", columns=dims[1]))
    if dims:
        out.append(ExploreQuery(rows=[dims[0]], measure=df.columns[0], agg="count"))
    return out


def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 400
    return pd.DataFrame(
        {
            "region": rng.choice(["US", "EU", "APAC", "LATAM"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "plan": rng.choice(["free", "pro", "team", "enterprise"], n),
            "channel": rng.choice(["organic", "paid", "referral"], n),
            "amount": np.round(rng.gamma(2.0, 40, n), 2),
            "seats": rng.integers(1, 25, n),
            "month": rng.choice(["2026-04", "2026-05", "2026-06"], n),
        }
    )
