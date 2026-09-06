from __future__ import annotations

"""Core logic: profile a DataFrame and emit a Great Expectations-style suite.

Kept import-light so it runs in a notebook, a CLI, or behind Streamlit without
pulling the full Great Expectations package. The output JSON matches the GX
``ExpectationSuite`` shape, so it can be dropped straight into a GX context.
"""

import json
from typing import Any, Dict, List

import pandas as pd
from pandas.api import types as ptypes

# Columns with a unique-ratio at or above this are treated as keys.
_UNIQUE_KEY_THRESHOLD = 0.95
# Categoricals: emit an allowed-value set only when distinct count is small.
_MAX_CATEGORY_VALUES = 20
# Pad numeric min/max bounds by this fraction so real data has headroom.
_NUMERIC_PADDING = 0.10


def _expectation(kind: str, **kwargs: Any) -> Dict[str, Any]:
    """Build one GX expectation config dict."""
    return {"expectation_type": kind, "kwargs": kwargs}


def _profile_numeric(col: str, s: pd.Series) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo
    pad = abs(span) * _NUMERIC_PADDING if span else abs(hi) * _NUMERIC_PADDING
    exps.append(
        _expectation(
            "expect_column_values_to_be_between",
            column=col,
            min_value=round(lo - pad, 4),
            max_value=round(hi + pad, 4),
        )
    )
    dtype = "int" if ptypes.is_integer_dtype(s) else "float"
    exps.append(_expectation("expect_column_values_to_be_of_type", column=col, type_=dtype))
    return exps


def _profile_categorical(col: str, s: pd.Series) -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []
    distinct = s.dropna().unique().tolist()
    if 0 < len(distinct) <= _MAX_CATEGORY_VALUES:
        exps.append(
            _expectation(
                "expect_column_values_to_be_in_set",
                column=col,
                value_set=sorted(map(str, distinct)),
            )
        )
    return exps


def profile_column(col: str, s: pd.Series, row_count: int) -> List[Dict[str, Any]]:
    """Return the list of expectations inferred for one column."""
    exps: List[Dict[str, Any]] = []

    # Always assert the column exists.
    exps.append(_expectation("expect_column_to_exist", column=col))

    non_null = int(s.notna().sum())
    null_ratio = 1 - (non_null / row_count) if row_count else 0.0

    # No nulls observed -> require not-null. Some nulls -> allow up to observed rate.
    if null_ratio == 0:
        exps.append(_expectation("expect_column_values_to_not_be_null", column=col))
    else:
        exps.append(
            _expectation(
                "expect_column_values_to_not_be_null",
                column=col,
                mostly=round(1 - null_ratio, 3),
            )
        )

    # Uniqueness -> key candidate.
    if non_null:
        unique_ratio = s.nunique(dropna=True) / non_null
        if unique_ratio >= _UNIQUE_KEY_THRESHOLD:
            exps.append(_expectation("expect_column_values_to_be_unique", column=col))

    if ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s):
        exps.extend(_profile_numeric(col, s))
    else:
        exps.extend(_profile_categorical(col, s))

    return exps


def generate_suite(df: pd.DataFrame, suite_name: str = "auto_generated_suite") -> Dict[str, Any]:
    """Profile every column and assemble a full GX expectation suite."""
    row_count = len(df)
    expectations: List[Dict[str, Any]] = [
        _expectation("expect_table_row_count_to_be_between", min_value=1, max_value=row_count * 10),
        _expectation("expect_table_columns_to_match_set", column_set=list(df.columns)),
    ]
    for col in df.columns:
        expectations.extend(profile_column(col, df[col], row_count))

    return {
        "expectation_suite_name": suite_name,
        "ge_cloud_id": None,
        "meta": {"generated_by": "gx-config-generator", "source_rows": row_count},
        "expectations": expectations,
    }


def suite_to_json(suite: Dict[str, Any]) -> str:
    return json.dumps(suite, indent=2)


def summarize(suite: Dict[str, Any]) -> pd.DataFrame:
    """Roll the suite up into a per-column count of expectations for display."""
    counts: Dict[str, int] = {}
    for exp in suite["expectations"]:
        col = exp["kwargs"].get("column", "<table-level>")
        counts[col] = counts.get(col, 0) + 1
    return (
        pd.DataFrame({"column": list(counts), "expectations": list(counts.values())})
        .sort_values("expectations", ascending=False)
        .reset_index(drop=True)
    )
