from __future__ import annotations

"""Core logic: clean a messy CSV.

Real-world exports arrive with smart-quotes in headers, trailing whitespace,
blank rows, duplicate records, and "N/A"/"-" strings masquerading as data. This
module fixes the common offenders and reports exactly what it changed so the
cleanup is auditable, not magic.
"""

import re
from typing import Dict, List, Tuple

import pandas as pd

# Strings that really mean "missing".
NULL_TOKENS = {"", "na", "n/a", "null", "none", "-", "--", "nan", "?", "#n/a"}


def normalize_header(name: str) -> str:
    """snake_case a column header: trim, lowercase, collapse non-alnum to '_'."""
    name = str(name).strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def _dedupe_headers(headers: List[str]) -> List[str]:
    """Append _2, _3 ... to collided header names."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            out.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 1
            out.append(h)
    return out


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean a DataFrame, returning the result and a report of what changed."""
    report: Dict[str, int] = {}
    start_rows = len(df)

    # 1. Normalize + dedupe headers.
    df = df.copy()
    df.columns = _dedupe_headers([normalize_header(c) for c in df.columns])

    # 2. Strip whitespace on every string cell.
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    # 3. Turn null-tokens into real NaN.
    def _nullify(v: object) -> object:
        if isinstance(v, str) and v.strip().lower() in NULL_TOKENS:
            return pd.NA
        return v

    df = df.map(_nullify)

    # 4. Drop fully-empty rows and columns.
    before = len(df)
    df = df.dropna(how="all")
    report["empty_rows_dropped"] = before - len(df)

    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    report["empty_cols_dropped"] = before_cols - df.shape[1]

    # 5. Drop exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_dropped"] = before - len(df)

    # 6. Best-effort numeric coercion for object columns that are mostly numbers.
    coerced = 0
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_null = df[col].notna().sum()
        if non_null > 0 and converted.notna().sum() >= 0.9 * non_null:
            df[col] = converted
            coerced += 1
    report["columns_coerced_numeric"] = coerced

    df = df.reset_index(drop=True)
    report["rows_in"] = start_rows
    report["rows_out"] = len(df)
    return df, report


def clean_csv(in_path: str, out_path: str) -> Dict[str, int]:
    """Read a CSV, clean it, write the result. Returns the change report."""
    # `keep_default_na=False` so we control exactly what counts as null (step 3).
    df = pd.read_csv(in_path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    cleaned, report = clean_dataframe(df)
    cleaned.to_csv(out_path, index=False)
    return report
