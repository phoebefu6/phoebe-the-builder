from __future__ import annotations

"""Core logic: profile any DataFrame in one pass.

Analysts burn hours doing the same first-look on every new dataset - shape,
missingness, dtypes, cardinality, distributions, obvious quality problems. This
module does that automatically and returns a structured profile plus a list of
quality flags worth a human's attention.

Pure pandas, no UI - shared by the Streamlit app and mountable as an "Auto-EDA"
app on the platform shell. (For a heavyweight HTML report, `ydata-profiling` is
the industry tool; this is the fast, embeddable, dependency-light core.)
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _infer_kind(s: pd.Series) -> str:
    """Coarse semantic type, beyond the raw dtype."""
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    # Object/string: decide categorical vs free text by cardinality ratio.
    non_null = s.dropna()
    if non_null.empty:
        return "empty"
    ratio = non_null.nunique() / len(non_null)
    return "categorical" if ratio < 0.5 else "text"


def profile_column(s: pd.Series) -> Dict[str, Any]:
    n = len(s)
    missing = int(s.isna().sum())
    non_null = s.dropna()
    kind = _infer_kind(s)
    prof: Dict[str, Any] = {
        "name": s.name,
        "dtype": str(s.dtype),
        "kind": kind,
        "missing": missing,
        "missing_pct": round(100 * missing / n, 1) if n else 0.0,
        "unique": int(non_null.nunique()),
        "unique_pct": round(100 * non_null.nunique() / len(non_null), 1) if len(non_null) else 0.0,
    }

    if kind == "numeric":
        prof.update(
            min=float(non_null.min()) if len(non_null) else None,
            max=float(non_null.max()) if len(non_null) else None,
            mean=round(float(non_null.mean()), 4) if len(non_null) else None,
            median=float(non_null.median()) if len(non_null) else None,
            std=round(float(non_null.std()), 4) if len(non_null) > 1 else None,
            zeros=int((non_null == 0).sum()),
        )
    elif kind in {"categorical", "boolean", "text"}:
        top = non_null.value_counts().head(5)
        prof["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
    return prof


def quality_flags(df: pd.DataFrame, profiles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Surface common data-quality problems for human review."""
    flags: List[Dict[str, str]] = []
    n = len(df)

    dups = int(df.duplicated().sum())
    if dups:
        flags.append({"severity": "warning", "column": "(table)",
                      "message": f"{dups} duplicate row(s)."})

    for p in profiles:
        col = str(p["name"])
        if p["missing_pct"] >= 50:
            flags.append({"severity": "error", "column": col,
                          "message": f"{p['missing_pct']}% missing - mostly empty."})
        elif p["missing_pct"] >= 20:
            flags.append({"severity": "warning", "column": col,
                          "message": f"{p['missing_pct']}% missing."})
        if p["unique"] <= 1 and n > 1:
            flags.append({"severity": "warning", "column": col,
                          "message": "constant column (single value) - no signal."})
        if p["kind"] == "categorical" and p["unique_pct"] >= 95 and n > 20:
            flags.append({"severity": "info", "column": col,
                          "message": "near-unique - looks like an ID, not a category."})
    return flags


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Full profile: table overview + per-column profiles + quality flags."""
    profiles = [profile_column(df[c]) for c in df.columns]
    overview = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_cells_pct": round(100 * df.isna().sum().sum() / (df.size or 1), 1),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "kinds": _kind_counts(profiles),
    }
    return {
        "overview": overview,
        "columns": profiles,
        "flags": quality_flags(df, profiles),
    }


def _kind_counts(profiles: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for p in profiles:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
    return counts


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix over numeric columns (empty frame if <2 numeric cols)."""
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True).round(2)
