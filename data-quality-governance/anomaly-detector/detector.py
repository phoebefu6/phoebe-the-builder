from __future__ import annotations

# Column Anomaly Detector - scan a DataFrame column-by-column for values that
# don't belong: numeric outliers (z-score / IQR / MAD), sudden null spikes, and
# rare categorical values. Every finding says WHICH method flagged it and WHY,
# so a data steward can trust the alert instead of eyeballing a spreadsheet.
# Fully offline, standard pandas/numpy only - no API keys.
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Anomaly:
    """One flagged value (or column-level issue) with the reason it fired."""

    column: str
    method: str          # "zscore" | "iqr" | "mad" | "null-rate" | "rare-category"
    severity: str        # "high" | "medium" | "low"
    row: Optional[int]   # row index for value-level findings, None for column-level
    value: object
    reason: str


@dataclass
class ColumnReport:
    column: str
    dtype: str
    kind: str                     # "numeric" | "categorical" | "other"
    n: int
    null_rate: float
    anomalies: List[Anomaly] = field(default_factory=list)


# Thresholds are the tunable quality bar - edit for your own tolerance.
Z_THRESH = 3.0          # |z| above this is an outlier
IQR_MULT = 1.5          # Tukey fences: Q1 - k*IQR, Q3 + k*IQR
MAD_THRESH = 3.5        # modified z-score (robust to outliers) cutoff
NULL_RATE_WARN = 0.20   # column-level null rate that warrants a flag
RARE_CATEGORY_FRAC = 0.01  # category rarer than 1% of rows is suspicious


def _severity_from_z(z: float) -> str:
    az = abs(z)
    if az >= 5:
        return "high"
    if az >= 4:
        return "medium"
    return "low"


def _numeric_anomalies(col: str, s: pd.Series) -> List[Anomaly]:
    """Flag numeric outliers by three complementary methods.

    z-score catches Gaussian-ish outliers; IQR is distribution-free; MAD is
    robust when a few extreme values would otherwise inflate the mean/std and
    hide the very outliers we're hunting. A value flagged by >=2 methods is the
    most trustworthy signal.
    """
    out: List[Anomaly] = []
    vals = s.dropna()
    if len(vals) < 5 or vals.nunique() < 2:
        return out  # too small / constant to reason about

    mean, std = vals.mean(), vals.std(ddof=0)
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    lo_fence, hi_fence = q1 - IQR_MULT * iqr, q3 + IQR_MULT * iqr
    median = vals.median()
    mad = (vals - median).abs().median()

    for idx, v in vals.items():
        methods: List[str] = []
        z = (v - mean) / std if std > 0 else 0.0
        if abs(z) > Z_THRESH:
            methods.append("zscore")
        if iqr > 0 and (v < lo_fence or v > hi_fence):
            methods.append("iqr")
        # Modified z-score: 0.6745 scales MAD to be comparable to std.
        mz = 0.6745 * (v - median) / mad if mad > 0 else 0.0
        if abs(mz) > MAD_THRESH:
            methods.append("mad")

        if not methods:
            continue
        # Consensus across methods raises severity.
        base = _severity_from_z(z if "zscore" in methods else mz)
        severity = "high" if len(methods) >= 2 and base != "low" else base
        reason = (
            f"value={v:g} vs median={median:g} (z={z:.1f}, "
            f"IQR fences=[{lo_fence:g}, {hi_fence:g}], modified-z={mz:.1f}); "
            f"flagged by {', '.join(methods)}"
        )
        out.append(Anomaly(col, "+".join(methods), severity, int(idx), v, reason))
    return out


def _categorical_anomalies(col: str, s: pd.Series) -> List[Anomaly]:
    """Flag rare categories - typos, junk codes, or leaked new values."""
    out: List[Anomaly] = []
    vals = s.dropna().astype(str)
    n = len(vals)
    if n < 20:
        return out
    # Skip identifier-like columns (emails, IDs, free text): nearly every value
    # is unique, so "rare category" is meaningless and would flag everything.
    if vals.nunique() / n > 0.5:
        return out
    counts = vals.value_counts()
    threshold = max(1, int(n * RARE_CATEGORY_FRAC))
    for cat, cnt in counts.items():
        if cnt <= threshold:
            frac = cnt / n
            severity = "high" if cnt == 1 else "medium" if frac < 0.005 else "low"
            reason = (
                f"category '{cat}' appears {cnt}x ({frac:.2%} of rows) - below the "
                f"{RARE_CATEGORY_FRAC:.0%} rare-value bar; possible typo or junk code"
            )
            out.append(Anomaly(col, "rare-category", severity, None, cat, reason))
    return out


def _classify(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "numeric"
    if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
        return "categorical"
    return "other"


def scan_column(df: pd.DataFrame, col: str) -> ColumnReport:
    s = df[col]
    n = len(s)
    null_rate = float(s.isna().mean()) if n else 0.0
    kind = _classify(s)
    report = ColumnReport(col, str(s.dtype), kind, n, null_rate)

    # Column-level: a high null rate is an anomaly on its own.
    if null_rate >= NULL_RATE_WARN:
        sev = "high" if null_rate >= 0.5 else "medium"
        report.anomalies.append(
            Anomaly(col, "null-rate", sev, None, None,
                    f"null rate {null_rate:.1%} exceeds {NULL_RATE_WARN:.0%} bar")
        )

    if kind == "numeric":
        report.anomalies.extend(_numeric_anomalies(col, s))
    elif kind == "categorical":
        report.anomalies.extend(_categorical_anomalies(col, s))
    return report


def scan_dataframe(df: pd.DataFrame) -> List[ColumnReport]:
    """Scan every column; edge case: empty frame returns no reports."""
    if df is None or df.empty:
        return []
    return [scan_column(df, c) for c in df.columns]


def summarize(reports: List[ColumnReport]) -> pd.DataFrame:
    """One row per column: counts by severity - the steward's triage table."""
    rows = []
    for r in reports:
        sev = {"high": 0, "medium": 0, "low": 0}
        for a in r.anomalies:
            sev[a.severity] = sev.get(a.severity, 0) + 1
        rows.append({
            "column": r.column,
            "kind": r.kind,
            "null_rate": round(r.null_rate, 3),
            "anomalies": len(r.anomalies),
            "high": sev["high"],
            "medium": sev["medium"],
            "low": sev["low"],
        })
    return pd.DataFrame(rows)


def anomalies_frame(reports: List[ColumnReport]) -> pd.DataFrame:
    """Flat table of every individual finding, for export / review."""
    rows = []
    for r in reports:
        for a in r.anomalies:
            rows.append({
                "column": a.column, "method": a.method, "severity": a.severity,
                "row": a.row, "value": a.value, "reason": a.reason,
            })
    cols = ["column", "method", "severity", "row", "value", "reason"]
    return pd.DataFrame(rows, columns=cols)


def make_sample_data(seed: int = 42) -> pd.DataFrame:
    """Realistic-ish orders table with planted anomalies for the demo."""
    rng = np.random.default_rng(seed)
    n = 300
    amount = rng.normal(120, 30, n).round(2)
    amount[7] = 9999.0      # fat-finger / fraud spike
    amount[150] = -50.0     # impossible negative
    amount[298] = 1500.0    # milder high outlier

    age = rng.integers(18, 70, n).astype(float)
    age[42] = 199.0         # bad age

    region = rng.choice(["North", "South", "East", "West"], n, p=[.3, .3, .25, .15])
    region = region.astype(object)
    region[10] = "Nrth"     # typo -> rare category
    region[11] = "XX"       # junk code -> rare category

    email = np.array([f"user{i}@shop.com" for i in range(n)], dtype=object)
    email[np.array([3, 20, 55, 90, 130, 200, 250, 275])] = None  # null spike

    return pd.DataFrame({
        "order_amount": amount,
        "customer_age": age,
        "region": region,
        "email": email,
    })


def _cli() -> None:
    df = make_sample_data()
    reports = scan_dataframe(df)
    print("=== Column Anomaly Detector ===\n")
    print(summarize(reports).to_string(index=False))
    print("\n--- top findings ---")
    flat = anomalies_frame(reports).sort_values(
        "severity", key=lambda s: s.map({"high": 0, "medium": 1, "low": 2})
    )
    print(flat.head(12).to_string(index=False))


if __name__ == "__main__":
    _cli()
