"""Data Quality Scorecard — core logic.

Profiles any tabular dataset against six data-quality dimensions and rolls the
results up into a single 0-100 score with a letter grade, so a team can answer
"how bad is our data?" with a number instead of a shrug.

The six dimensions (Great-Expectations-inspired, but implemented natively so
there's no heavy dependency):

  1. Completeness  — how few missing values
  2. Uniqueness    — how few duplicate rows
  3. Validity      — values match expected type/range/format
  4. Consistency   — categorical columns aren't fragmented by casing/whitespace
  5. Timeliness    — date columns aren't stale or implausibly future-dated
  6. Distribution  — numeric columns aren't dominated by outliers

Pure pandas + numpy — no external services or API keys, so it runs standalone
in a notebook or CI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Dimension weights (sum to 1.0). Completeness and validity matter most.
WEIGHTS: Dict[str, float] = {
    "completeness": 0.25,
    "uniqueness": 0.15,
    "validity": 0.25,
    "consistency": 0.15,
    "timeliness": 0.10,
    "distribution": 0.10,
}


@dataclass
class Check:
    dimension: str
    column: str
    score: float          # 0-100
    detail: str


@dataclass
class Scorecard:
    overall: float
    grade: str
    dimension_scores: Dict[str, float]
    checks: List[Check]
    n_rows: int
    n_cols: int
    issues: List[str] = field(default_factory=list)


def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _is_emailish(name: str) -> bool:
    return "email" in name.lower() or "e_mail" in name.lower()


def check_completeness(df: pd.DataFrame) -> List[Check]:
    checks = []
    for c in df.columns:
        filled = float(df[c].notna().mean()) * 100
        missing = 100 - filled
        detail = f"{missing:.1f}% missing" if missing else "no missing values"
        checks.append(Check("completeness", c, round(filled, 1), detail))
    return checks


def check_uniqueness(df: pd.DataFrame) -> List[Check]:
    dup_frac = float(df.duplicated().mean())
    score = round((1 - dup_frac) * 100, 1)
    n_dup = int(df.duplicated().sum())
    detail = f"{n_dup} duplicate rows" if n_dup else "no duplicate rows"
    return [Check("uniqueness", "<rows>", score, detail)]


def check_validity(df: pd.DataFrame) -> List[Check]:
    checks = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            checks.append(Check("validity", c, 100.0, "no values to validate"))
            continue
        if _is_emailish(c) and s.dtype == object:
            valid = s.astype(str).str.match(EMAIL_RE).mean() * 100
            checks.append(
                Check("validity", c, round(valid, 1),
                      f"{100 - valid:.1f}% invalid emails")
            )
        elif pd.api.types.is_numeric_dtype(s):
            # Validity here = finite and non-absurd (no inf)
            finite = np.isfinite(s.to_numpy(dtype=float)).mean() * 100
            neg = (s < 0).mean() * 100
            detail = f"{neg:.0f}% negative" if neg else "all finite"
            checks.append(Check("validity", c, round(finite, 1), detail))
        else:
            # Object/string: validity = parseable, non-empty after strip
            nonempty = (s.astype(str).str.strip().str.len() > 0).mean() * 100
            checks.append(
                Check("validity", c, round(nonempty, 1),
                      "non-empty strings" if nonempty == 100 else "some blank strings")
            )
    return checks


def check_consistency(df: pd.DataFrame) -> List[Check]:
    """Categorical columns shouldn't fragment under casing/whitespace normalization."""
    checks = []
    for c in df.columns:
        s = df[c].dropna()
        if s.dtype != object or s.empty:
            continue
        raw = s.astype(str)
        # Only treat as categorical if low cardinality
        if raw.nunique() > max(50, 0.5 * len(raw)):
            continue
        norm = raw.str.strip().str.lower()
        raw_n, norm_n = raw.nunique(), norm.nunique()
        if raw_n == 0:
            continue
        score = round(norm_n / raw_n * 100, 1)
        collapsed = raw_n - norm_n
        detail = (
            f"{collapsed} variant(s) differ only by case/space"
            if collapsed else "categories are consistent"
        )
        checks.append(Check("consistency", c, score, detail))
    return checks or [Check("consistency", "<none>", 100.0, "no categorical columns")]


def _detect_date_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            cols.append(c)
        elif df[c].dtype == object and ("date" in c.lower() or "time" in c.lower()):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                cols.append(c)
    return cols


def check_timeliness(
    df: pd.DataFrame, now: Optional[pd.Timestamp] = None, stale_days: int = 365
) -> List[Check]:
    now = now or pd.Timestamp.utcnow().tz_localize(None)
    checks = []
    for c in _detect_date_cols(df):
        s = pd.to_datetime(df[c], errors="coerce").dropna()
        if hasattr(s.dtype, "tz") and s.dt.tz is not None:
            s = s.dt.tz_localize(None)
        if s.empty:
            continue
        future = (s > now).mean() * 100
        stale = (s < now - pd.Timedelta(days=stale_days)).mean() * 100
        score = round(max(0.0, 100 - future - 0.5 * stale), 1)
        detail = f"{future:.0f}% future-dated, {stale:.0f}% older than {stale_days}d"
        checks.append(Check("timeliness", c, score, detail))
    return checks or [Check("timeliness", "<none>", 100.0, "no date columns")]


def check_distribution(df: pd.DataFrame) -> List[Check]:
    """Penalize numeric columns where a large share are extreme outliers (IQR)."""
    checks = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty or s.nunique() < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            checks.append(Check("distribution", c, 100.0, "no spread"))
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_frac = ((s < lo) | (s > hi)).mean() * 100
        score = round(max(0.0, 100 - 4 * outlier_frac), 1)
        checks.append(
            Check("distribution", c, score, f"{outlier_frac:.1f}% extreme outliers")
        )
    return checks or [Check("distribution", "<none>", 100.0, "no numeric columns")]


def score_dataframe(
    df: pd.DataFrame, now: Optional[pd.Timestamp] = None
) -> Scorecard:
    """Run all six dimensions and roll up to a weighted 0-100 score + grade."""
    all_checks: List[Check] = []
    all_checks += check_completeness(df)
    all_checks += check_uniqueness(df)
    all_checks += check_validity(df)
    all_checks += check_consistency(df)
    all_checks += check_timeliness(df, now=now)
    all_checks += check_distribution(df)

    dim_scores: Dict[str, float] = {}
    for dim in WEIGHTS:
        vals = [c.score for c in all_checks if c.dimension == dim]
        dim_scores[dim] = round(float(np.mean(vals)), 1) if vals else 100.0

    overall = round(sum(dim_scores[d] * w for d, w in WEIGHTS.items()), 1)

    # Surface the worst offenders as actionable issues
    issues = []
    for chk in sorted(all_checks, key=lambda c: c.score)[:6]:
        if chk.score < 95:
            issues.append(f"[{chk.dimension}] {chk.column}: {chk.detail} ({chk.score})")

    return Scorecard(
        overall=overall,
        grade=_grade(overall),
        dimension_scores=dim_scores,
        checks=all_checks,
        n_rows=len(df),
        n_cols=df.shape[1],
        issues=issues,
    )


def checks_to_frame(card: Scorecard) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dimension": c.dimension, "column": c.column,
             "score": c.score, "detail": c.detail}
            for c in card.checks
        ]
    )


def sample_dirty_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Deterministic messy dataset with planted quality problems."""
    rng = np.random.default_rng(random_state)

    emails = []
    for i in range(n):
        if rng.random() < 0.12:
            emails.append("not-an-email")          # invalid
        else:
            emails.append(f"user{i}@example.com")

    countries = rng.choice(
        ["USA", "usa", " USA ", "Canada", "canada", "UK"], size=n
    )  # casing/space fragmentation

    amount = rng.normal(100, 30, size=n)
    amount[rng.integers(0, n, size=8)] = 99999      # outliers
    amount[rng.integers(0, n, size=int(n * 0.05))] = -50  # invalid negatives

    signup = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 900, size=n), unit="D"
    )
    signup_str = signup.astype(str)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "email": emails,
            "country": countries,
            "amount": np.round(amount, 2),
            "signup_date": signup_str,
        }
    )

    # Plant missing values
    miss_idx = rng.integers(0, n, size=int(n * 0.08))
    df.loc[miss_idx, "amount"] = np.nan
    # Plant duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df
