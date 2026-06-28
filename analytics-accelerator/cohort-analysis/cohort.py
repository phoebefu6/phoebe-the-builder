from __future__ import annotations

import numpy as np
import pandas as pd


def assign_cohort(df: pd.DataFrame, user_col: str, date_col: str) -> pd.DataFrame:
    """Tag each event with the user's signup cohort month and period index."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["cohort_month"] = df.groupby(user_col)[date_col].transform("min").dt.to_period("M")
    df["event_month"] = df[date_col].dt.to_period("M")
    df["period_number"] = (
        (df["event_month"] - df["cohort_month"]).apply(lambda p: p.n)
    )
    return df


def build_retention_matrix(
    df: pd.DataFrame, user_col: str = "user_id", date_col: str = "event_date"
) -> pd.DataFrame:
    """Return a cohort x period retention-rate matrix (percent of cohort active)."""
    tagged = assign_cohort(df, user_col, date_col)

    cohort_sizes = (
        tagged[tagged["period_number"] == 0]
        .groupby("cohort_month")[user_col]
        .nunique()
    )

    active = (
        tagged.groupby(["cohort_month", "period_number"])[user_col]
        .nunique()
        .reset_index(name="active_users")
    )

    pivot = active.pivot(index="cohort_month", columns="period_number", values="active_users")
    retention = pivot.divide(cohort_sizes, axis=0) * 100
    retention.index = retention.index.astype(str)
    return retention.round(1)


def generate_sample_events(n_users: int = 500, n_months: int = 6, seed: int = 42) -> pd.DataFrame:
    """Mock signup + repeat-activity event log for demoing the tool without real data."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")

    signup_months = rng.integers(0, n_months, size=n_users)
    rows = []
    for uid in range(n_users):
        signup_month = signup_months[uid]
        signup_date = start + pd.DateOffset(months=int(signup_month), days=int(rng.integers(0, 28)))
        rows.append({"user_id": uid, "event_date": signup_date})

        decay = rng.uniform(0.55, 0.75)
        for period in range(1, n_months - signup_month):
            if rng.random() < decay**period:
                event_date = signup_date + pd.DateOffset(months=period, days=int(rng.integers(0, 28)))
                rows.append({"user_id": uid, "event_date": event_date})

    return pd.DataFrame(rows)
