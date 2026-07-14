from __future__ import annotations

"""Feature Factory - auto-build a reusable sklearn ColumnTransformer from a DataFrame.

The pain: every new project you rewrite the same impute/scale/one-hot boilerplate by
hand, get the column lists slightly wrong, and leak test statistics into training.

This inspects a DataFrame, classifies each column into a role, and assembles a fitted
ColumnTransformer plus the expanded output feature names - one call, no leakage,
reusable across projects via a plain-dict spec you can save and replay.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _as_str(X):  # noqa: N803 - sklearn API; makes bool/mixed one-hot-safe & picklable
    return pd.DataFrame(X).astype(str)

# Column roles the factory knows how to build for.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
BINARY = "binary"
DATETIME = "datetime"
DROP = "drop"


@dataclass
class ColumnPlan:
    """One column's inferred role and the reason we picked it."""

    column: str
    role: str
    reason: str


@dataclass
class FeaturePlan:
    """The full inferred spec - inspect it, edit it, or save it as JSON."""

    plans: List[ColumnPlan] = field(default_factory=list)

    def by_role(self, role: str) -> List[str]:
        return [p.column for p in self.plans if p.role == role]

    def to_dict(self) -> Dict[str, str]:
        return {p.column: p.role for p in self.plans}


def infer_plan(
    df: pd.DataFrame,
    target: Optional[str] = None,
    max_cardinality: int = 20,
    id_like_unique_ratio: float = 0.95,
) -> FeaturePlan:
    """Classify every column into a feature-engineering role.

    Rules, in order:
      - the target (if given) is dropped from features
      - datetime dtype -> DATETIME (expanded to year/month/dow/hour later)
      - 2 unique non-null values -> BINARY (single 0/1 column, no one-hot blowup)
      - numeric dtype -> NUMERIC
      - near-unique object column (looks like an id/free text) -> DROP
      - high-cardinality object above max_cardinality -> DROP (would explode one-hot)
      - otherwise object/low-card -> CATEGORICAL
    """
    plans: List[ColumnPlan] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nunique = s.nunique(dropna=True)
        if target is not None and col == target:
            plans.append(ColumnPlan(col, DROP, "target column"))
        elif pd.api.types.is_datetime64_any_dtype(s):
            plans.append(ColumnPlan(col, DATETIME, "datetime dtype"))
        elif nunique <= 2 and nunique > 0:
            plans.append(ColumnPlan(col, BINARY, f"{nunique} unique values"))
        elif pd.api.types.is_numeric_dtype(s):
            plans.append(ColumnPlan(col, NUMERIC, "numeric dtype"))
        elif n and (nunique / n) >= id_like_unique_ratio:
            plans.append(ColumnPlan(col, DROP, f"id-like ({nunique}/{n} unique)"))
        elif nunique > max_cardinality:
            plans.append(
                ColumnPlan(col, DROP, f"high cardinality ({nunique} > {max_cardinality})")
            )
        else:
            plans.append(ColumnPlan(col, CATEGORICAL, f"{nunique} categories"))
    return FeaturePlan(plans)


class _DateExpander(BaseEstimator, TransformerMixin):
    """Turn each datetime column into year/month/day/dayofweek/hour numeric features."""

    parts = ("year", "month", "day", "dayofweek", "hour")

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803
        X = pd.DataFrame(X, columns=self.columns).apply(pd.to_datetime, errors="coerce")
        out = {}
        for col in self.columns:
            for part in self.parts:
                out[f"{col}_{part}"] = getattr(X[col].dt, part).fillna(-1).astype(float)
        return pd.DataFrame(out).to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"{c}_{p}" for c in self.columns for p in self.parts])


def build_transformer(df: pd.DataFrame, plan: FeaturePlan) -> ColumnTransformer:
    """Assemble a fitted-on-call ColumnTransformer from an inferred/edited plan.

    - numeric   -> median impute + standardize
    - categorical -> most-frequent impute + one-hot (unknowns ignored at score time)
    - binary    -> most-frequent impute + one-hot with a single dropped reference level
    - datetime  -> expanded to calendar parts
    """
    numeric = plan.by_role(NUMERIC)
    categorical = plan.by_role(CATEGORICAL)
    binary = plan.by_role(BINARY)
    datetime_cols = plan.by_role(DATETIME)

    transformers = []
    if numeric:
        transformers.append(
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric)
        )
    if categorical:
        transformers.append(
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical)
        )
    if binary:
        transformers.append(
            ("bin", Pipeline([
                ("str", FunctionTransformer(_as_str, feature_names_out="one-to-one")),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False)),
            ]), binary)
        )
    if datetime_cols:
        transformers.append(("dt", _DateExpander(datetime_cols), datetime_cols))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def fit_transform(df: pd.DataFrame, plan: FeaturePlan):
    """Fit the transformer and return (feature_matrix_df, ColumnTransformer)."""
    ct = build_transformer(df, plan)
    matrix = ct.fit_transform(df)
    names = ct.get_feature_names_out()
    return pd.DataFrame(matrix, columns=names, index=df.index), ct


def make_sample_data(n: int = 200) -> pd.DataFrame:
    """A messy mixed-type customer table: numerics with gaps, categories, a binary,
    a datetime, an id column, and a high-cardinality free-text column."""
    rng = np.random.default_rng(71)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],           # id-like -> drop
        "age": rng.integers(18, 80, n).astype(float),             # numeric
        "monthly_spend": rng.gamma(2.0, 40.0, n).round(2),        # numeric
        "plan": rng.choice(["free", "pro", "team", "enterprise"], n),  # categorical
        "region": rng.choice(["NA", "EU", "APAC", "LATAM"], n),   # categorical
        "is_active": rng.choice([True, False], n),                # binary
        "signup_date": pd.to_datetime("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 500, n), unit="D"),     # datetime
        "notes": [f"ticket-{rng.integers(0, 99999)}" for _ in range(n)],  # high-card -> drop
        "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),           # target
    })
    # Punch real holes so imputation earns its keep.
    df.loc[rng.choice(n, size=n // 10, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, size=n // 12, replace=False), "plan"] = np.nan
    return df
