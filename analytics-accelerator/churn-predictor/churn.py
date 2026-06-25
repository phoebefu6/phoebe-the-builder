"""Churn Predictor — core logic.

Trains a gradient-boosted classifier on historical customer data (one row per
customer, a binary `churned` label) and turns it into something a retention
team can act on *before* people leave: a churn-risk score per customer, the
features that drive churn globally, and a held-out evaluation (AUC, precision,
recall) so you know whether to trust the scores.

Pure scikit-learn + pandas — no external services or API keys, so it runs
standalone in a notebook or CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = ("churn", "churned", "is_churn", "exited", "left")


@dataclass
class ChurnModel:
    model: GradientBoostingClassifier
    scaler: StandardScaler
    feature_cols: List[str]
    label_col: str
    threshold: float
    metrics: Dict[str, float] = field(default_factory=dict)
    importances: List[Tuple[str, float]] = field(default_factory=list)


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Locate the churn label: a (near-)binary column named like churn."""
    # Prefer an explicitly churn-named column
    for c in df.columns:
        if c.lower() in LABEL_CANDIDATES or "churn" in c.lower():
            if df[c].dropna().nunique() <= 2:
                return c
    # Fall back to any binary 0/1 column
    for c in df.columns:
        vals = set(pd.to_numeric(df[c], errors="coerce").dropna().unique())
        if vals <= {0, 1} and len(vals) == 2:
            return c
    return None


def _coerce_label(series: pd.Series) -> pd.Series:
    """Map a churn label to 0/1 regardless of yes/no, true/false, etc."""
    s = series.copy()
    if s.dtype == object:
        mapping = {
            "yes": 1, "no": 0, "true": 1, "false": 0,
            "churned": 1, "active": 0, "y": 1, "n": 0, "1": 1, "0": 0,
        }
        s = s.astype(str).str.strip().str.lower().map(mapping)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def select_features(df: pd.DataFrame, label_col: str) -> List[str]:
    """Numeric feature columns, excluding the label and ID-like columns."""
    cols: List[str] = []
    for c in df.columns:
        if c == label_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.9:
            continue
        if s.nunique() == len(s) and float(s.dropna().mod(1).abs().sum()) == 0.0:
            continue  # ID-like
        cols.append(c)
    return cols


def train_churn_model(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    threshold: float = 0.5,
    random_state: int = 42,
) -> ChurnModel:
    """Train + evaluate a churn classifier on a held-out split."""
    if label_col is None:
        label_col = find_label_column(df)
    if label_col is None:
        raise ValueError(
            "No churn label column found (expected a binary churn column)."
        )
    if feature_cols is None:
        feature_cols = select_features(df, label_col)
    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 numeric features to train a churn model.")

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = _coerce_label(df[label_col])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(x_train_s, y_train)

    proba = model.predict_proba(x_test_s)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "auc": round(float(roc_auc_score(y_test, proba)), 3),
        "avg_precision": round(float(average_precision_score(y_test, proba)), 3),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 3),
        "churn_rate": round(float(y.mean()), 3),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )
    importances = [(c, round(float(v), 3)) for c, v in importances]

    return ChurnModel(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        label_col=label_col,
        threshold=threshold,
        metrics=metrics,
        importances=importances,
    )


def score_customers(cm: ChurnModel, df: pd.DataFrame) -> pd.DataFrame:
    """Return per-customer churn probability + risk band, sorted high to low."""
    x = df[cm.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    proba = cm.model.predict_proba(cm.scaler.transform(x))[:, 1]
    out = df.copy()
    out["churn_risk"] = np.round(proba, 3)
    out["risk_band"] = pd.cut(
        proba,
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    )
    return out.sort_values("churn_risk", ascending=False)


def sample_customers(n: int = 800, random_state: int = 42) -> pd.DataFrame:
    """Deterministic mock base where churn depends on real drivers + noise."""
    rng = np.random.default_rng(random_state)

    tenure_months = rng.integers(1, 72, size=n)
    monthly_charges = np.round(rng.normal(70, 25, size=n).clip(15, 150), 2)
    support_tickets = rng.poisson(2, size=n)
    logins_last_month = rng.poisson(12, size=n)
    is_month_to_month = rng.integers(0, 2, size=n)

    # Latent churn propensity: short tenure, high charges, many tickets,
    # few logins, and month-to-month contracts push churn up.
    z = (
        -0.05 * tenure_months
        + 0.02 * (monthly_charges - 70)
        + 0.35 * support_tickets
        - 0.12 * logins_last_month
        + 1.1 * is_month_to_month
        + rng.normal(0, 1.0, size=n)
    )
    prob = 1 / (1 + np.exp(-z))
    churned = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_tickets": support_tickets,
            "logins_last_month": logins_last_month,
            "is_month_to_month": is_month_to_month,
            "churned": churned,
        }
    )
