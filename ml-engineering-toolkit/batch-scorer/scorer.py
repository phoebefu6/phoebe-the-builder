from __future__ import annotations

"""Batch Scoring Service - core logic.

Load a trained model bundle and score a CSV of new rows, producing
prediction probabilities and hard labels. Solves the "scoring new data is
manual" pain point with a single reusable pipeline.
"""

import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


# Meaningful churn-style feature names for the synthetic dataset.
FEATURE_NAMES: List[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_session_minutes",
    "days_since_last_login",
    "num_features_used",
    "contract_length_months",
]

DEFAULT_THRESHOLD: float = 0.5


def _make_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Build a synthetic churn-style DataFrame with labels."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=5,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.72, 0.28],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["churned"] = y
    return df


def demo_train(seed: int = 42) -> str:
    """Train a RandomForestClassifier on synthetic churn data and persist it.

    Persists a joblib bundle holding the fitted model, the feature names it
    expects, and the decision threshold. No timestamps are stored.

    Returns the path to the written ``model.joblib`` bundle.
    """
    df = _make_dataset(n_samples=2000, seed=seed)
    X = df[FEATURE_NAMES]
    y = df["churned"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)

    bundle: Dict[str, object] = {
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "threshold": DEFAULT_THRESHOLD,
    }

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str) -> Dict[str, object]:
    """Load a joblib model bundle from ``path``."""
    return joblib.load(path)


def score_frame(bundle: Dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of new rows.

    Adds two columns:
      * ``score``      - probability of the positive class
      * ``prediction`` - 0/1 hard label from the bundle threshold

    Schema reconciliation:
      * Raises ``ValueError`` listing any required feature columns that are
        missing.
      * Silently drops extra columns and reorders to match ``feature_names``.
      * Handles an empty DataFrame gracefully (returns it with the two
        columns added).
    """
    model = bundle["model"]
    feature_names: List[str] = list(bundle["feature_names"])
    threshold: float = float(bundle.get("threshold", DEFAULT_THRESHOLD))

    incoming = set(df.columns)
    required = set(feature_names)
    missing = [c for c in feature_names if c not in incoming]
    if missing:
        raise ValueError(
            "Input is missing required feature column(s): "
            + ", ".join(missing)
            + ". Expected columns: "
            + ", ".join(feature_names)
        )

    # Drop extras and reorder to the exact training schema.
    X = df[feature_names].copy()

    result = df.copy()
    if len(X) == 0:
        result["score"] = pd.Series(dtype="float64")
        result["prediction"] = pd.Series(dtype="int64")
        return result

    proba = model.predict_proba(X)[:, 1]
    result["score"] = proba
    result["prediction"] = (proba >= threshold).astype(int)
    return result


def score_csv(bundle: Dict[str, object], in_path: str, out_path: str) -> Dict[str, object]:
    """Read a CSV, score it, write the scored CSV, and return a summary dict.

    Summary keys: ``n_rows``, ``n_flagged``, ``positive_rate``.
    """
    df = pd.read_csv(in_path)
    scored = score_frame(bundle, df)
    scored.to_csv(out_path, index=False)

    n_rows = int(len(scored))
    n_flagged = int(scored["prediction"].sum()) if n_rows else 0
    positive_rate = float(n_flagged / n_rows) if n_rows else 0.0

    return {
        "n_rows": n_rows,
        "n_flagged": n_flagged,
        "positive_rate": positive_rate,
    }


def sample_new_data(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Return ``n`` unlabeled new rows sharing the training feature schema."""
    df = _make_dataset(n_samples=n, seed=seed)
    return df[FEATURE_NAMES].copy()


if __name__ == "__main__":
    model_path = demo_train()
    print(f"Trained model persisted to: {model_path}")

    loaded = load_bundle(model_path)
    new_rows = sample_new_data()
    scored = score_frame(loaded, new_rows)

    n_rows = len(scored)
    n_flagged = int(scored["prediction"].sum())
    positive_rate = n_flagged / n_rows if n_rows else 0.0

    print("--- Batch Scoring Summary ---")
    print(f"Rows scored:    {n_rows}")
    print(f"Rows flagged:   {n_flagged}")
    print(f"Positive rate:  {positive_rate:.1%}")
    print(f"Score range:    {scored['score'].min():.3f} - {scored['score'].max():.3f}")
    print(scored.head().to_string())
