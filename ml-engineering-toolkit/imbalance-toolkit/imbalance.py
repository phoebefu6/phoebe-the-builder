"""Class Imbalance Toolkit — core logic.

Compares strategies for training a classifier on imbalanced data:
  1. Baseline           — plain LogisticRegression, no rebalancing
  2. Class Weights      — class_weight="balanced"
  3. SMOTE              — synthetic minority oversampling (imblearn)
  4. Random Undersample — drop majority rows (imblearn)

The point Phoebe's fraud team keeps missing: accuracy is a trap on skewed
data. A model that predicts "not fraud" every time scores 99% accuracy and
catches zero fraud. This toolkit ranks strategies by minority-class recall
and PR-AUC — the metrics that actually matter when the rare class is the
one you care about.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_sample_data(
    n: int = 4000, fraud_rate: float = 0.03, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Generate a synthetic fraud-detection dataset with a rare positive class.

    Returns a DataFrame with numeric features + a binary ``is_fraud`` target.
    The minority class (fraud) is shifted in feature space so it's learnable
    but rare — the realistic hard case.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n * fraud_rate)))
    n_legit = n - n_fraud

    # Legit transactions: centered near zero.
    legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 4))
    # Fraud: shifted mean + wider spread on amount/velocity features.
    fraud = rng.normal(loc=1.4, scale=1.3, size=(n_fraud, 4))

    X = np.vstack([legit, fraud])
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    df = pd.DataFrame(
        X, columns=["amount_z", "velocity_z", "geo_risk", "device_age_z"]
    )
    df["is_fraud"] = y
    # Shuffle so order carries no signal.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def imbalance_report(df: pd.DataFrame, target: str) -> Dict[str, float]:
    """Summarize how skewed the target is."""
    counts = df[target].value_counts().sort_index()
    minority = int(counts.min())
    majority = int(counts.max())
    total = int(counts.sum())
    return {
        "total_rows": total,
        "minority_count": minority,
        "majority_count": majority,
        "minority_pct": round(100.0 * minority / total, 3),
        "imbalance_ratio": round(majority / max(1, minority), 1),
    }


def _build_estimators() -> Dict[str, object]:
    """Return one fitted-able estimator per strategy (scaler included)."""
    def scaler():
        return StandardScaler()

    def lr(**kw):
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **kw)

    return {
        "Baseline": ImbPipeline([("scale", scaler()), ("clf", lr())]),
        "Class Weights": ImbPipeline(
            [("scale", scaler()), ("clf", lr(class_weight="balanced"))]
        ),
        "SMOTE": ImbPipeline(
            [
                ("scale", scaler()),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", lr()),
            ]
        ),
        "Random Undersample": ImbPipeline(
            [
                ("scale", scaler()),
                ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
                ("clf", lr()),
            ]
        ),
    }


def _score(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "true_pos": int(tp),
        "false_neg": int(fn),
        "false_pos": int(fp),
    }


def compare_strategies(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.3,
    strategies: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Train every strategy on the same split and score on a held-out test set.

    Returns (leaderboard_df sorted by recall then pr_auc, probs_by_strategy).
    Edge case handled: if the minority class has too few rows to stratify or
    to run SMOTE (needs >= 2 minority samples), those strategies are skipped
    gracefully rather than crashing the whole run.
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not in dataframe")

    X = df.drop(columns=[target])
    y = df[target].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("target has only one class — nothing to classify")

    stratify = y if np.bincount(y).min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    estimators = _build_estimators()
    if strategies:
        estimators = {k: v for k, v in estimators.items() if k in strategies}

    minority_train = int(np.bincount(y_train).min())
    rows: List[Dict[str, object]] = []
    probs: Dict[str, np.ndarray] = {}

    for name, est in estimators.items():
        # SMOTE needs at least a couple minority samples; skip if too sparse.
        if name == "SMOTE" and minority_train < 2:
            continue
        est.fit(X_train, y_train)
        y_prob = est.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = _score(y_test, y_prob=y_prob, y_pred=y_pred)
        metrics["strategy"] = name
        rows.append(metrics)
        probs[name] = y_prob

    board = pd.DataFrame(rows).set_index("strategy")
    cols = [
        "recall",
        "precision",
        "f1",
        "pr_auc",
        "roc_auc",
        "accuracy",
        "true_pos",
        "false_neg",
        "false_pos",
    ]
    board = board[cols].sort_values(["recall", "pr_auc"], ascending=False)
    return board, probs


def recommend(board: pd.DataFrame) -> str:
    """One-line recommendation: best strategy by recall, with the trade-off."""
    best = board.index[0]
    base_recall = (
        float(board.loc["Baseline", "recall"]) if "Baseline" in board.index else 0.0
    )
    best_recall = float(board.loc[best, "recall"])
    lift = best_recall - base_recall
    prec = float(board.loc[best, "precision"])
    return (
        f"Use **{best}**: catches {best_recall:.0%} of the minority class "
        f"(+{lift:.0%} vs baseline) at {prec:.0%} precision."
    )


if __name__ == "__main__":
    data = make_sample_data()
    print("Imbalance:", imbalance_report(data, "is_fraud"))
    lb, _ = compare_strategies(data, target="is_fraud")
    print(lb.to_string())
    print(recommend(lb))
