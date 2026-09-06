"""Feature Importance Explainer — core logic.

Stakeholders distrust a model they can't see inside. One importance number is
easy to game; three that *agree* is evidence. This module computes feature
importance three independent ways and reports where they agree (trust) and
where they disagree (investigate) — no SHAP dependency, so it runs anywhere.

Methods:
  * impurity     — tree's built-in mean decrease in impurity (fast, biased to
                   high-cardinality features)
  * permutation  — shuffle one column, measure score drop (model-agnostic,
                   the gold-standard for "does the model actually use this?")
  * drop_column  — retrain without the column, measure score drop (honest but
                   expensive; the ground truth the other two approximate)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def demo_data(seed: int = 42) -> tuple:
    """Synthetic churn-style dataset with named features.

    3 informative, 2 redundant, 3 noise columns — so the explainer has
    something real to separate signal from noise.
    """
    X, y = make_classification(
        n_samples=1500,
        n_features=8,
        n_informative=3,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=seed,
        shuffle=False,
    )
    names = [
        "tenure_months",
        "monthly_spend",
        "support_tickets",
        "redundant_a",
        "redundant_b",
        "noise_login_hour",
        "noise_device_id",
        "noise_random",
    ]
    df = pd.DataFrame(X, columns=names)
    return df, pd.Series(y, name="churned")


def train_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> tuple:
    """Fit a RandomForest and return (model, X_train, X_test, y_train, y_test)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_tr, y_tr)
    return model, X_tr, X_te, y_tr, y_te


def _score(model, X: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def impurity_importance(model, feature_names: List[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_names)


def permutation_scores(
    model, X_test: pd.DataFrame, y_test: pd.Series, seed: int = 42, n_repeats: int = 10
) -> pd.Series:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=seed,
        scoring="roc_auc", n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_test.columns)


def drop_column_importance(
    model, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.Series:
    """Retrain without each column; importance = AUC lost by dropping it.

    Edge case: dropping a column can *raise* the score (noise removed) — we
    keep the sign so a negative value flags a feature the model is better off
    without.
    """
    baseline = _score(model, X_test, y_test)
    drops: Dict[str, float] = {}
    for col in X_train.columns:
        m = clone(model)
        m.fit(X_train.drop(columns=[col]), y_train)
        drops[col] = baseline - _score(m, X_test.drop(columns=[col]), y_test)
    return pd.Series(drops)


def _rank(s: pd.Series) -> pd.Series:
    """Rank 1 = most important."""
    return s.rank(ascending=False, method="min").astype(int)


def explain(model, X_train, y_train, X_test, y_test, seed: int = 42) -> Dict:
    """Run all three methods, normalize, rank, and flag agreement.

    Returns a dict with a tidy DataFrame plus a verdict per feature:
      * "trusted"   — top-half by all three methods
      * "noise"     — bottom-half by all three (safe to drop)
      * "review"    — methods disagree; a human should look
    """
    imp = impurity_importance(model, list(X_train.columns))
    perm = permutation_scores(model, X_test, y_test, seed=seed)
    drop = drop_column_importance(model, X_train, y_train, X_test, y_test)

    df = pd.DataFrame({"impurity": imp, "permutation": perm, "drop_column": drop})

    # Normalize each column to 0..1 for a fair side-by-side (clip negatives to 0).
    norm = df.clip(lower=0)
    norm = norm / norm.max().replace(0, np.nan)
    norm = norm.fillna(0.0)
    norm.columns = [c + "_norm" for c in norm.columns]

    ranks = pd.DataFrame({c + "_rank": _rank(df[c]) for c in df.columns})

    n = len(df)
    half = n / 2
    top_all = (ranks <= half).all(axis=1)
    bottom_all = (ranks > half).all(axis=1)
    verdict = np.where(top_all, "trusted", np.where(bottom_all, "noise", "review"))

    out = pd.concat([df, norm, ranks], axis=1)
    out["consensus"] = norm.mean(axis=1)
    out["verdict"] = verdict
    out = out.sort_values("consensus", ascending=False)

    return {
        "table": out,
        "baseline_auc": _score(model, X_test, y_test),
        "n_features": n,
        "n_trusted": int((verdict == "trusted").sum()),
        "n_noise": int((verdict == "noise").sum()),
        "n_review": int((verdict == "review").sum()),
    }


if __name__ == "__main__":
    X, y = demo_data()
    model, X_tr, X_te, y_tr, y_te = train_model(X, y)
    res = explain(model, X_tr, y_tr, X_te, y_te)
    print(f"Baseline test AUC: {res['baseline_auc']:.3f}")
    print(f"trusted={res['n_trusted']}  noise={res['n_noise']}  review={res['n_review']}\n")
    cols = ["impurity", "permutation", "drop_column", "consensus", "verdict"]
    print(res["table"][cols].round(4).to_string())
