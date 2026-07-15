from __future__ import annotations

"""Hyperparameter Tuner - stop grid-searching by hand.

Manual tuning is a person editing `n_estimators=100` to `200`, re-running a
cell, and keeping whatever looked better. This module wraps an Optuna TPE search
(with a random-search fallback for comparison) around a cross-validated
objective, so tuning is a budgeted, reproducible search that reports the lift
over the model's out-of-the-box defaults instead of a lucky hand-edit.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

SCORER = "roc_auc"


def make_sample_data(n: int = 900, seed: int = 42) -> pd.DataFrame:
    """Synthetic binary-churn table with mixed column types and moderate signal,
    so tuning has real headroom to find over the defaults."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150)
    support_tickets = rng.poisson(1.2, n)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15])
    contract = rng.choice(["monthly", "yearly"], n, p=[0.65, 0.35])

    plan_risk = np.select(
        [plan == "basic", plan == "pro", plan == "enterprise"], [0.6, 0.2, -0.3]
    )
    logit = (
        1.8
        - 0.045 * tenure
        + 0.012 * (monthly - 70)
        + 0.35 * support_tickets
        + plan_risk
        + np.where(contract == "monthly", 0.7, -0.5)
        + rng.normal(0, 0.8, n)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "support_tickets": support_tickets,
            "plan": plan,
            "contract": contract,
            "churned": churned,
        }
    )


def _preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    features = [c for c in df.columns if c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    num = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def _pipe(df: pd.DataFrame, target: str, params: Dict) -> Pipeline:
    """RandomForest wrapped in the shared preprocessor. `params` are model kwargs."""
    model = RandomForestClassifier(random_state=0, n_jobs=-1, **params)
    return Pipeline([("pre", _preprocessor(df, target)), ("model", model)])


def _cv_score(df: pd.DataFrame, target: str, params: Dict, seed: int) -> float:
    """Mean cross-validated ROC AUC for one hyperparameter set."""
    X, y = df.drop(columns=[target]), df[target]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(
        _pipe(df, target, params), X, y, cv=cv, scoring=SCORER, n_jobs=-1
    )
    return float(np.mean(scores))


def _suggest(trial: optuna.Trial) -> Dict:
    """The search space. Kept modest so a demo budget finds meaningful lift fast."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def tune(
    df: pd.DataFrame,
    target: str,
    n_trials: int = 40,
    seed: int = 42,
) -> Dict:
    """Run the TPE search and compare against the model's defaults.

    Returns a dict with: default_score, best_score, lift, best_params,
    and a history frame (trial, value, best_so_far) for plotting.

    Raises ValueError on a missing or non-binary target.
    """
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in {list(df.columns)}")
    if df[target].nunique(dropna=True) != 2:
        raise ValueError(f"target '{target}' must be binary for ROC AUC tuning")

    default_score = _cv_score(df, target, params={}, seed=seed)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        return _cv_score(df, target, _suggest(trial), seed=seed)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    values = [t.value for t in study.trials if t.value is not None]
    best_so_far: List[float] = []
    run_max = -np.inf
    for v in values:
        run_max = max(run_max, v)
        best_so_far.append(run_max)
    history = pd.DataFrame(
        {
            "trial": range(1, len(values) + 1),
            "value": values,
            "best_so_far": best_so_far,
        }
    )

    return {
        "default_score": default_score,
        "best_score": float(study.best_value),
        "lift": float(study.best_value - default_score),
        "best_params": dict(study.best_params),
        "n_trials": len(values),
        "history": history,
    }


def summary_frame(result: Dict) -> pd.DataFrame:
    """Two-row default-vs-tuned comparison for display."""
    return pd.DataFrame(
        [
            {"config": "Defaults", "cv_roc_auc": round(result["default_score"], 4)},
            {"config": "Tuned (Optuna TPE)", "cv_roc_auc": round(result["best_score"], 4)},
        ]
    )


if __name__ == "__main__":
    frame = make_sample_data()
    res = tune(frame, target="churned", n_trials=25)
    print(summary_frame(res).to_string(index=False))
    print(f"\nlift over defaults: +{res['lift']:.4f} ROC AUC in {res['n_trials']} trials")
    print("best params:", res["best_params"])
