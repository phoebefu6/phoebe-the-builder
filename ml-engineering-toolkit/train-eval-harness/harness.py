from __future__ import annotations

"""Train/Eval Leaderboard - cross-validated model comparison in one call.

Model selection is usually ad-hoc: someone trains three classifiers in a
notebook, eyeballs a single accuracy number, and ships whichever looked best on
one lucky split. This harness runs the same stratified cross-validation over a
roster of models, scores every fold on several metrics, and ranks them with a
mean +/- std leaderboard so the comparison is honest and reproducible.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

# Metrics scored on every fold. sklearn scorer name -> friendly label.
DEFAULT_METRICS: Dict[str, str] = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
}


def make_sample_data(n: int = 900, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic binary-churn table with mixed column types.

    Signal is intentionally moderate so models actually separate on the
    leaderboard instead of everyone hitting a ceiling of 1.0.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n).clip(15, 150)
    support_tickets = rng.poisson(1.2, n)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15])
    contract = rng.choice(["monthly", "yearly"], n, p=[0.65, 0.35])

    # Latent churn score -> probability. Short tenure, high bill, many tickets,
    # and month-to-month contracts push churn up.
    plan_risk = np.select(
        [plan == "basic", plan == "pro", plan == "enterprise"],
        [0.6, 0.2, -0.3],
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


def default_models() -> Dict[str, object]:
    """The roster. A DummyClassifier is included as an honest baseline -
    any model that cannot beat 'always predict majority' is worthless.
    """
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=0, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
    }


def _build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot categoricals. Wrapping this in
    the CV pipeline is what keeps fold statistics from leaking across splits.
    """
    features = [c for c in df.columns if c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]

    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)]
    )


def run_leaderboard(
    df: pd.DataFrame,
    target: str,
    models: Optional[Dict[str, object]] = None,
    metrics: Optional[Dict[str, str]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate every model on every metric.

    Returns (leaderboard, fold_detail):
      - leaderboard: one row per model, mean +/- std per metric, ranked by the
        first metric, plus mean fit time.
      - fold_detail: tidy long frame (model, metric, fold, score) for plots.

    Raises ValueError on a target that is missing, non-binary, or single-class.
    """
    if target not in df.columns:
        raise ValueError(f"target '{target}' not found in columns: {list(df.columns)}")

    y = df[target]
    classes = pd.unique(y.dropna())
    if len(classes) != 2:
        raise ValueError(
            f"leaderboard supports binary targets only; '{target}' has "
            f"{len(classes)} classes: {sorted(map(str, classes))[:5]}"
        )

    models = models or default_models()
    metrics = metrics or DEFAULT_METRICS

    X = df.drop(columns=[target])
    # Don't ask for more folds than the rarest class can supply.
    min_class = int(y.value_counts().min())
    n_splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    board_rows: List[dict] = []
    fold_rows: List[dict] = []

    for name, estimator in models.items():
        pipe = Pipeline(
            [
                ("pre", _build_preprocessor(df, target)),
                ("model", estimator),
            ]
        )
        scoring = list(metrics.keys())
        cv_res = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise"
        )

        row: dict = {"model": name, "fit_time_s": float(np.mean(cv_res["fit_time"]))}
        for scorer, label in metrics.items():
            scores = cv_res[f"test_{scorer}"]
            row[f"{label} mean"] = float(np.mean(scores))
            row[f"{label} std"] = float(np.std(scores))
            for i, s in enumerate(scores):
                fold_rows.append(
                    {"model": name, "metric": label, "fold": i + 1, "score": float(s)}
                )
        board_rows.append(row)

    leaderboard = pd.DataFrame(board_rows)
    rank_col = f"{list(metrics.values())[0]} mean"
    leaderboard = leaderboard.sort_values(rank_col, ascending=False).reset_index(
        drop=True
    )
    leaderboard.insert(0, "rank", leaderboard.index + 1)
    return leaderboard, pd.DataFrame(fold_rows)


def format_leaderboard(leaderboard: pd.DataFrame, metrics: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Collapse mean/std pairs into readable 'mean +/- std' strings for display."""
    metrics = metrics or DEFAULT_METRICS
    out = pd.DataFrame({"rank": leaderboard["rank"], "model": leaderboard["model"]})
    for label in metrics.values():
        m, s = leaderboard[f"{label} mean"], leaderboard[f"{label} std"]
        out[label] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m, s)]
    out["fit time (s)"] = leaderboard["fit_time_s"].round(3)
    return out


if __name__ == "__main__":
    frame = make_sample_data()
    board, _ = run_leaderboard(frame, target="churned")
    print(format_leaderboard(board).to_string(index=False))
