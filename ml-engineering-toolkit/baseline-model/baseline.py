from __future__ import annotations

# Baseline Model - core logic.
#
# Every model review starts with a number and no context. "We got 0.87 AUC" -
# compared to what? A dumb baseline is the only thing that turns a metric into
# a claim, and it is the step teams skip because it feels beneath them.
#
# This module fits a ladder of deliberately stupid models (majority class,
# stratified guess, single best rule, one-feature tree) plus one honest simple
# model (logistic regression / linear), then reports the lift of the candidate
# over the strongest baseline - and refuses to call a model "good" when a
# one-line rule matches it.
#
# Runs offline on sklearn's bundled data or your own DataFrame.
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

RANDOM_SEED = 42


@dataclass
class BaselineResult:
    """One model's score on the held-out set."""

    name: str
    kind: str  # "trivial" | "simple" | "candidate"
    metrics: Dict[str, float]
    note: str = ""

    def primary(self, metric: str) -> float:
        return self.metrics.get(metric, float("nan"))


# --------------------------------------------------------------------------
# Classification ladder
# --------------------------------------------------------------------------


def _clf_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    # A rung with no ranking to offer (the majority-class baseline emits one
    # constant prediction) is worth chance. sklearn would also return 0.5 for
    # constant scores, but recording it explicitly makes the cell a deliberate
    # statement rather than an artifact - and a reader comparing rows needs a
    # value in every cell.
    if y_prob is None or len(np.unique(y_prob)) < 2:
        out["roc_auc"] = 0.5
    else:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return out


def _best_single_rule(X_tr: pd.DataFrame, y_tr: np.ndarray,
                      X_te: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Best 'if feature > threshold' rule found on train, applied to test.

    This is the baseline that most often embarrasses a real model: one feature,
    one threshold, chosen by brute force over the training set.
    """
    best = (-1.0, None, None, None)  # score, col, thresh, direction
    for col in X_tr.columns:
        vals = X_tr[col].to_numpy(dtype=float)
        # quantile grid keeps this O(features * 9) instead of O(features * rows)
        for q in np.quantile(vals, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            for direction in (1, -1):
                pred = ((vals > q) if direction == 1 else (vals <= q)).astype(int)
                s = f1_score(y_tr, pred, zero_division=0)
                if s > best[0]:
                    best = (s, col, float(q), direction)
    _, col, thresh, direction = best
    te = X_te[col].to_numpy(dtype=float)
    pred = ((te > thresh) if direction == 1 else (te <= thresh)).astype(int)
    op = ">" if direction == 1 else "<="
    return pred, f"{col} {op} {thresh:.3g}"


def run_classification(
    X: pd.DataFrame,
    y: np.ndarray,
    candidate=None,
    candidate_name: str = "candidate",
    test_size: float = 0.25,
) -> List[BaselineResult]:
    """Fit the baseline ladder plus an optional candidate; score on held-out data."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    results: List[BaselineResult] = []
    major = int(pd.Series(y_tr).mode()[0])
    prevalence = float(np.mean(y_te))

    # 1 - majority class: the floor. Accuracy here is the prevalence, which is
    # why accuracy alone is a useless metric on imbalanced data.
    d = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    results.append(BaselineResult(
        "majority class", "trivial",
        _clf_metrics(y_te, d.predict(X_te), None),
        f"always predicts {major}; accuracy = majority share",
    ))

    # 2 - stratified random: keeps the class ratio, learns nothing.
    d = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED).fit(X_tr, y_tr)
    results.append(BaselineResult(
        "stratified guess", "trivial",
        _clf_metrics(y_te, d.predict(X_te), d.predict_proba(X_te)[:, 1]),
        "random, class ratio preserved",
    ))

    # 3 - best single rule: one feature, one threshold.
    pred, rule = _best_single_rule(X_tr, y_tr, X_te)
    results.append(BaselineResult(
        "best single rule", "trivial",
        _clf_metrics(y_te, pred, pred.astype(float)),
        f"if {rule}",
    ))

    # 4 - depth-2 tree: a rule a human could memorise.
    t = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_SEED).fit(X_tr, y_tr)
    results.append(BaselineResult(
        "depth-2 tree", "simple",
        _clf_metrics(y_te, t.predict(X_te), t.predict_proba(X_te)[:, 1]),
        "at most 3 splits",
    ))

    # 5 - logistic regression: the honest simple model most projects should ship.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    results.append(BaselineResult(
        "logistic regression", "simple",
        _clf_metrics(y_te, lr.predict(X_te), lr.predict_proba(X_te)[:, 1]),
        "linear, scaled features",
    ))

    if candidate is not None:
        candidate.fit(X_tr, y_tr)
        prob = (
            candidate.predict_proba(X_te)[:, 1]
            if hasattr(candidate, "predict_proba")
            else None
        )
        results.append(BaselineResult(
            candidate_name, "candidate",
            _clf_metrics(y_te, candidate.predict(X_te), prob),
            "the model under review",
        ))

    for r in results:
        r.metrics["prevalence"] = round(prevalence, 4)
    return results


# --------------------------------------------------------------------------
# Regression ladder
# --------------------------------------------------------------------------


def _reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def run_regression(
    X: pd.DataFrame,
    y: np.ndarray,
    candidate=None,
    candidate_name: str = "candidate",
    test_size: float = 0.25,
) -> List[BaselineResult]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )
    results: List[BaselineResult] = []

    d = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
    results.append(BaselineResult(
        "predict the mean", "trivial", _reg_metrics(y_te, d.predict(X_te)),
        "the model R2 is defined against; slightly negative out-of-sample "
        "because the train mean is not the test mean",
    ))

    d = DummyRegressor(strategy="median").fit(X_tr, y_tr)
    results.append(BaselineResult(
        "predict the median", "trivial", _reg_metrics(y_te, d.predict(X_te)),
        "beats the mean on MAE when the target is skewed",
    ))

    t = DecisionTreeRegressor(max_depth=2, random_state=RANDOM_SEED).fit(X_tr, y_tr)
    results.append(BaselineResult(
        "depth-2 tree", "simple", _reg_metrics(y_te, t.predict(X_te)), "at most 3 splits",
    ))

    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    results.append(BaselineResult(
        "linear regression", "simple", _reg_metrics(y_te, lin.predict(X_te)),
        "linear, scaled features",
    ))

    if candidate is not None:
        candidate.fit(X_tr, y_tr)
        results.append(BaselineResult(
            candidate_name, "candidate", _reg_metrics(y_te, candidate.predict(X_te)),
            "the model under review",
        ))
    return results


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------

# Lift below this over the best baseline means the candidate is not carrying
# its own complexity cost. Tunable - the point is that the bar is explicit.
MIN_LIFT = 0.02


def verdict(
    results: List[BaselineResult], metric: str = "roc_auc", min_lift: float = MIN_LIFT
) -> Dict[str, object]:
    """Compare the candidate against the strongest baseline on `metric`."""
    cand = next((r for r in results if r.kind == "candidate"), None)
    baselines = [r for r in results if r.kind != "candidate"]
    if not baselines:
        return {"ok": False, "reason": "no baselines"}

    higher_is_better = metric != "mae"
    key = (lambda r: r.primary(metric)) if higher_is_better else (lambda r: -r.primary(metric))
    best_base = max(baselines, key=key)
    best_trivial = max([r for r in baselines if r.kind == "trivial"], key=key)

    out = {
        "metric": metric,
        "best_baseline": best_base.name,
        "best_baseline_score": best_base.primary(metric),
        "best_trivial": best_trivial.name,
        "best_trivial_score": best_trivial.primary(metric),
        "min_lift": min_lift,
    }
    if cand is None:
        out.update({"ok": False, "reason": "no candidate supplied", "lift": None})
        return out

    cand_score = cand.primary(metric)
    lift = (cand_score - best_base.primary(metric)) if higher_is_better else (
        best_base.primary(metric) - cand_score
    )
    out.update({
        "candidate": cand.name,
        "candidate_score": cand_score,
        "lift": round(float(lift), 4),
        "ok": bool(lift >= min_lift),
    })
    out["reason"] = (
        f"beats '{best_base.name}' by {lift:+.4f} {metric}"
        if out["ok"]
        else f"only {lift:+.4f} {metric} over '{best_base.name}' - "
             f"below the {min_lift} bar, so the added complexity is not paying for itself"
    )
    return out


def to_frame(results: List[BaselineResult], metric_order: Optional[List[str]] = None):
    rows = []
    for r in results:
        row = {"model": r.name, "kind": r.kind}
        row.update(r.metrics)
        row["note"] = r.note
        rows.append(row)
    df = pd.DataFrame(rows)
    if metric_order:
        cols = ["model", "kind"] + [c for c in metric_order if c in df.columns] + ["note"]
        df = df[cols]
    return df


def sample_classification() -> Tuple[pd.DataFrame, np.ndarray, str]:
    """Breast cancer data - a case where a one-line rule is genuinely strong."""
    from sklearn.datasets import load_breast_cancer

    d = load_breast_cancer(as_frame=True)
    return d.data, d.target.to_numpy(), "breast_cancer (569 rows, 30 features)"


def sample_regression() -> Tuple[pd.DataFrame, np.ndarray, str]:
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True)
    return d.data, d.target.to_numpy(), "diabetes (442 rows, 10 features)"


def main() -> None:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    X, y, label = sample_classification()
    print("=" * 82)
    print(f"CLASSIFICATION LADDER - {label}")
    print("=" * 82)
    res = run_classification(
        X, y, GradientBoostingClassifier(random_state=RANDOM_SEED), "gradient boosting"
    )
    print(to_frame(res, ["roc_auc", "f1", "accuracy"]).to_string(index=False))
    v = verdict(res, "roc_auc")
    print(f"\n  strongest baseline : {v['best_baseline']} ({v['best_baseline_score']})")
    print(f"  strongest TRIVIAL  : {v['best_trivial']} ({v['best_trivial_score']})")
    print(f"  candidate          : {v['candidate']} ({v['candidate_score']})")
    print(f"  lift               : {v['lift']:+.4f}   -> {'WORTH IT' if v['ok'] else 'NOT WORTH IT'}")
    print(f"  {v['reason']}")

    X, y, label = sample_regression()
    print("\n" + "=" * 82)
    print(f"REGRESSION LADDER - {label}")
    print("=" * 82)
    res = run_regression(
        X, y, GradientBoostingRegressor(random_state=RANDOM_SEED), "gradient boosting"
    )
    print(to_frame(res, ["r2", "mae"]).to_string(index=False))
    v = verdict(res, "r2")
    print(f"\n  strongest baseline : {v['best_baseline']} ({v['best_baseline_score']})")
    print(f"  candidate          : {v['candidate']} ({v['candidate_score']})")
    print(f"  lift               : {v['lift']:+.4f}   -> {'WORTH IT' if v['ok'] else 'NOT WORTH IT'}")
    print(f"  {v['reason']}")


if __name__ == "__main__":
    main()
