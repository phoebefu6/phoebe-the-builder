from __future__ import annotations

"""Probability Calibration Checker - core logic.

Pain point: "Our probabilities are meaningless."

A model that outputs 0.9 should be right ~90% of the time. Many models
(naive Bayes, shallow trees, over-regularized nets) are systematically
over- or under-confident. This module quantifies that with a reliability
curve, the Brier score, and Expected Calibration Error (ECE), and compares
an uncalibrated base learner against Platt (sigmoid) and isotonic
recalibration.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def demo_data(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic binary-classification dataset.

    The features are correlated/redundant, which makes GaussianNB (our base
    learner elsewhere) violate its independence assumption and become
    over-confident - a good stress test for calibration.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X, y = make_classification(
        n_samples=6000,
        n_features=20,
        n_informative=6,
        n_redundant=8,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],
        flip_y=0.05,
        class_sep=0.8,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_models(
    X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 42
) -> Dict[str, object]:
    """Fit an uncalibrated base learner plus Platt and isotonic variants.

    GaussianNB is a deliberately mis-calibrated base learner on correlated
    features, so recalibration should visibly help.

    Returns:
        dict {"uncalibrated": base, "platt": ..., "isotonic": ...}
    """
    base = GaussianNB()
    base.fit(X_tr, y_tr)

    platt = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
    platt.fit(X_tr, y_tr)

    isotonic = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
    isotonic.fit(X_tr, y_tr)

    return {"uncalibrated": base, "platt": platt, "isotonic": isotonic}


def reliability_points(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability-curve points with equal-width probability bins.

    Empty bins are dropped (they contribute no point to the curve).

    Returns:
        mean_predicted, fraction_positive, bin_counts  (aligned, non-empty bins)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right edges; clip so prob==1.0 lands in the last bin.
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)

    mean_predicted: List[float] = []
    fraction_positive: List[float] = []
    bin_counts: List[int] = []

    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        fraction_positive.append(float(y_true[mask].mean()))
        bin_counts.append(count)

    return (
        np.array(mean_predicted),
        np.array(fraction_positive),
        np.array(bin_counts),
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error: sample-weighted mean |confidence - accuracy|.

    For binary probabilities of the positive class, per bin we compare the
    mean predicted probability (confidence) to the observed fraction of
    positives (accuracy), weighted by the share of samples in the bin.
    """
    mean_pred, frac_pos, counts = reliability_points(y_true, y_prob, n_bins=n_bins)
    if counts.sum() == 0:
        return float("nan")
    weights = counts / counts.sum()
    return float(np.sum(weights * np.abs(mean_pred - frac_pos)))


def evaluate(
    models: Dict[str, object], X_te: np.ndarray, y_te: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Score each model. Lower brier/log_loss/ece = better calibrated.

    Returns:
        DataFrame indexed by model name with columns
        [brier, log_loss, roc_auc, ece].
    """
    rows: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        prob = model.predict_proba(X_te)[:, 1]
        rows[name] = {
            "brier": float(brier_score_loss(y_te, prob)),
            "log_loss": float(log_loss(y_te, prob, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y_te, prob)),
            "ece": expected_calibration_error(y_te, prob, n_bins=n_bins),
        }
    return pd.DataFrame(rows).T[["brier", "log_loss", "roc_auc", "ece"]]


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = demo_data()
    models = fit_models(X_train, y_train)
    table = evaluate(models, X_test, y_test)
    table = table.sort_values("ece")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nProbability Calibration Checker - results (sorted by ECE, lower=better)\n")
    print(table.to_string())
    best = table.index[0]
    print(f"\nBest calibrated by ECE: {best} "
          f"(ECE={table.loc[best, 'ece']:.4f}, Brier={table.loc[best, 'brier']:.4f})")
