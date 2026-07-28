"""Core threshold-sweep logic: pick a classification cutoff on purpose, not by default.

Pure numpy — no scikit-learn required, so the notebook runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Point:
    """Everything you need to judge one operating point."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def flag_rate(self) -> float:
        """Share of the population this cutoff sends downstream (review queue size)."""
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.fp) / n if n else 0.0

    def cost(self, cost_fp: float, cost_fn: float) -> float:
        return self.fp * cost_fp + self.fn * cost_fn

    def as_row(self, cost_fp: float = 1.0, cost_fn: float = 1.0) -> Dict[str, float]:
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "specificity": round(self.specificity, 4),
            "accuracy": round(self.accuracy, 4),
            "flag_rate": round(self.flag_rate, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "cost": round(self.cost(cost_fp, cost_fn), 2),
        }


def _validate(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    if yt.size != ys.size:
        raise ValueError(f"Length mismatch: {yt.size} labels vs {ys.size} scores.")
    if not np.isin(yt, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if np.isnan(ys).any():
        raise ValueError("y_score contains NaN.")
    if yt.sum() == 0 or yt.sum() == yt.size:
        raise ValueError("y_true has only one class — a threshold sweep is meaningless.")
    return yt, ys


def sweep(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_steps: int = 101,
    thresholds: Optional[Sequence[float]] = None,
) -> List[Point]:
    """Score every candidate cutoff. Predict positive when score >= threshold."""
    yt, ys = _validate(y_true, y_score)
    grid = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.0, 1.0, n_steps)
    )
    pts: List[Point] = []
    for t in grid:
        pred = ys >= t
        pts.append(
            Point(
                threshold=float(t),
                tp=int(np.sum(pred & (yt == 1))),
                fp=int(np.sum(pred & (yt == 0))),
                fn=int(np.sum(~pred & (yt == 1))),
                tn=int(np.sum(~pred & (yt == 0))),
            )
        )
    return pts


def best_by(points: List[Point], metric: str = "f1") -> Point:
    """Highest-scoring cutoff for a named metric (f1, precision, recall, accuracy...)."""
    if not points:
        raise ValueError("No points to choose from.")
    if not hasattr(points[0], metric):
        raise ValueError(f"Unknown metric '{metric}'.")
    return max(points, key=lambda p: getattr(p, metric))


def min_cost(points: List[Point], cost_fp: float, cost_fn: float) -> Point:
    """The cutoff that minimises real money lost, given what each error type costs."""
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("Costs must be non-negative.")
    return min(points, key=lambda p: p.cost(cost_fp, cost_fn))


def under_constraint(
    points: List[Point],
    floor_metric: str,
    floor_value: float,
    maximize: str,
    min_flags: int = 30,
) -> Optional[Point]:
    """Best `maximize` among cutoffs that clear a hard floor (e.g. precision >= 0.80).

    `min_flags` guards against the classic trap: at an extreme cutoff you flag three
    rows, get all three right, and report "precision 1.00". That is not an operating
    point, it is a small sample. Cutoffs flagging fewer than `min_flags` are excluded.

    Returns None when nothing clears the floor — an answer in itself: the model
    cannot meet that SLA at any usable cutoff.
    """
    feasible = [
        p
        for p in points
        if getattr(p, floor_metric) >= floor_value and (p.tp + p.fp) >= min_flags
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda p: getattr(p, maximize))


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Threshold-free ranking quality, via the Mann-Whitney U identity."""
    yt, ys = _validate(y_true, y_score)
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty(ys.size, dtype=float)
    ranks[order] = np.arange(1, ys.size + 1, dtype=float)
    # average ranks within ties so duplicate scores don't inflate the score
    uniq, inv, counts = np.unique(ys, return_inverse=True, return_counts=True)
    sums = np.zeros(uniq.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = float(yt.sum()), float((1 - yt).sum())
    return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def sample_scores(n: int = 4000, prevalence: float = 0.08, seed: int = 42):
    """Imbalanced fraud-style sample: 8% positives, a decent-but-imperfect model."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    # positives score higher on average, with heavy overlap — like real life
    score = np.where(
        y == 1,
        rng.beta(5.0, 2.6, size=n),
        rng.beta(2.0, 6.5, size=n),
    )
    return y, np.clip(score, 0.0, 1.0)
