from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class OutlierRow:
    index: int
    score: float                       # anomaly score (higher = more anomalous)
    reasons: list = field(default_factory=list)  # [(feature, zscore, direction)]


def detect_outliers(df: pd.DataFrame, contamination: float = 0.05, seed: int = 0):
    """Flag anomalous rows with an Isolation Forest, then EXPLAIN each via per-feature z-scores.

    The model says which rows are weird; the z-score attribution says *why* - the feature(s) that
    make each flagged row stand out, and in which direction.
    """
    num = df.select_dtypes("number")
    X = num.values.astype(float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1e-9

    iso = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    iso.fit(X)
    raw = -iso.score_samples(X)        # higher = more anomalous
    flags = iso.predict(X) == -1

    cols = list(num.columns)
    outliers = []
    for i in np.where(flags)[0]:
        z = (X[i] - means) / stds
        order = np.argsort(np.abs(z))[::-1]
        reasons = []
        for j in order[:3]:
            if abs(z[j]) >= 1.5:
                reasons.append((cols[j], round(float(z[j]), 2), "high" if z[j] > 0 else "low"))
        outliers.append(OutlierRow(int(i), round(float(raw[i]), 3), reasons))

    outliers.sort(key=lambda o: o.score, reverse=True)
    return outliers, raw, flags


def explain_text(row: OutlierRow, df: pd.DataFrame) -> str:
    if not row.reasons:
        return "Anomalous on the multivariate pattern (no single dominant feature)."
    parts = [f"{feat} unusually {direction} ({z:+.1f}σ)" for feat, z, direction in row.reasons]
    return "; ".join(parts) + "."


def summary(outliers: list, n_total: int) -> dict:
    feat_counts: dict = {}
    for o in outliers:
        for feat, _, _ in o.reasons:
            feat_counts[feat] = feat_counts.get(feat, 0) + 1
    return {
        "outliers": len(outliers),
        "total": n_total,
        "pct": round(100 * len(outliers) / n_total, 1) if n_total else 0,
        "top_drivers": dict(sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)),
    }


def sample_dataframe() -> pd.DataFrame:
    """Mostly-normal transactions with a few planted anomalies (big amount, odd hour, many items)."""
    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame({
        "amount": rng.normal(80, 20, n).round(2),
        "items": rng.integers(1, 6, n).astype(float),
        "hour": rng.integers(8, 22, n).astype(float),
        "distance_km": rng.gamma(2, 3, n).round(1),
    })
    # planted anomalies
    df.loc[10] = [980.0, 3, 14, 6.0]      # huge amount
    df.loc[40] = [75.0, 30, 3, 5.0]       # 3am, 30 items
    df.loc[120] = [60.0, 2, 15, 140.0]    # 140 km away
    df.loc[170] = [1200.0, 25, 2, 200.0]  # everything extreme
    return df
