"""Customer Segmentation Tool — core logic.

Takes a customer table (one row per customer, numeric behavioral columns) and
finds natural groups with KMeans. It standardizes features, picks a sensible k
automatically via silhouette score when the caller doesn't specify one, then
profiles each cluster so the segments are human-readable ("high-value loyalists",
"at-risk", etc.) rather than just numbers.

Pure scikit-learn + pandas — no external services or API keys, so it runs
standalone in a notebook or CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentProfile:
    label: int
    name: str
    size: int
    share_pct: float
    means: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    k: int
    silhouette: float
    labels: np.ndarray
    profiles: List[SegmentProfile]
    feature_cols: List[str]
    scaler: StandardScaler
    model: KMeans
    auto_k_scores: Dict[int, float] = field(default_factory=dict)


def select_numeric_features(df: pd.DataFrame) -> List[str]:
    """Numeric columns usable as clustering features.

    Drops obvious ID-like columns (all-unique integers) so a customer_id
    doesn't get treated as behavior.
    """
    cols: List[str] = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() < 0.9:
            continue
        # Skip ID-like: unique count == row count and integer-valued
        if s.nunique() == len(s) and float(s.dropna().mod(1).abs().sum()) == 0.0:
            continue
        cols.append(c)
    return cols


def choose_k(
    x_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """Pick k by best silhouette score over a small range."""
    n = x_scaled.shape[0]
    k_max = min(k_max, n - 1)
    scores: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(x_scaled)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(x_scaled, labels))
    best_k = max(scores, key=scores.get) if scores else k_min
    return best_k, scores


def _name_segment(
    means: Dict[str, float],
    global_means: Dict[str, float],
    feature_cols: List[str],
) -> str:
    """Heuristic, human-readable label from how a cluster compares to average.

    Uses spend/monetary, recency, and frequency-like columns when present;
    otherwise falls back to a generic high/low descriptor on the strongest
    deviating feature.
    """
    def rel(col: str) -> float:
        g = global_means.get(col, 0.0)
        return (means[col] - g) / (abs(g) + 1e-9) if col in means else 0.0

    # Find columns by common naming
    def find(*keys: str) -> Optional[str]:
        for c in feature_cols:
            lc = c.lower()
            if any(k in lc for k in keys):
                return c
        return None

    spend_col = find("spend", "monetary", "revenue", "value", "ltv", "total")
    recency_col = find("recency", "days_since", "last")
    freq_col = find("frequency", "orders", "purchases", "visits", "count")

    spend = rel(spend_col) if spend_col else 0.0
    recency = rel(recency_col) if recency_col else 0.0
    freq = rel(freq_col) if freq_col else 0.0

    if spend_col and freq_col:
        if spend > 0.25 and freq > 0.1:
            return "High-value loyalists"
        if recency_col and recency > 0.3 and spend < 0:
            return "At-risk / lapsing"
        if spend < -0.25 and freq < 0:
            return "Low-engagement"
        if freq > 0.25 and spend < 0.1:
            return "Frequent low-spenders"
        return "Mainstream / mid-tier"

    # Fallback: name by strongest deviation
    devs = {c: rel(c) for c in feature_cols}
    strongest = max(devs, key=lambda c: abs(devs[c]))
    direction = "High" if devs[strongest] > 0 else "Low"
    return f"{direction} {strongest}"


def segment_customers(
    df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> SegmentationResult:
    """Full pipeline: select features, scale, (auto-)pick k, cluster, profile."""
    if feature_cols is None:
        feature_cols = select_numeric_features(df)
    if len(feature_cols) < 2:
        raise ValueError(
            "Need at least 2 numeric behavioral features to segment customers."
        )

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    auto_scores: Dict[int, float] = {}
    if n_clusters is None:
        n_clusters, auto_scores = choose_k(x_scaled, random_state=random_state)

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(x_scaled)
    sil = (
        float(silhouette_score(x_scaled, labels))
        if len(set(labels)) > 1
        else 0.0
    )

    global_means = {c: float(x[c].mean()) for c in feature_cols}
    profiles: List[SegmentProfile] = []
    n = len(df)
    for lab in sorted(set(labels)):
        mask = labels == lab
        means = {c: round(float(x[c][mask].mean()), 2) for c in feature_cols}
        name = _name_segment(means, global_means, feature_cols)
        profiles.append(
            SegmentProfile(
                label=int(lab),
                name=name,
                size=int(mask.sum()),
                share_pct=round(100 * mask.sum() / n, 1),
                means=means,
            )
        )

    return SegmentationResult(
        k=n_clusters,
        silhouette=round(sil, 3),
        labels=labels,
        profiles=profiles,
        feature_cols=feature_cols,
        scaler=scaler,
        model=model,
        auto_k_scores=auto_scores,
    )


def sample_customers(n: int = 300, random_state: int = 42) -> pd.DataFrame:
    """Deterministic mock customer base with 3 latent groups."""
    rng = np.random.default_rng(random_state)

    # Three latent segments with distinct, well-separated behavior
    groups = [
        # (size, (recency_mean, sd), (freq_mean, sd), (spend_mean, sd))
        (int(n * 0.25), (8, 4), (22, 3), (1200, 120)),   # loyalists
        (int(n * 0.45), (55, 12), (8, 2), (400, 60)),    # mainstream
        (n - int(n * 0.25) - int(n * 0.45), (190, 25), (1, 1), (70, 25)),  # lapsing
    ]
    rows = []
    cid = 1000
    for size, (rec_m, rec_s), (fr_m, fr_s), (sp_m, sp_s) in groups:
        for _ in range(size):
            rows.append(
                {
                    "customer_id": cid,
                    "recency_days": max(1, int(rng.normal(rec_m, rec_s))),
                    "frequency": max(1, int(rng.normal(fr_m, fr_s))),
                    "monetary_spend": round(max(10.0, rng.normal(sp_m, sp_s)), 2),
                    "tenure_months": max(1, int(rng.normal(fr_m * 1.5, 6))),
                }
            )
            cid += 1
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=random_state)
    return df.reset_index(drop=True)
