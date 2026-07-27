from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass
class Projection:
    coords: np.ndarray            # (n, 2)
    method: str
    explained_variance: list      # per-component ratio (PCA only)
    labels: np.ndarray


def _numeric(df: pd.DataFrame, label_col: str | None):
    feats = df.select_dtypes("number")
    if label_col and label_col in feats.columns:
        feats = feats.drop(columns=[label_col])
    labels = df[label_col].values if label_col and label_col in df.columns else np.zeros(len(df))
    return feats, labels


def pca_project(df: pd.DataFrame, label_col: str | None = None, n_components: int = 2) -> Projection:
    """Standardize then PCA to 2D. Explained-variance tells you how much you kept."""
    feats, labels = _numeric(df, label_col)
    X = StandardScaler().fit_transform(feats.values)
    p = PCA(n_components=n_components, random_state=0)
    coords = p.fit_transform(X)
    return Projection(coords[:, :2], "PCA", [round(float(r), 4) for r in p.explained_variance_ratio_], labels)


def tsne_project(df: pd.DataFrame, label_col: str | None = None, perplexity: float = 30.0) -> Projection:
    """Standardize then t-SNE to 2D - non-linear, good at separating clusters visually."""
    feats, labels = _numeric(df, label_col)
    X = StandardScaler().fit_transform(feats.values)
    perp = min(perplexity, max(5, (len(X) - 1) / 3))
    t = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca")
    coords = t.fit_transform(X)
    return Projection(coords, "t-SNE", [], labels)


def scree(df: pd.DataFrame, label_col: str | None = None) -> list:
    """Explained-variance per component - how many dimensions you really need."""
    feats, _ = _numeric(df, label_col)
    X = StandardScaler().fit_transform(feats.values)
    p = PCA(random_state=0).fit(X)
    return [round(float(r), 4) for r in p.explained_variance_ratio_]


def sample_dataframe() -> pd.DataFrame:
    """3 gaussian blobs in 8D with a label - so a 2D projection should show 3 clusters."""
    rng = np.random.default_rng(7)
    dims = 8
    centers = [rng.normal(0, 1, dims) * 6 for _ in range(3)]
    rows, labels = [], []
    for k, ctr in enumerate(centers):
        pts = ctr + rng.normal(0, 1.5, size=(80, dims))
        rows.append(pts)
        labels += [f"group_{k}"] * 80
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(dims)])
    df["label"] = labels
    return df
