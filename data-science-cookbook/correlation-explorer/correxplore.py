from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CorrPair:
    a: str
    b: str
    corr: float


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Numeric-only correlation matrix - the first look at 'which features relate?'."""
    num = df.select_dtypes("number")
    return num.corr(method=method)


def high_correlations(df: pd.DataFrame, threshold: float = 0.8, method: str = "pearson") -> list[CorrPair]:
    """Feature pairs whose |correlation| exceeds a threshold - redundancy candidates."""
    corr = correlation_matrix(df, method)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append(CorrPair(cols[i], cols[j], float(r)))
    return sorted(pairs, key=lambda p: abs(p.corr), reverse=True)


def vif(df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per numeric feature.

    VIF_i = 1 / (1 - R^2_i), where R^2_i comes from regressing feature i on all the others.
    VIF > 5 hints at multicollinearity; > 10 is serious. Computed with numpy least squares.
    """
    num = df.select_dtypes("number").dropna()
    cols = list(num.columns)
    X = num.values.astype(float)
    n, k = X.shape
    rows = []
    for i in range(k):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        # design matrix with intercept
        A = np.column_stack([np.ones(n), others])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        v = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
        flag = "serious" if v > 10 else "elevated" if v > 5 else "ok"
        rows.append({"feature": cols[i], "R2": round(r2, 3), "VIF": round(v, 2), "flag": flag})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def suggest_drops(df: pd.DataFrame, vif_threshold: float = 10.0) -> list[str]:
    """Greedily suggest which collinear features to drop: remove the highest-VIF one, recompute, repeat."""
    work = df.select_dtypes("number").dropna().copy()
    dropped = []
    while work.shape[1] > 2:
        v = vif(work)
        worst = v.iloc[0]
        if worst["VIF"] <= vif_threshold:
            break
        dropped.append(worst["feature"])
        work = work.drop(columns=[worst["feature"]])
    return dropped


def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 200
    height = rng.normal(170, 10, n)
    weight = height * 0.9 - 100 + rng.normal(0, 5, n)     # correlated with height
    bmi = weight / (height / 100) ** 2                     # derived -> collinear with both
    age = rng.integers(20, 60, n).astype(float)
    income = age * 800 + rng.normal(0, 5000, n)            # correlated with age
    return pd.DataFrame({"height": height, "weight": weight, "bmi": bmi, "age": age, "income": income})
