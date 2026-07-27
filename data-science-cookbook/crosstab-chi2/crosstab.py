from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ChiSquareResult:
    table: pd.DataFrame            # observed contingency table
    expected: pd.DataFrame         # expected counts under independence
    residuals: pd.DataFrame        # standardized residuals (which cells drive it)
    chi2: float
    p_value: float
    dof: int
    cramers_v: float               # effect size (0 = none, 1 = perfect association)
    significant: bool


def crosstab_chi2(df: pd.DataFrame, row: str, col: str, alpha: float = 0.05) -> ChiSquareResult:
    """Contingency table + chi-square test of independence + effect size + residuals.

    Answers 'do these two groups differ?' for survey/categorical data - and, via residuals, *where*
    the difference is.
    """
    table = pd.crosstab(df[row], df[col])
    chi2, p, dof, expected = stats.chi2_contingency(table.values)
    exp_df = pd.DataFrame(expected, index=table.index, columns=table.columns)

    # standardized (Pearson) residuals: (observed - expected) / sqrt(expected)
    resid = (table.values - expected) / np.sqrt(expected)
    resid_df = pd.DataFrame(resid.round(2), index=table.index, columns=table.columns)

    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else 0.0

    return ChiSquareResult(table, exp_df.round(1), resid_df, float(chi2), float(p), int(dof),
                           round(cramers_v, 3), p < alpha)


def interpret_v(v: float) -> str:
    if v < 0.1:
        return "negligible association"
    if v < 0.3:
        return "weak association"
    if v < 0.5:
        return "moderate association"
    return "strong association"


def key_cells(result: ChiSquareResult, threshold: float = 2.0) -> list:
    """Cells whose standardized residual exceeds ~2 - the group combinations driving the result."""
    out = []
    r = result.residuals
    for i in r.index:
        for cval in r.columns:
            val = r.loc[i, cval]
            if abs(val) >= threshold:
                out.append((str(i), str(cval), float(val), "more" if val > 0 else "fewer"))
    return sorted(out, key=lambda x: abs(x[2]), reverse=True)


def narrate(result: ChiSquareResult, row: str, col: str) -> str:
    verdict = ("a statistically significant association" if result.significant
               else "no statistically significant association")
    s = (f"{row} and {col} show {verdict} (chi2={result.chi2:.1f}, p={result.p_value:.4f}), "
         f"{interpret_v(result.cramers_v)} (Cramer's V={result.cramers_v}).")
    cells = key_cells(result)
    if result.significant and cells:
        r0 = cells[0]
        s += f" Biggest driver: {r0[0]} has notably {r0[3]} '{r0[1]}' than expected ({r0[2]:+.1f}σ)."
    return s


def sample_dataframe() -> pd.DataFrame:
    """Survey responses: subscription plan vs satisfaction - with a real association baked in."""
    rng = np.random.default_rng(6)
    n = 400
    plans = rng.choice(["free", "pro", "enterprise"], n, p=[0.5, 0.35, 0.15])
    sat = []
    for p in plans:
        # enterprise skews satisfied, free skews dissatisfied
        probs = {"free": [0.45, 0.35, 0.20], "pro": [0.25, 0.40, 0.35],
                 "enterprise": [0.10, 0.30, 0.60]}[p]
        sat.append(rng.choice(["dissatisfied", "neutral", "satisfied"], p=probs))
    return pd.DataFrame({"plan": plans, "satisfaction": sat})
