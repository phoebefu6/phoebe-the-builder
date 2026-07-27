from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class TestRecommendation:
    """Which test to run and why - the decision-tree output before any numbers are crunched."""

    test: str
    reason: str
    parametric: bool
    assumptions: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    test: str
    statistic: float
    p_value: float
    effect: str = ""
    conclusion: str = ""


def is_normal(x: np.ndarray, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk normality check - decides parametric vs non-parametric downstream."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return True
    if len(x) > 5000:
        x = np.random.default_rng(0).choice(x, 5000, replace=False)
    try:
        return stats.shapiro(x).pvalue > alpha
    except Exception:
        return True


def recommend(kind: str, groups: list[np.ndarray], paired: bool = False) -> TestRecommendation:
    """A compact decision tree: outcome type + #groups + pairing + normality -> the right test.

    kind: 'numeric_2groups' | 'numeric_multi' | 'categorical_2vars' | 'correlation'
    """
    if kind == "numeric_2groups":
        normal = all(is_normal(g) for g in groups)
        if paired:
            return TestRecommendation(
                "Paired t-test" if normal else "Wilcoxon signed-rank",
                "Two related measurements per subject" + (" (normal)" if normal else " (non-normal)"),
                normal, ["paired samples"] + (["approx normal differences"] if normal else []))
        return TestRecommendation(
            "Independent t-test" if normal else "Mann-Whitney U",
            "Comparing means of two independent groups" + (" (normal)" if normal else " (non-normal)"),
            normal, ["independent groups"] + (["normality", "equal variance"] if normal else []))

    if kind == "numeric_multi":
        normal = all(is_normal(g) for g in groups)
        return TestRecommendation(
            "One-way ANOVA" if normal else "Kruskal-Wallis",
            "Comparing 3+ groups" + (" (normal)" if normal else " (non-normal)"),
            normal, ["independent groups"] + (["normality", "equal variance"] if normal else []))

    if kind == "categorical_2vars":
        return TestRecommendation(
            "Chi-square test of independence", "Association between two categorical variables",
            False, ["expected cell counts >= 5"])

    if kind == "correlation":
        normal = all(is_normal(g) for g in groups)
        return TestRecommendation(
            "Pearson correlation" if normal else "Spearman correlation",
            "Relationship between two numeric variables" + (" (linear/normal)" if normal else " (monotonic)"),
            normal, ["linearity"] if normal else ["monotonic relationship"])

    return TestRecommendation("Descriptive stats only", "Could not match a test", False)


def run_test(rec: TestRecommendation, groups: list[np.ndarray], alpha: float = 0.05,
             contingency: Optional[np.ndarray] = None) -> TestResult:
    """Run the recommended test with scipy and return statistic, p-value, and a plain conclusion."""
    t = rec.test
    if t == "Independent t-test":
        st = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    elif t == "Paired t-test":
        st = stats.ttest_rel(groups[0], groups[1])
    elif t == "Mann-Whitney U":
        st = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    elif t == "Wilcoxon signed-rank":
        st = stats.wilcoxon(groups[0], groups[1])
    elif t == "One-way ANOVA":
        st = stats.f_oneway(*groups)
    elif t == "Kruskal-Wallis":
        st = stats.kruskal(*groups)
    elif t == "Chi-square test of independence":
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        st = type("R", (), {"statistic": chi2, "pvalue": p})()
    elif t == "Pearson correlation":
        r, p = stats.pearsonr(groups[0], groups[1])
        st = type("R", (), {"statistic": r, "pvalue": p})()
    elif t == "Spearman correlation":
        r, p = stats.spearmanr(groups[0], groups[1])
        st = type("R", (), {"statistic": r, "pvalue": p})()
    else:
        return TestResult(t, float("nan"), float("nan"), conclusion="No test run.")

    sig = st.pvalue < alpha
    effect = _effect(rec.test, groups, contingency)
    concl = (f"p = {st.pvalue:.4f} < {alpha}: statistically significant - reject the null."
             if sig else f"p = {st.pvalue:.4f} >= {alpha}: not significant - fail to reject the null.")
    return TestResult(t, float(st.statistic), float(st.pvalue), effect, concl)


def _effect(test: str, groups: list[np.ndarray], contingency) -> str:
    try:
        if "t-test" in test or "Mann-Whitney" in test or "Wilcoxon" in test:
            a, b = np.asarray(groups[0], float), np.asarray(groups[1], float)
            pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            d = (a.mean() - b.mean()) / pooled if pooled else 0
            return f"Cohen's d = {d:.2f}"
        if "correlation" in test:
            return "see statistic (r)"
    except Exception:
        return ""
    return ""


def sample_data(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "group_A": rng.normal(100, 15, 60),
        "group_B": rng.normal(108, 15, 60),
        "group_C": rng.normal(103, 15, 60),
        "contingency": np.array([[30, 10], [20, 40]]),  # e.g. plan x churned
    }
