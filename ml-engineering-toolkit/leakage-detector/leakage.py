from __future__ import annotations

"""Data Leakage Detector - core logic.

A set of independent, heuristic leak checks that look for the two classic
failure modes behind "great CV, terrible in prod":

  1. Target leakage  - a feature secretly encodes the answer.
  2. Train/test leakage - the same rows appear in both splits, so the
     validation score is optimistic.

Each check returns structured findings. `run_all` orchestrates them into a
single verdict.
"""

from typing import Optional, List, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _encode_target(y: pd.Series) -> Optional[pd.Series]:
    """Return a numeric encoding of the target, or None if not possible.

    Numeric targets pass through. Two-class categorical/object/bool targets
    are encoded 0/1. Multi-class categorical targets return None (the
    correlation/AUC checks are only meaningful for numeric or binary).
    """
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        return y.astype(float)
    # bool or categorical/object
    uniques = pd.Series(y.dropna().unique())
    if len(uniques) == 2:
        ordered = sorted(uniques.tolist(), key=lambda v: str(v))
        mapping = {ordered[0]: 0.0, ordered[1]: 1.0}
        return y.map(mapping).astype(float)
    return None


def _numeric_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def _simple_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney U). Returns None if degenerate.

    Falls back gracefully without sklearn so the check works anywhere.
    """
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    # A feature can be inversely predictive; report the stronger direction.
    return float(max(auc, 1.0 - auc))


def _finding(check: str, severity: str, feature: Optional[str], detail: str) -> Dict:
    return {"check": check, "severity": severity, "feature": feature, "detail": detail}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_target_correlation(df: pd.DataFrame, target: str, thresh: float = 0.95) -> List[Dict]:
    """Flag numeric features whose |correlation| with the target exceeds thresh.

    A feature almost perfectly correlated with the target is suspiciously
    predictive and usually a leak (the answer copied into a column).
    """
    findings: List[Dict] = []
    y = _encode_target(df[target])
    if y is None:
        return findings
    for col in _numeric_feature_columns(df, target):
        x = df[col].astype(float)
        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 3 or pair.iloc[:, 0].std() == 0:
            continue
        corr = float(np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1])
        if np.isnan(corr):
            continue
        if abs(corr) >= thresh:
            findings.append(
                _finding(
                    "target_correlation",
                    "high",
                    col,
                    f"|corr| with target = {abs(corr):.3f} (>= {thresh}); "
                    "feature almost perfectly tracks the target.",
                )
            )
    return findings


def check_single_feature_auc(df: pd.DataFrame, target: str, thresh: float = 0.90) -> List[Dict]:
    """Flag any single feature that on its own achieves AUC >= thresh.

    Uses the raw numeric feature value as a classification score (rank AUC).
    A lone feature that nearly solves a binary task is a classic leak signal.
    """
    findings: List[Dict] = []
    y = _encode_target(df[target])
    if y is None:
        return findings
    labels = y.to_numpy(dtype=float)
    # Only meaningful for a binary target.
    if set(np.unique(labels[~np.isnan(labels)]).tolist()) - {0.0, 1.0}:
        return findings
    for col in _numeric_feature_columns(df, target):
        scores = df[col].to_numpy(dtype=float)
        auc = _simple_auc(scores, labels)
        if auc is None:
            continue
        if auc >= thresh:
            findings.append(
                _finding(
                    "single_feature_auc",
                    "high",
                    col,
                    f"single-feature AUC = {auc:.3f} (>= {thresh}); "
                    "one feature nearly solves the task alone.",
                )
            )
    return findings


def check_duplicate_rows(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict]:
    """Count rows in test that also appear (exactly) in train.

    Shared common columns are used for the comparison. Overlap means the
    model is being validated on rows it was trained on -> contamination.
    """
    findings: List[Dict] = []
    common = [c for c in train_df.columns if c in test_df.columns]
    if not common:
        return findings
    train_keys = train_df[common].astype(str).agg("||".join, axis=1)
    test_keys = test_df[common].astype(str).agg("||".join, axis=1)
    train_set = set(train_keys.tolist())
    overlap_mask = test_keys.isin(train_set)
    count = int(overlap_mask.sum())
    frac = count / len(test_df) if len(test_df) else 0.0
    if count > 0:
        severity = "high" if frac >= 0.01 else "medium"
        findings.append(
            _finding(
                "duplicate_rows",
                severity,
                None,
                f"{count} test rows ({frac:.1%}) also appear in train "
                "(exact duplicates) -> train/test contamination.",
            )
        )
    return findings


def check_id_like_columns(df: pd.DataFrame) -> List[Dict]:
    """Flag columns that look like identifiers.

    Near-unique, monotonic, or id/uuid/index-named columns can leak row
    identity or encode ordering that correlates with the label.
    """
    findings: List[Dict] = []
    n = len(df)
    if n == 0:
        return findings
    id_tokens = ("id", "uuid", "guid", "index", "idx", "key", "identifier")
    for col in df.columns:
        name = str(col).lower()
        name_hit = any(tok in name for tok in id_tokens)
        s = df[col]
        nunique = s.nunique(dropna=True)
        uniqueness = nunique / n if n else 0.0
        # Continuous floats are naturally near-unique, so that alone is not
        # id-like. Restrict the near-unique signal to integer/string columns.
        is_float = (
            pd.api.types.is_float_dtype(s)
            and not pd.api.types.is_bool_dtype(s)
        )
        near_unique = uniqueness >= 0.98 and not is_float
        monotonic = False
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals = s.dropna()
            if len(vals) >= 3:
                monotonic = bool(vals.is_monotonic_increasing or vals.is_monotonic_decreasing)
        if name_hit or near_unique or monotonic:
            reasons: List[str] = []
            if name_hit:
                reasons.append("name matches id/uuid/index")
            if near_unique:
                reasons.append(f"near-unique ({uniqueness:.0%} distinct)")
            if monotonic:
                reasons.append("monotonic sequence")
            severity = "high" if (near_unique or monotonic) else "medium"
            findings.append(
                _finding(
                    "id_like_column",
                    severity,
                    col,
                    "looks like an identifier: " + ", ".join(reasons)
                    + "; can leak row identity.",
                )
            )
    return findings


def check_train_test_distribution(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str
) -> List[Dict]:
    """Compare the positive rate of a binary target across the two splits.

    A wildly different positive rate between train and test is a sign of a
    bad (non-random / leaky) split.
    """
    findings: List[Dict] = []
    if target not in train_df.columns or target not in test_df.columns:
        return findings
    y_tr = _encode_target(train_df[target])
    y_te = _encode_target(test_df[target])
    if y_tr is None or y_te is None:
        return findings
    vals = set(np.unique(np.concatenate([
        y_tr.dropna().to_numpy(), y_te.dropna().to_numpy()
    ])).tolist())
    if vals - {0.0, 1.0}:
        return findings  # not binary
    rate_tr = float(y_tr.mean())
    rate_te = float(y_te.mean())
    diff = abs(rate_tr - rate_te)
    flag = diff >= 0.10
    if flag:
        severity = "high" if diff >= 0.25 else "medium"
        findings.append(
            _finding(
                "train_test_distribution",
                severity,
                target,
                f"positive rate train={rate_tr:.2%} vs test={rate_te:.2%} "
                f"(gap {diff:.2%}); possible bad split.",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_all(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> Dict:
    """Run every check and produce a combined verdict.

    Returns
    -------
    dict with keys:
        findings : list of finding dicts (check, severity, feature, detail)
        summary  : {n_high, n_medium, n_low}
        verdict  : "leaky" if any high finding, else "clean"
    """
    findings: List[Dict] = []
    findings += check_target_correlation(train_df, target)
    findings += check_single_feature_auc(train_df, target)
    findings += check_duplicate_rows(train_df, test_df)
    findings += check_id_like_columns(train_df)
    findings += check_train_test_distribution(train_df, test_df, target)

    n_high = sum(1 for f in findings if f["severity"] == "high")
    n_medium = sum(1 for f in findings if f["severity"] == "medium")
    n_low = sum(1 for f in findings if f["severity"] == "low")

    verdict = "leaky" if n_high >= 1 else "clean"
    return {
        "findings": findings,
        "summary": {"n_high": n_high, "n_medium": n_medium, "n_low": n_low},
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------
def demo_leaky_data(seed: int = 42):
    """Return (train_df, test_df, target_name) with DELIBERATE leaks injected.

    Injected problems:
      * `leaky_score` = target + tiny noise      -> target leakage
      * `row_id`      = monotonic unique id      -> id-like leak
      * overlapping duplicate rows train<->test  -> train/test contamination
    """
    rng = np.random.default_rng(seed)
    n_train, n_test = 600, 200

    def _make(n: int, start_id: int) -> pd.DataFrame:
        age = rng.integers(18, 70, size=n)
        income = rng.normal(50_000, 15_000, size=n).round(2)
        tenure = rng.integers(0, 15, size=n)
        # honest signal -> probability -> label
        logit = 0.03 * (age - 40) + 0.00002 * (income - 50_000) - 0.05 * tenure
        prob = 1.0 / (1.0 + np.exp(-logit))
        target = (rng.random(n) < prob).astype(int)
        # LEAK 1: feature = target + tiny noise
        leaky_score = target + rng.normal(0, 0.01, size=n)
        # LEAK 2: id-like column (monotonic + unique)
        row_id = np.arange(start_id, start_id + n)
        return pd.DataFrame(
            {
                "row_id": row_id,
                "age": age,
                "income": income,
                "tenure": tenure,
                "leaky_score": leaky_score,
                "target": target,
            }
        )

    train_df = _make(n_train, start_id=1000)
    test_df = _make(n_test, start_id=1000 + n_train)

    # LEAK 3: copy some train rows straight into test (contamination)
    dupes = train_df.iloc[:20].copy()
    test_df = pd.concat([test_df, dupes], ignore_index=True)

    return train_df, test_df, "target"


def demo_clean_data(seed: int = 42):
    """Return (train_df, test_df, target_name) with NO injected leaks.

    Honest features, a random stratified-ish split, no id column, no dupes.
    """
    rng = np.random.default_rng(seed)
    n = 800

    age = rng.integers(18, 70, size=n)
    income = rng.normal(50_000, 15_000, size=n).round(2)
    tenure = rng.integers(0, 15, size=n)
    logit = 0.03 * (age - 40) + 0.00002 * (income - 50_000) - 0.05 * tenure
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {"age": age, "income": income, "tenure": tenure, "target": target}
    )
    # clean random split
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(0.75 * n)
    train_df = shuffled.iloc[:cut].reset_index(drop=True)
    test_df = shuffled.iloc[cut:].reset_index(drop=True)
    return train_df, test_df, "target"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_report(title: str, result: Dict) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    s = result["summary"]
    print(f"VERDICT: {result['verdict'].upper()}  "
          f"(high={s['n_high']}, medium={s['n_medium']}, low={s['n_low']})")
    if not result["findings"]:
        print("  No findings.")
    for f in result["findings"]:
        feat = f["feature"] if f["feature"] is not None else "-"
        print(f"  [{f['severity'].upper():6}] {f['check']:24} {feat:14} {f['detail']}")


if __name__ == "__main__":
    tr, te, tgt = demo_leaky_data()
    leaky_result = run_all(tr, te, tgt)
    _print_report("LEAKY DEMO", leaky_result)

    ctr, cte, ctgt = demo_clean_data()
    clean_result = run_all(ctr, cte, ctgt)
    _print_report("CLEAN DEMO", clean_result)
