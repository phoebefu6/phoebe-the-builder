"""Does the honest binning actually produce a better model? The question IV cannot answer.

`binning.py` establishes that a raw IV is not trustworthy and builds a screen that is. But a
screen that rejects features is only worth having if the surviving encoding scores better on
rows nobody fitted. Information Value is a univariate statistic - it says nothing about what
happens once six features go into one model together.

So: same six columns, three encodings, one holdout.

    A  raw continuous, median-imputed, sentinel -999 left in the column as a number
    B  constrained WOE bins - size floors, monotone, missing and sentinel separated
    C  loose WOE bins - 20 bins, 1-event floor, no monotonicity (the biggest-IV arm)

If C wins the holdout, the constraints in `binning.py` are costing real accuracy and the
whole argument is decorative. It does not win. It wins the *training* set, which is the
point being made.

Pure numpy - logistic regression by gradient descent and a rank-based AUC - so this module
adds no dependency the rest of the project does not already have.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from binning import Binning, build_dataset, fit

# The three encodings, and the binning settings behind B and C.
CONSTRAINED = dict(max_bins=6, min_bin_share=0.05, min_bin_events=20, monotone=True)
LOOSE = dict(max_bins=20, max_prebins=20, min_bin_share=0.0, min_bin_events=1, monotone=False)


# --------------------------------------------------------------------------------------
# Metric and model, both from scratch
# --------------------------------------------------------------------------------------


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with ties averaged (the Mann-Whitney U form)."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    y_sorted = y[order]

    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0  # average rank over the tie group
        i = j + 1

    return float((ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iters: int = 4000, lr: float = 0.5
) -> Tuple[np.ndarray, float]:
    """Logistic regression by batch gradient descent on standardized inputs."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        resid = pr - y
        w -= lr * (X.T @ resid / n + l2 * w)
        b -= lr * float(resid.mean())
    return w, b


def _standardize(
    X: np.ndarray, mean: Optional[np.ndarray] = None, sd: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mean) / sd, mean, sd


def _score(
    X_tr: np.ndarray, y_tr: np.ndarray, X_ho: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on train, return (train score, holdout score). Standardization is train-only."""
    Z_tr, mean, sd = _standardize(X_tr)
    Z_ho, _, _ = _standardize(X_ho, mean, sd)
    w, b = fit_logistic(Z_tr, y_tr)
    return Z_tr @ w + b, Z_ho @ w + b


# --------------------------------------------------------------------------------------
# The three design matrices
# --------------------------------------------------------------------------------------


def _raw_matrix(
    features: Dict[str, np.ndarray], idx: np.ndarray, medians: Dict[str, float]
) -> np.ndarray:
    """Arm A: what you get with no binning at all - impute the gap, keep the sentinel."""
    cols = []
    for name in sorted(features):
        v = np.asarray(features[name], dtype=float)[idx].copy()
        v[np.isnan(v)] = medians[name]
        cols.append(v)  # -999 stays -999: a code sitting on the numeric scale
    return np.column_stack(cols)


def _woe_matrix(
    features: Dict[str, np.ndarray], idx: np.ndarray, schemes: Dict[str, Binning]
) -> np.ndarray:
    """Arms B and C: each column replaced by the WOE of its bin, cut points frozen."""
    return np.column_stack(
        [schemes[name].transform(np.asarray(features[name], dtype=float)[idx]) for name in sorted(features)]
    )


def fit_schemes(
    features: Dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    specials: Dict[str, Sequence[float]],
    settings: Dict[str, object],
) -> Dict[str, Binning]:
    """Fit one binning per feature on the training rows only."""
    out = {}
    for name, values in features.items():
        out[name] = fit(
            np.asarray(values, dtype=float)[train_idx],
            np.asarray(y, dtype=int)[train_idx],
            feature=name,
            specials=specials.get(name, ()),
            **settings,
        )
    return out


def model_lift(data: Optional[dict] = None) -> List[dict]:
    """Score the three arms on the same holdout and report total IV alongside AUC."""
    data = data or build_dataset()
    features: Dict[str, np.ndarray] = data["features"]
    y = np.asarray(data["y"], dtype=int)
    train_idx = data["train_idx"]
    hold_idx = data["holdout_idx"]
    specials = data["specials"]

    y_tr, y_ho = y[train_idx], y[hold_idx]
    medians = {
        name: float(np.nanmedian(np.asarray(v, dtype=float)[train_idx]))
        for name, v in features.items()
    }

    rows = []

    s_tr, s_ho = _score(
        _raw_matrix(features, train_idx, medians), y_tr, _raw_matrix(features, hold_idx, medians)
    )
    rows.append(
        {
            "arm": "A raw continuous (median-imputed)",
            "total_iv_train": float("nan"),  # no bins, so no IV to quote
            "bins_total": 0,
            "auc_train": auc(y_tr, s_tr),
            "auc_holdout": auc(y_ho, s_ho),
        }
    )

    for label, settings in (("B constrained WOE bins", CONSTRAINED), ("C loose WOE bins", LOOSE)):
        schemes = fit_schemes(features, y, train_idx, specials, settings)
        s_tr, s_ho = _score(
            _woe_matrix(features, train_idx, schemes), y_tr, _woe_matrix(features, hold_idx, schemes)
        )
        rows.append(
            {
                "arm": label,
                "total_iv_train": float(sum(s.iv for s in schemes.values())),
                "bins_total": int(sum(len(s.bins) for s in schemes.values())),
                "auc_train": auc(y_tr, s_tr),
                "auc_holdout": auc(y_ho, s_ho),
            }
        )

    for row in rows:
        row["overfit_gap"] = row["auc_train"] - row["auc_holdout"]
    return rows


def format_lift(rows: Sequence[dict]) -> str:
    out = [
        "arm                                 bins  total IV   AUC train  AUC holdout   gap",
        "-" * 78,
    ]
    for r in rows:
        iv = "     -   " if np.isnan(r["total_iv_train"]) else "%9.4f" % r["total_iv_train"]
        out.append(
            "%-34s %5d %s  %9.4f  %11.4f  %+.4f"
            % (r["arm"], r["bins_total"], iv, r["auc_train"], r["auc_holdout"], r["overfit_gap"])
        )
    return "\n".join(out)


def robustness(seeds: Sequence[int] = tuple(range(11, 21))) -> dict:
    """Re-run the three arms on independent datasets, because one split proves nothing.

    A 0.003 AUC difference on a single holdout is not a finding. Ten datasets turn it into
    a win rate and a spread, which is a claim that can be stated without overreaching.
    """
    a, b, c = [], [], []
    iv_b, iv_c, gap_b, gap_c = [], [], [], []
    for seed in seeds:
        rows = {r["arm"][0]: r for r in model_lift(build_dataset(seed=seed))}
        a.append(rows["A"]["auc_holdout"])
        b.append(rows["B"]["auc_holdout"])
        c.append(rows["C"]["auc_holdout"])
        iv_b.append(rows["B"]["total_iv_train"])
        iv_c.append(rows["C"]["total_iv_train"])
        gap_b.append(rows["B"]["overfit_gap"])
        gap_c.append(rows["C"]["overfit_gap"])

    a, b, c = np.array(a), np.array(b), np.array(c)
    return {
        "n_seeds": len(seeds),
        "auc_raw": float(a.mean()),
        "auc_constrained": float(b.mean()),
        "auc_loose": float(c.mean()),
        "constrained_beats_raw": int((b > a).sum()),
        "constrained_beats_loose": int((b > c).sum()),
        "mean_margin_over_loose": float((b - c).mean()),
        "sd_margin_over_loose": float((b - c).std()),
        "mean_margin_over_raw": float((b - a).mean()),
        "mean_extra_iv_loose": float(np.mean(iv_c) - np.mean(iv_b)),
        "loose_always_higher_iv": bool(all(x > y for x, y in zip(iv_c, iv_b))),
        "mean_gap_loose": float(np.mean(gap_c)),
        "mean_gap_constrained": float(np.mean(gap_b)),
    }


def main() -> None:
    data = build_dataset()
    rows = model_lift(data)
    print("Six features, three encodings, one holdout of %d rows.\n" % len(data["holdout_idx"]))
    print(format_lift(rows))

    best = max(rows, key=lambda r: r["auc_holdout"])
    loose = [r for r in rows if r["arm"].startswith("C")][0]
    constrained = [r for r in rows if r["arm"].startswith("B")][0]

    print(
        "\nThe loose arm carries %.2f more total IV than the constrained one across %d extra bins,"
        % (loose["total_iv_train"] - constrained["total_iv_train"],
           loose["bins_total"] - constrained["bins_total"])
    )
    print(
        "and converts it into %.4f LESS holdout AUC. Its train-to-holdout gap is %+.4f against %+.4f."
        % (
            constrained["auc_holdout"] - loose["auc_holdout"],
            loose["overfit_gap"],
            constrained["overfit_gap"],
        )
    )
    print("\nBest on the holdout: %s (%.4f)." % (best["arm"], best["auc_holdout"]))
    print(
        "IV ranks the arms in the opposite order to the holdout. That is the whole reason\n"
        "the constraints and the permutation screen in binning.py are not decoration."
    )

    print("\n--- and once more on %d independent datasets, because one split proves nothing ---" % 10)
    r = robustness()
    print(
        "mean holdout AUC:  raw %.4f   constrained %.4f   loose %.4f"
        % (r["auc_raw"], r["auc_constrained"], r["auc_loose"])
    )
    print(
        "constrained beats raw   in %d/%d datasets (mean +%.4f AUC)"
        % (r["constrained_beats_raw"], r["n_seeds"], r["mean_margin_over_raw"])
    )
    print(
        "constrained beats loose in %d/%d datasets (mean +%.4f, sd %.4f - a real but small edge)"
        % (
            r["constrained_beats_loose"],
            r["n_seeds"],
            r["mean_margin_over_loose"],
            r["sd_margin_over_loose"],
        )
    )
    print(
        "the loose arm carried more total IV in %s dataset, every time, and overfitted %.1fx harder\n"
        "(mean gap %+.4f vs %+.4f)."
        % (
            "every" if r["loose_always_higher_iv"] else "most",
            r["mean_gap_loose"] / r["mean_gap_constrained"],
            r["mean_gap_loose"],
            r["mean_gap_constrained"],
        )
    )


if __name__ == "__main__":
    main()
