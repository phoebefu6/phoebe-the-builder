"""Tests for downstream.py - the metric, the model, and the claims made from them."""

from __future__ import annotations

import numpy as np
from binning import build_dataset, fit
from downstream import (
    CONSTRAINED,
    LOOSE,
    auc,
    fit_logistic,
    fit_schemes,
    model_lift,
    robustness,
)

DATA = build_dataset()
LIFT = {r["arm"][0]: r for r in model_lift(DATA)}

checks = 0
failed = 0


def check(label: str, condition: bool) -> None:
    global checks, failed
    checks += 1
    if condition:
        print("  PASS  %s" % label)
    else:
        failed += 1
        print("  FAIL  %s" % label)


# --------------------------------------------------------------------------------------
print("1. AUC from scratch agrees with the definition")

rng = np.random.default_rng(0)
y = (rng.random(400) < 0.3).astype(int)
s = rng.normal(0, 1, 400) + y * 0.8

# Brute-force Mann-Whitney: fraction of (positive, negative) pairs ranked correctly.
pos, neg = s[y == 1], s[y == 0]
wins = sum(float((p > neg).sum() + 0.5 * (p == neg).sum()) for p in pos)
brute = wins / (len(pos) * len(neg))
check("matches a brute-force pair count", abs(auc(y, s) - brute) < 1e-12)
check("a perfect score gives 1.0", auc(np.array([0, 0, 1, 1]), np.array([1.0, 2.0, 3.0, 4.0])) == 1.0)
check("a reversed score gives 0.0", auc(np.array([0, 0, 1, 1]), np.array([4.0, 3.0, 2.0, 1.0])) == 0.0)
check("all-ties gives 0.5", auc(np.array([0, 1, 0, 1]), np.array([2.0, 2.0, 2.0, 2.0])) == 0.5)
check("one class gives nan", np.isnan(auc(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]))))

# --------------------------------------------------------------------------------------
print("\n2. Logistic regression recovers a known signal")

n = 4000
X = rng.normal(0, 1, (n, 2))
true_w = np.array([1.5, -0.8])
p = 1.0 / (1.0 + np.exp(-(X @ true_w - 0.4)))
yb = (rng.random(n) < p).astype(int)
w, b = fit_logistic(X, yb)
check("first coefficient sign and rough size", 1.0 < w[0] < 2.0)
check("second coefficient sign and rough size", -1.3 < w[1] < -0.4)
check("intercept is negative as planted", b < 0)
check("separates in-sample", auc(yb, X @ w + b) > 0.8)

# --------------------------------------------------------------------------------------
print("\n3. The loose arm buys IV and bins, not accuracy")

check("loose fits far more bins", LIFT["C"]["bins_total"] > 3 * LIFT["B"]["bins_total"])
check("loose carries more total IV", LIFT["C"]["total_iv_train"] > LIFT["B"]["total_iv_train"])
check("loose wins the training set", LIFT["C"]["auc_train"] > LIFT["B"]["auc_train"])
check("loose overfits harder", LIFT["C"]["overfit_gap"] > LIFT["B"]["overfit_gap"])
check("raw arm quotes no IV", np.isnan(LIFT["A"]["total_iv_train"]))
check("every AUC is a real number", all(np.isfinite(r["auc_holdout"]) for r in LIFT.values()))

# --------------------------------------------------------------------------------------
print("\n4. Cut points are frozen, so the holdout is untouched by fitting")

schemes = fit_schemes(DATA["features"], DATA["y"], DATA["train_idx"], DATA["specials"], CONSTRAINED)
loose_schemes = fit_schemes(DATA["features"], DATA["y"], DATA["train_idx"], DATA["specials"], LOOSE)
direct = fit(
    np.asarray(DATA["features"]["utilization"])[DATA["train_idx"]],
    np.asarray(DATA["y"])[DATA["train_idx"]],
    feature="utilization",
    **CONSTRAINED,
)
check("fit_schemes matches a direct fit", schemes["utilization"].cuts == direct.cuts)
check("constrained honours the monotone requirement", schemes["utilization"].is_monotone() is not None)
check("loose is allowed to wobble", len(loose_schemes["noise"].bins) > len(schemes["noise"].bins))
check(
    "the sentinel stays off the numeric scale",
    -999.0 in schemes["n_inquiries"].specials,
)

hold = np.asarray(DATA["features"]["months_employed"], dtype=float)[DATA["holdout_idx"]]
encoded = schemes["months_employed"].transform(hold)
check("holdout rows all encode to a finite WOE", np.isfinite(encoded).all())
check("missing holdout rows encode too", np.isfinite(encoded[np.isnan(hold)]).all())

# --------------------------------------------------------------------------------------
print("\n5. The claims hold across independent datasets, not just one split")

r = robustness()
check("binning beats raw continuous every time", r["constrained_beats_raw"] == r["n_seeds"])
check("and by a margin worth having", r["mean_margin_over_raw"] > 0.01)
check(
    "constrained beats loose in most datasets, not all",
    r["n_seeds"] * 0.6 <= r["constrained_beats_loose"] < r["n_seeds"],
)
check("the margin over loose is small", 0 < r["mean_margin_over_loose"] < 0.02)
check("loose always carried more IV", r["loose_always_higher_iv"])
check("loose overfits at least twice as hard", r["mean_gap_loose"] > 2 * r["mean_gap_constrained"])

# --------------------------------------------------------------------------------------
print("\n6. Determinism")

check("model_lift is reproducible", model_lift(build_dataset())[1] == model_lift(build_dataset())[1])

print("\n%d/%d checks passed" % (checks - failed, checks))
raise SystemExit(1 if failed else 0)
