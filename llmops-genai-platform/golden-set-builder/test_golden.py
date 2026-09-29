"""Engine tests. Each checker is also shown able to FAIL (a verifier must be tested against passing)."""

from __future__ import annotations

import golden as G
import numpy as np
import pytest
from scipy import stats


def test_mixes_are_distributions_and_new_intents_start_empty() -> None:
    for t in (0, 5, 12):
        assert G.mix(t).sum() == pytest.approx(1.0) and np.all(G.mix(t) >= 0)
    assert np.all(G.W0[G.EXISTING:] == 0) and np.all(G.W12[G.EXISTING:] > 0)


def test_allocations_spend_the_budget_and_skip_unknown_intents() -> None:
    for d in ("proportional", "sqrt", "equal"):
        n = G.allocate(d)
        assert n.sum() == G.M and np.all(n[:G.EXISTING] >= 1) and np.all(n[G.EXISTING:] == 0)
    with pytest.raises(ValueError):
        G.allocate("random")


def test_calibration_is_clean() -> None:
    cal = G.calibrate()
    assert cal["mcnemar_vs_binomtest_mismatches"] == 0
    for r in cal["moments"]:
        assert abs(r["bias_z"]) < 3.5 and abs(r["sd_ratio"] - 1) < 0.03


def test_calibration_can_fail() -> None:
    """The exact SD must NOT match a simulation run at a different budget."""
    n = G.allocate("proportional")
    k = np.random.default_rng(0).binomial(2 * n, G.Q, size=(4000, len(G.Q)))
    est = np.sum(n / n.sum() * np.where(n > 0, k / np.maximum(2 * n, 1), 0), axis=1)
    assert abs(est.std() / G.moments(n, 0, False)["sd"] - 1) > 0.2


def test_reweighting_is_unbiased_at_month_0_for_every_fixed_design() -> None:
    for d in ("proportional", "sqrt", "equal"):
        assert abs(G.moments(G.allocate(d), 0, True)["bias"]) < 1e-12


def test_reweighted_bias_at_month_12_is_exactly_the_coverage_gap() -> None:
    gap = G.coverage_gap(12)["bias_left"]
    for d in ("proportional", "sqrt", "equal"):
        assert G.moments(G.allocate(d), 12, True)["bias"] == pytest.approx(gap, abs=1e-12)


def test_bias_does_not_shrink_with_m() -> None:
    a, b = G.headline("proportional", 12, True, 240), G.headline("proportional", 12, True, 2400)
    assert a["bias"] == pytest.approx(b["bias"], abs=1e-12) and b["sd"] < a["sd"] / 3


def test_refresh_removes_the_gap() -> None:
    n = G.refresh(G.allocate("proportional"), 5)
    assert abs(G.moments(n, 12, True)["bias"]) < 1e-12


def test_noise_model_keeps_the_same_model_honest() -> None:
    """Day 186 defect: per-outcome flips lowered the re-run's pass rate; the same model flagged 58%."""
    f = G.regression_power(G.allocate("proportional"), flip=0, reps=4000)
    assert f["aggregate"] <= G.ALPHA + 0.01 and f["per_intent"] <= G.ALPHA


def test_regression_power_orders_by_cases_in_the_intent() -> None:
    p = [G.regression_power(G.allocate(d), reps=4000)["aggregate"] for d in ("proportional", "sqrt", "equal")]
    assert p[0] < p[1] < p[2]


def test_mcnemar_matches_binomtest() -> None:
    assert float(G.mcnemar_p(7, 1)) == pytest.approx(stats.binomtest(7, 8, 0.5, alternative="greater").pvalue)


def test_audit_numbers_and_guards() -> None:
    rows = [{"intent": "a", "cases": 90, "passes": 81, "traffic": 100},
            {"intent": "b", "cases": 10, "passes": 5, "traffic": 100},
            {"intent": "c", "cases": 0, "passes": 0, "traffic": 50}]
    a = G.audit(rows)
    assert a["raw"] == pytest.approx(0.86) and a["reweighted"] == pytest.approx(0.7)
    assert a["gaps"] == ["c"] and a["uncovered_share"] == pytest.approx(0.2)
    for bad in ([{**rows[0], "passes": 91}], rows + rows[:1], [{**rows[0], "cases": 0, "passes": 0}]):
        with pytest.raises(ValueError):
            G.audit(bad)
