"""Engine tests. Each checker is also shown able to FAIL (a verifier must be tested against passing)."""

from __future__ import annotations

import bayes as B
import numpy as np
import pytest
from scipy import stats


def test_calibration_is_clean() -> None:
    cal = B.calibrate()
    assert cal["max_log_gap_closed_vs_quadrature"] < 1e-8
    assert cal["max_log_gap_jzs_g_vs_delta"] < 1e-6
    assert cal["scipy_decision_mismatches"] == 0
    if np.isfinite(cal["max_log_gap_jzs_vs_pingouin"]):
        assert cal["max_log_gap_jzs_vs_pingouin"] < 1e-6


def test_quadrature_check_can_fail() -> None:
    """The closed form must NOT match the quadrature run at a different prior width."""
    assert abs(np.log(B.bf01_normal(1.96, 1000, 1.0)) - np.log(B.bf01_normal_by_quadrature(1.96, 1000, 0.5))) > 0.1


def test_jzs_check_can_fail() -> None:
    assert abs(np.log(B.bf10_jzs(2.1, 40, 0.707)) - np.log(B.bf10_jzs_by_delta(2.1, 40, 1.0))) > 0.05


def test_null_size_is_alpha_at_every_n() -> None:
    for n in B.NS:
        assert B.verdicts(n, 0.0)["p_significant"] == pytest.approx(B.ALPHA, abs=1e-12)


def test_lindley_bf01_grows_like_sqrt_n() -> None:
    a, b = B.bf01_at_p(0.05, 10 ** 5), B.bf01_at_p(0.05, 10 ** 6)
    assert b / a == pytest.approx(np.sqrt(10), rel=1e-3)


def test_flip_n_is_the_rising_crossing_and_is_n_not_log_n() -> None:
    """Two defects the first run shipped: the descending root, and returning log n."""
    n1 = B.n_where_p_supports_null(0.05, 1.0)
    assert 30 < n1 < 100
    assert B.bf01_at_p(0.05, n1) == pytest.approx(1.0, rel=1e-6)
    assert B.bf01_at_p(0.05, 0.9 * n1) < 1 < B.bf01_at_p(0.05, 1.1 * n1)


def test_ceiling_matches_search_and_sits_under_sbb() -> None:
    mx = B.max_bf10_over_normal_priors(B.Z_CRIT)
    assert mx["closed_form"] == pytest.approx(mx["by_search"], rel=1e-8)
    assert mx["closed_form"] < B.sbb_bound(0.05)
    for tau in (0.01, 0.05, 0.3, 1, 3):
        assert 1 / B.bf01_normal(B.Z_CRIT, 1, tau) <= mx["closed_form"] + 1e-12


def test_verdict_probabilities_are_consistent() -> None:
    for n in (10, 300, 10 ** 5):
        for d in (0.0, 0.02, 0.2):
            v = B.verdicts(n, d)
            assert v["sig_and_bf01_over_3"] <= v["sig_and_bf01_over_1"] <= v["p_significant"] + 1e-15
            assert v["bf10_over_3"] + v["bf01_over_3"] <= 1 + 1e-12


def test_monte_carlo_agrees() -> None:
    rows = B.monte_carlo(4000, seed=3)
    misses = sum(not r[k]["inside_99"] for r in rows for k in ("p_significant", "bf01_over_3", "sig_and_bf01_over_1"))
    assert misses <= 1


def test_monte_carlo_can_fail() -> None:
    """Replicates drawn at the wrong effect must land outside the exact interval."""
    rng = np.random.default_rng(0)
    z = rng.normal(0.2, 1, (4000, 200)).mean(axis=1) * np.sqrt(200)
    k = int((np.abs(z) > B.Z_CRIT).sum())
    lo, hi = B.wilson(k, 4000)
    assert not lo <= B.verdicts(200, 0.0)["p_significant"] <= hi


def test_n_for_power_is_the_smallest_such_n() -> None:
    n = B.n_for_power(0.2, "p_significant")
    assert B.verdicts(n, 0.2)["p_significant"] >= 0.8 > B.verdicts(n - 1, 0.2)["p_significant"]


def test_analyse_matches_scipy_and_guards_input() -> None:
    x = np.random.default_rng(1).normal(0.3, 1, 25)
    a = B.analyse(x)
    assert a["p"] == pytest.approx(stats.ttest_1samp(x, 0).pvalue, rel=1e-10)
    for bad in ([1, 2], [2, 2, 2, 2], [1, float("nan"), 3]):
        with pytest.raises(ValueError):
            B.analyse(bad)
