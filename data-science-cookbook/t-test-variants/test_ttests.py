"""The claims this build makes, as assertions."""

from __future__ import annotations

import math

import numpy as np
import pytest
import ttests as tt
from scipy import stats


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


# ---------------------------------------------------------------- the statistics are right
def test_matches_scipy_to_machine_precision() -> None:
    for name, worst in tt.scipy_agreement().items():
        assert worst < 1e-12, f"{name} disagrees with scipy by {worst}"


def test_paired_is_exactly_one_sample_on_differences(rng: np.random.Generator) -> None:
    x, y = rng.standard_normal(30), rng.standard_normal(30) * 2.3
    a, b = tt.paired_t(x, y), tt.one_sample_t(x - y)
    assert a.p == pytest.approx(b.p, abs=1e-15)
    assert a.t == pytest.approx(b.t, abs=1e-12)
    assert a.df == b.df


def test_vectorised_path_agrees_with_scalar_path(rng: np.random.Generator) -> None:
    X, Y = rng.standard_normal((50, 12)), rng.standard_normal((50, 7)) * 2
    sv, wv = tt.student_p_vec(X, Y), tt.welch_p_vec(X, Y)
    for i in range(50):
        assert sv[i] == pytest.approx(tt.student_t(X[i], Y[i]).p, abs=1e-12)
        assert wv[i] == pytest.approx(tt.welch_t(X[i], Y[i]).p, abs=1e-12)


def test_welch_df_lies_between_min_group_df_and_pooled_df(rng: np.random.Generator) -> None:
    """Satterthwaite's df is bounded by min(n1,n2)-1 and n1+n2-2. A formula slip breaks this."""
    for _ in range(200):
        n1, n2 = int(rng.integers(3, 60)), int(rng.integers(3, 60))
        x = rng.standard_normal(n1) * rng.uniform(0.2, 5)
        y = rng.standard_normal(n2) * rng.uniform(0.2, 5)
        df = tt.welch_t(x, y).df
        assert min(n1, n2) - 1 - 1e-9 <= df <= n1 + n2 - 2 + 1e-9


def test_welch_equals_student_when_variances_and_sizes_match(rng: np.random.Generator) -> None:
    """With n1 == n2 the two statistics are algebraically identical; only the df differs."""
    x, y = rng.standard_normal(25), rng.standard_normal(25)
    assert tt.welch_t(x, y).t == pytest.approx(tt.student_t(x, y).t, rel=1e-12)


# ---------------------------------------------------------------- degenerate input
def test_zero_variance_does_not_raise_and_does_not_claim_significance() -> None:
    z = np.zeros(10)
    for res in (tt.student_t(z, z), tt.welch_t(z, z), tt.one_sample_t(z)):
        assert res.p == 1.0
        assert math.isnan(res.t)
        assert res.note


def test_too_few_observations_is_reported_not_crashed() -> None:
    res = tt.welch_t(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    assert res.p == 1.0 and "2 observations" in res.note


def test_paired_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        tt.paired_t(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------- the draws are standardised
@pytest.mark.parametrize("dist", tt.DISTRIBUTIONS)
def test_every_distribution_is_drawn_with_the_requested_mean_and_sd(dist: str) -> None:
    v = tt.draw(np.random.default_rng(4), 4000, 400, sd=2.0, dist=dist, shift=1.0)
    assert v.mean() == pytest.approx(1.0, abs=0.08)
    assert v.std() == pytest.approx(2.0, rel=0.08)


def test_unknown_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown distribution"):
        tt.draw(np.random.default_rng(0), 10, 10, 1.0, "gaussian")


# ---------------------------------------------------------------- the harness can pass
def test_known_truth_cell_recovers_exactly_nominal() -> None:
    """Student's on balanced equal-variance normal data is EXACT at 0.05.

    If the harness cannot recover a rate it is mathematically guaranteed to produce, every
    other number it reports is unreadable. This is the null the detector has to pass.
    """
    rate = tt.simulate(tt.Cell(n1=20, n2=20), reps=40_000, seed=1)["student"]
    lo, hi = tt.wilson(rate * 40_000, 40_000)
    assert lo <= tt.ALPHA <= hi, f"harness measured {rate:.4f} on an exactly-0.05 cell"


# ---------------------------------------------------------------- the findings
def test_student_inflates_when_the_small_group_has_the_large_variance() -> None:
    rate = tt.simulate(tt.Cell(n1=50, n2=10, sd2=3.0), reps=20_000, seed=2)["student"]
    assert rate > 0.20, rate


def test_student_collapses_when_the_large_group_has_the_large_variance() -> None:
    rate = tt.simulate(tt.Cell(n1=10, n2=50, sd2=3.0), reps=20_000, seed=2)["student"]
    assert rate < 0.005, rate


def test_a_variance_gap_alone_is_not_the_problem() -> None:
    """The negative result: balanced designs survive a 3x variance ratio."""
    rate = tt.simulate(tt.Cell(n1=20, n2=20, sd2=3.0), reps=40_000, seed=3)["student"]
    assert rate < 0.06, rate


def test_welch_holds_nominal_across_the_whole_normal_grid() -> None:
    for k, (n1, n2) in enumerate([(20, 20), (30, 10), (10, 30), (50, 10), (10, 50)]):
        for j, r in enumerate((1.0, 2.0, 3.0)):
            rate = tt.simulate(tt.Cell(n1=n1, n2=n2, sd2=r), reps=20_000, seed=500 + 10 * k + j)["welch"]
            assert 0.040 < rate < 0.062, f"n={n1}/{n2} ratio={r}: {rate}"


def test_the_normal_curve_shortcut_is_inflated_everywhere() -> None:
    for k, (n1, n2) in enumerate([(20, 20), (30, 10), (50, 10), (10, 50)]):
        rate = tt.simulate(tt.Cell(n1=n1, n2=n2), reps=20_000, seed=700 + k)["z-shortcut"]
        assert rate > tt.ALPHA, f"n={n1}/{n2}: {rate}"


def test_welch_premium_is_negligible_on_balanced_designs() -> None:
    r = tt.simulate(tt.Cell(n1=50, n2=50, shift=0.8), reps=20_000, seed=9)
    assert abs(r["student"] - r["welch"]) < 0.005


def test_welch_gives_up_real_power_on_unbalanced_equal_variance_designs() -> None:
    """The honest cost of the default, which the README does not hide."""
    r = tt.simulate(tt.Cell(n1=50, n2=10, shift=0.8), reps=20_000, seed=10)
    assert r["student"] - r["welch"] > 0.02


def test_under_skew_and_imbalance_the_ranking_reverses() -> None:
    r = tt.simulate(tt.Cell(n1=50, n2=10, dist="lognormal"), reps=40_000, seed=11)
    assert r["welch"] > 0.08, r
    assert r["student"] < 0.06, r
    assert r["welch"] > r["student"]


def test_the_clt_repairs_the_skew_case_only_slowly() -> None:
    small = tt.simulate(tt.Cell(n1=50, n2=10, dist="lognormal"), reps=20_000, seed=12)["welch"]
    big = tt.simulate(tt.Cell(n1=2500, n2=500, dist="lognormal"), reps=20_000, seed=13)["welch"]
    assert small > big > 0.05


# ---------------------------------------------------------------- the interval
def test_wilson_brackets_the_point_estimate_and_narrows_with_n() -> None:
    for n in (1_000, 40_000):
        lo, hi = tt.wilson(0.05 * n, n)
        assert lo < 0.05 < hi
    w_small = np.diff(tt.wilson(0.05 * 1_000, 1_000))[0]
    w_big = np.diff(tt.wilson(0.05 * 40_000, 40_000))[0]
    assert w_big < w_small / 5


def test_wilson_stays_inside_zero_one_at_the_boundary() -> None:
    lo, hi = tt.wilson(0, 500)
    assert 0.0 <= lo < 1e-12 and 0 < hi < 1
    assert tt.wilson(500, 500)[1] == 1.0


def test_wilson_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        tt.wilson(1, 0)


def test_verdict_names_the_direction() -> None:
    assert tt.verdict(0.29, 40_000).startswith("INFLATED")
    assert tt.verdict(0.001, 40_000).startswith("conservative")
    assert tt.verdict(0.0502, 40_000) == "nominal"


# ---------------------------------------------------------------- cell metadata
def test_balance_label_tracks_which_group_carries_the_variance() -> None:
    assert tt.Cell(20, 20).balance == "balanced"
    assert tt.Cell(50, 10, sd1=1.0, sd2=3.0).balance == "small-n has large-sd"
    assert tt.Cell(10, 50, sd1=1.0, sd2=3.0).balance == "large-n has large-sd"


def test_scipy_is_still_the_reference_for_the_null_distribution() -> None:
    """Sanity: our p at t=0 is 1, and at the 97.5th percentile of t(df) it is 0.05."""
    df = 18
    crit = stats.t.isf(0.025, df)
    assert 2 * stats.t.sf(crit, df) == pytest.approx(0.05, abs=1e-12)
