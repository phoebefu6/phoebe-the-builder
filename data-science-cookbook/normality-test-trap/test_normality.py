"""The measured error rates are the whole product, so the statistics underneath them are checked
against scipy, and the samplers are checked against their closed-form moments.

A measured rejection rate is worthless if the statistic producing it is wrong, and a
false-positive study is worthless if the null it claims to simulate is not actually true.
Both are tested here.
"""

from __future__ import annotations

import normality as N
import numpy as np
import pytest
from scipy import stats

# --------------------------------------------------------------------------- samplers


@pytest.mark.parametrize("dist", N.DISTRIBUTIONS)
def test_sampler_is_standardised(dist: str) -> None:
    """Population mean 0 and sd 1 - otherwise the null under test is not the null claimed."""
    rng = np.random.default_rng(7)
    x = N.draw(dist, 400, 6000, rng).ravel()
    # Heavy-tailed populations need a loose tolerance on the sd; the mean is the load-bearing one
    # because it IS the null hypothesis.
    assert abs(x.mean()) < 0.01, f"{dist} is not centred: mean {x.mean():.4f}"
    assert abs(x.std() - 1.0) < 0.05, f"{dist} is not unit-scale: sd {x.std():.4f}"


@pytest.mark.parametrize("dist", ["normal", "uniform", "lognormal-mild", "exponential", "contaminated"])
def test_sampler_matches_declared_shape(dist: str) -> None:
    """Sample skew and excess kurtosis land on the analytic values in SHAPE.

    Restricted to the populations whose sample moments converge at a reachable sample size:
    t5 has infinite theoretical kurtosis sensitivity and lognormal-strong's fourth moment
    needs far more data than a test should draw, so their SHAPE entries are documentation
    rather than assertions.
    """
    rng = np.random.default_rng(11)
    x = N.draw(dist, 500, 40_000, rng).ravel()
    skew, kurt = N.SHAPE[dist]
    assert stats.skew(x) == pytest.approx(skew, abs=0.15)
    assert stats.kurtosis(x) == pytest.approx(kurt, rel=0.25, abs=0.2)


def test_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError):
        N.draw("gaussian-ish", 10, 10, np.random.default_rng(0))


# --------------------------------------------------------------------------- statistics


def test_one_sample_t_matches_scipy() -> None:
    """The vectorised t is the same number scipy computes, to machine precision."""
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((200, 37))
    t, p = N.one_sample_t_p(samples)
    for i in range(samples.shape[0]):
        ref = stats.ttest_1samp(samples[i], 0.0)
        assert t[i] == pytest.approx(ref.statistic, rel=1e-12, abs=1e-12)
        assert p[i] == pytest.approx(ref.pvalue, rel=1e-12, abs=1e-12)


def test_one_sample_t_handles_zero_variance() -> None:
    """A constant sample has no t statistic. It must not raise, and it must not reject."""
    samples = np.zeros((3, 8))
    t, p = N.one_sample_t_p(samples)
    assert np.all(np.isfinite(t))
    assert np.all(p >= 0.05), "a sample with no variance must never count as a rejection"


def test_shapiro_reject_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    samples = rng.standard_normal((50, 40))
    got = N.shapiro_reject(samples, alpha=0.05)
    want = np.array([stats.shapiro(r).pvalue < 0.05 for r in samples])
    assert np.array_equal(got, want)


def test_wilcoxon_survives_an_all_zero_sample() -> None:
    """scipy raises on an all-zero input; the harness must return the conservative p=1.0."""
    samples = np.zeros((2, 10))
    p = N.wilcoxon_p(samples)
    assert np.all(p == 1.0)


# --------------------------------------------------------------------------- calibration


def test_normal_cell_is_calibrated() -> None:
    """The known-truth anchor. A one-sample t on normal data is exact, so it must read 0.05.

    If this drifts, nothing else in the study can be trusted - the harness would be measuring
    its own bug. Tolerance is set from noise_floor(), not guessed.
    """
    c = N.run_cell("normal", 50, 20_000, seed=42)
    assert c.t_error == pytest.approx(0.05, abs=0.005)
    assert c.shapiro_reject_rate == pytest.approx(0.05, abs=0.01)


def test_noise_floor_is_narrower_than_the_broken_band() -> None:
    """The band used to call a cell broken has to be resolvable by this harness."""
    lo, hi = N.noise_floor(n=50, runs=6)
    assert hi - lo < 0.010, f"noise {hi - lo:.4f} is too wide for a +/-0.005 band"


def test_shapiro_power_increases_with_n() -> None:
    """The central mechanism: same population, more data, more rejections."""
    small = N.run_cell("lognormal-mild", 20, 4_000, seed=1, include_wilcoxon=False)
    large = N.run_cell("lognormal-mild", 500, 4_000, seed=2, include_wilcoxon=False)
    assert small.shapiro_reject_rate < 0.95
    assert large.shapiro_reject_rate > 0.99
    assert large.shapiro_reject_rate > small.shapiro_reject_rate


def test_clt_repairs_the_t_test_where_shapiro_does_not_relent() -> None:
    """The other half: on the same population the t-test gets BETTER as Shapiro gets surer."""
    small = N.run_cell("exponential", 20, 8_000, seed=3, include_wilcoxon=False)
    large = N.run_cell("exponential", 1000, 8_000, seed=4, include_wilcoxon=False)
    assert large.shapiro_reject_rate >= small.shapiro_reject_rate
    assert abs(large.t_error - 0.05) < abs(small.t_error - 0.05)


# --------------------------------------------------------------------------- scoring


def _cell(dist: str, n: int, shapiro: float, t_err: float, reps: int = 20_000) -> N.CellResult:
    """A synthetic cell. `reps` is load-bearing - the broken flag is an interval, not a compare."""
    return N.CellResult(
        dist=dist,
        n=n,
        reps=reps,
        alpha=0.05,
        shapiro_reject_rate=shapiro,
        t_error=t_err,
        t_error_lower_tail=t_err / 2,
        t_error_upper_tail=t_err / 2,
        wilcoxon_error=0.05,
        conditional_error=t_err,
        t_error_given_shapiro_passed=t_err,
        t_error_given_shapiro_rejected=t_err,
    )


def test_diagnostic_score_counts_the_four_corners() -> None:
    cells = [
        _cell("a", 10, 0.9, 0.09),  # fires, broken      -> true positive
        _cell("b", 10, 0.9, 0.050),  # fires, fine       -> false alarm
        _cell("c", 10, 0.1, 0.09),  # quiet, broken      -> miss
        _cell("d", 10, 0.1, 0.050),  # quiet, fine       -> true negative
    ]
    s = N.score_as_diagnostic(cells)
    assert s.cells == 4 and s.broken_cells == 2
    assert s.sensitivity == pytest.approx(0.5)
    assert s.specificity == pytest.approx(0.5)
    assert s.false_alarm_cells == 1 and s.missed_cells == 1


def test_score_can_report_a_perfect_gate() -> None:
    """A scorer that cannot return 1.00 would make every bad score meaningless.

    This is the sibling builds' standing rule: a bench must be able to report the good outcome
    as well as the bad one, or it is not measuring anything.
    """
    cells = [_cell("a", 10, 0.9, 0.09), _cell("d", 10, 0.1, 0.050)]
    s = N.score_as_diagnostic(cells)
    assert s.sensitivity == pytest.approx(1.0)
    assert s.specificity == pytest.approx(1.0)


def test_tail_asymmetry_and_broken_flag() -> None:
    c = _cell("a", 10, 0.5, 0.05)
    assert c.tail_asymmetry == pytest.approx(1.0)
    assert not c.t_is_broken
    skewed = N.CellResult("a", 10, 20_000, 0.05, 0.5, 0.05, 0.045, 0.005, 0.05, 0.05, 0.05, 0.05)
    assert skewed.tail_asymmetry == pytest.approx(9.0)
    assert not skewed.t_is_broken, "two-sided inside the band while the tails are 9:1 apart"


def test_crossover_and_blind_spot_readers() -> None:
    cells = [
        _cell("x", 20, 0.30, 0.070),  # quiet, broken -> blind spot
        _cell("x", 200, 0.99, 0.060),  # fires, broken
        _cell("x", 1000, 1.00, 0.050),  # fires, fine  -> crossover
    ]
    assert N.crossover_n(cells, "x") == 1000
    assert N.blind_spot_n(cells, "x") == [20]
    assert N.crossover_n(cells, "nope") is None


def test_wilson_interval_brackets_the_rate_and_narrows_with_reps() -> None:
    for reps in (500, 5_000, 50_000):
        lo, hi = N.wilson(0.05, reps)
        assert lo < 0.05 < hi
    assert (N.wilson(0.05, 50_000)[1] - N.wilson(0.05, 50_000)[0]) < (
        N.wilson(0.05, 500)[1] - N.wilson(0.05, 500)[0]
    )
    assert N.wilson(0.0, 1_000)[0] == 0.0, "an interval must not run below zero"


def test_a_noisy_cell_is_never_called_broken() -> None:
    """The regression guard for the bug this build shipped and then fixed.

    An earlier version compared the point estimate straight to [0.045, 0.055] and flagged the
    EXACT calibration cell - a one-sample t on normal data - as broken at n=5000, purely because
    that tier ran 1,500 replicates. A detector that fires on its own known-good cell is measuring
    its own noise.
    """
    noisy = _cell("normal", 5000, 1.0, 0.0433, reps=1_500)
    assert not noisy.t_is_broken, "a point estimate inside Monte-Carlo noise is not a finding"
    assert not noisy.resolvable, "and the cell should admit it cannot resolve the band"

    # 0.0433 is genuinely marginal: even at 40,000 replicates its interval still touches 0.045,
    # so the honest verdict there is still "not shown to be broken". A rate that IS clear of the
    # band, measured at the same replicate count, must flag - otherwise the rule flags nothing.
    marginal = _cell("normal", 5000, 1.0, 0.0433, reps=40_000)
    assert not marginal.t_is_broken and marginal.resolvable

    honest = _cell("skewed", 5000, 1.0, 0.0400, reps=40_000)
    assert honest.t_is_broken, "a rate clear of the band at 40,000 reps IS a finding"
    assert honest.resolvable


def test_every_grid_tier_can_resolve_the_band() -> None:
    """If a tier cannot report 'broken', its 'not broken' cells are absence of evidence only."""
    for n in N.GRID_N:
        c = _cell("normal", n, 0.5, 0.05, reps=N.reps_for(n))
        assert c.resolvable, f"n={n} runs {N.reps_for(n)} reps, too few to judge the band"
