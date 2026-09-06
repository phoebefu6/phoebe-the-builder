"""Assertions for every number the README quotes, plus the invariants.

The multi-path statistics are re-measured here on a *different* set of seeds
than ``evidence.py`` uses, with tolerances wide enough to survive the sampling
error and narrow enough that a real regression fails. A claim that only holds
on one draw of the process is not a claim this build is allowed to make.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import targets as T
from evidence import ORIGIN_REF
from scipy import stats

N_TEST_PATHS = 200
TEST_SEED0 = 77_000


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return T.make_history()


@pytest.fixture(scope="module")
def mp():
    return T.multipath(N_TEST_PATHS, TEST_SEED0)


# --------------------------------------------------------------------------
# Invariants: the machinery has to be honest before the numbers mean anything
# --------------------------------------------------------------------------


def test_history_is_deterministic() -> None:
    assert np.array_equal(T.make_history(), T.make_history())
    assert len(T.make_history()) == T.N_MONTHS


def test_seasonal_index_has_mean_one() -> None:
    assert T.SEASONAL.mean() == pytest.approx(1.0, abs=1e-9)


def test_no_method_looks_forward(series: np.ndarray) -> None:
    """Changing the future must not change any target set before it."""
    origin = 100
    before = T.targets_at(series, origin)
    tampered = series.copy()
    tampered[origin:] *= 3.0
    after = T.targets_at(tampered, origin)
    for name in T.METHODS:
        assert before[name] == pytest.approx(after[name]), name


def test_every_method_is_documented() -> None:
    assert set(T.METHODS) == set(T.PROVENANCE)


def test_every_method_returns_a_positive_finite_number(series) -> None:
    for name, value in T.targets_at(series, ORIGIN_REF).items():
        assert np.isfinite(value) and value > 0, name


def test_backtest_scores_every_origin(series: np.ndarray) -> None:
    res = T.backtest(series)
    n = len(T.origins(series))
    assert n == 94
    for r in res.values():
        assert len(r.targets) == len(r.actuals) == n


def test_expected_level_exceeds_median_level() -> None:
    """The lognormal mean is above the median; section 3 depends on it."""
    for t in (0, 37, 120):
        assert T.expected_level(t) > T.median_level(t)
        ratio = T.expected_level(t) / T.median_level(t)
        assert ratio == pytest.approx(np.exp(T.SIGMA**2 / 2), rel=1e-12)


# --------------------------------------------------------------------------
# Section 1: the spread
# --------------------------------------------------------------------------


def test_twelve_methods_span_forty_percent(series: np.ndarray) -> None:
    tg = T.targets_at(series, ORIGIN_REF)
    assert len(tg) == 12
    lo, hi = min(tg.values()), max(tg.values())
    assert hi / lo == pytest.approx(1.400, abs=0.005)


def test_method_spread_exceeds_the_move_being_targeted(series) -> None:
    tg = T.targets_at(series, ORIGIN_REF)
    last_q = series[ORIGIN_REF - T.HORIZON : ORIGIN_REF].sum()
    actual = series[ORIGIN_REF : ORIGIN_REF + T.HORIZON].sum()
    spread = (max(tg.values()) - min(tg.values())) / last_q
    move = abs(actual / last_q - 1)
    assert spread == pytest.approx(0.311, abs=0.005)
    assert spread > move


# --------------------------------------------------------------------------
# Section 2: the hit rate is a property of the method
# --------------------------------------------------------------------------


def test_hit_rate_spans_almost_the_whole_range(mp) -> None:
    hits = {n: mp[n]["hit_rate"].mean() for n in T.METHODS}
    assert hits["top_down"] < 0.06
    assert hits["seasonal_naive"] > 0.90
    assert max(hits.values()) - min(hits.values()) > 0.85


def test_ambition_predicts_hit_rate(mp) -> None:
    names = list(T.METHODS)
    amb = [mp[n]["ambition"].mean() for n in names]
    hit = [mp[n]["hit_rate"].mean() for n in names]
    rho, p = stats.spearmanr(amb, hit)
    assert rho < -0.85
    assert p < 1e-3


def test_ambition_does_not_fully_determine_hit_rate(mp) -> None:
    """Some harder targets are met more often. A target is random too."""
    names = list(T.METHODS)
    amb = np.array([mp[n]["ambition"].mean() for n in names])
    hit = np.array([mp[n]["hit_rate"].mean() for n in names])
    inversions = sum(
        1
        for i, j in itertools.combinations(range(len(names)), 2)
        if (amb[i] - amb[j]) * (hit[i] - hit[j]) > 0
    )
    assert inversions >= 5


# --------------------------------------------------------------------------
# Section 3: an unbiased target is missed more often than it is hit
# --------------------------------------------------------------------------


def test_mean_target_is_missed_more_often_than_hit() -> None:
    oracle = T.oracle_hit_rates(100, 55_000)
    assert oracle["mean_target"] < 0.50
    assert oracle["median_target"] > 0.50
    assert 1 - stats.norm.cdf(T.SIGMA / 2) == pytest.approx(0.4761, abs=5e-4)


def test_a_sub_one_percent_change_moves_the_hit_rate_several_points(
    series, mp
) -> None:
    tg = T.targets_at(series, ORIGIN_REF)
    gap = tg["trend_seasonal"] / tg["trend_seasonal_median"] - 1
    assert 0.0 < gap < 0.01
    delta = (
        mp["trend_seasonal_median"]["hit_rate"].mean()
        - mp["trend_seasonal"]["hit_rate"].mean()
    )
    assert delta > 0.02


# --------------------------------------------------------------------------
# Section 4: a hit rate is not a reproducible measurement
# --------------------------------------------------------------------------


def test_best_specified_forecasts_have_the_least_stable_hit_rate(mp) -> None:
    sds = {n: mp[n]["hit_rate"].std() for n in T.METHODS}
    worst_two = sorted(sds, key=lambda n: -sds[n])[:2]
    assert set(worst_two) == {"trend_seasonal", "trend_seasonal_median"}
    third = sorted(sds.values(), reverse=True)[2]
    assert sds[worst_two[0]] > 0.06
    assert sds[worst_two[0]] > 1.15 * third


def test_it_takes_decades_to_grade_a_team_by_hit_rate() -> None:
    n = T.quarters_to_distinguish(0.50, 0.65)
    assert n == 134
    assert n / 4 > 30


def test_eight_of_twelve_is_not_evidence() -> None:
    p = stats.binomtest(8, 12, 0.5, alternative="greater").pvalue
    assert p > 0.05


# --------------------------------------------------------------------------
# Section 5: the answer depends on the month you ask in
# --------------------------------------------------------------------------


def test_run_rate_swings_by_month_of_origin(series: np.ndarray) -> None:
    ors = T.origins(series)
    by_month = {m: [] for m in range(12)}
    for o in ors:
        by_month[o % 12].append(
            T.m_run_rate(series[:o], o) / T.truth_quarter(o)[0]
        )
    means = {m: float(np.mean(v)) for m, v in by_month.items()}
    swing = max(means.values()) / min(means.values()) - 1
    assert swing > 0.30


def test_seasonal_method_swings_less_than_run_rate(series) -> None:
    def swing(fn) -> float:
        ors = T.origins(series)
        by_month = {m: [] for m in range(12)}
        for o in ors:
            by_month[o % 12].append(fn(series[:o], o) / T.truth_quarter(o)[0])
        means = [float(np.mean(v)) for v in by_month.values()]
        return max(means) / min(means) - 1

    assert swing(T.m_trend_seasonal) < swing(T.m_run_rate)


# --------------------------------------------------------------------------
# Section 6: the midpoint
# --------------------------------------------------------------------------


def test_midpoint_is_above_capacity_and_below_top_down(series) -> None:
    ors = T.origins(series)
    cap = np.array([T.m_capacity(series[:o], o) for o in ors])
    top = np.array([T.m_top_down(series[:o], o) for o in ors])
    spl = np.array([T.m_split_difference(series[:o], o) for o in ors])
    assert (spl > cap).mean() > 0.95
    assert (spl < top).mean() > 0.95


def test_midpoint_is_closer_to_the_forecast_and_met_far_less_often(
    series, mp
) -> None:
    """The two-sided accuracy statistic and the one-sided verdict disagree."""
    ors = T.origins(series)
    ref = np.array([T.m_trend_seasonal(series[:o], o) for o in ors])
    cap = np.array([T.m_capacity(series[:o], o) for o in ors])
    spl = np.array([T.m_split_difference(series[:o], o) for o in ors])
    assert np.abs(spl / ref - 1).mean() < np.abs(cap / ref - 1).mean()
    assert (cap / ref - 1).mean() < 0 < (spl / ref - 1).mean()
    assert (
        mp["capacity"]["hit_rate"].mean()
        > 3 * mp["split_difference"]["hit_rate"].mean()
    )


# --------------------------------------------------------------------------
# Sections 7-8: incentives
# --------------------------------------------------------------------------


def test_method_choice_is_worth_a_year_of_growth(mp) -> None:
    gap = (
        mp["trend_seasonal"]["ambition"].mean()
        / mp["seasonal_naive"]["ambition"].mean()
        - 1
    )
    months = np.log(1 + gap) / np.log(1 + T.G)
    assert months > 12
    delta_hit = (
        mp["seasonal_naive"]["hit_rate"].mean()
        - mp["trend_seasonal"]["hit_rate"].mean()
    )
    assert delta_hit > 0.40


def test_stretch_needs_a_payout_multiple_nobody_offers(mp) -> None:
    p_real = mp["trend_seasonal"]["hit_rate"].mean()
    p_str = mp["stretch_best_ever"]["hit_rate"].mean()
    p_top = mp["top_down"]["hit_rate"].mean()
    assert p_real / p_str > 1.3
    assert p_real / p_top > 8


# --------------------------------------------------------------------------
# Section 9: the interval
# --------------------------------------------------------------------------


def test_most_disagreements_are_inside_the_interval(series) -> None:
    lo, hi = T.prediction_interval(series, ORIGIN_REF, level=0.80)
    assert lo < hi
    width = hi - lo
    tg = T.targets_at(series, ORIGIN_REF)
    pairs = list(itertools.combinations(tg.values(), 2))
    inside = sum(1 for a, b in pairs if abs(a - b) < width)
    assert len(pairs) == 66
    assert inside / len(pairs) > 0.70


def test_interval_is_wide_relative_to_the_forecast(series) -> None:
    lo, hi = T.prediction_interval(series, ORIGIN_REF, level=0.80)
    assert (hi - lo) / ((hi + lo) / 2) > 0.12


# --------------------------------------------------------------------------
# Section 10: the ensemble
# --------------------------------------------------------------------------


def test_ensemble_is_accurate_and_still_a_bad_target(series, mp) -> None:
    hits, mapes, ambs = [], [], []
    for i in range(60):
        e = T.ensemble_result(T.make_history(seed=TEST_SEED0 + i))
        hits.append(e.hit_rate)
        mapes.append(e.mape)
        ambs.append(e.ambition)
    ens_hit, ens_mape = float(np.mean(hits)), float(np.mean(mapes))
    beaten = sum(1 for n in T.METHODS if mp[n]["mape"].mean() < ens_mape)
    met_more = sum(1 for n in T.METHODS if mp[n]["hit_rate"].mean() > ens_hit)
    assert beaten <= 4          # accuracy: near the top of the list
    assert met_more >= 7        # being met: near the bottom
    assert float(np.mean(ambs)) > 1.0


def test_accuracy_and_hit_rate_are_close_to_unrelated(mp) -> None:
    names = list(T.METHODS)
    mape = np.array([mp[n]["mape"].mean() for n in names])
    hit = np.array([mp[n]["hit_rate"].mean() for n in names])
    rho, _ = stats.spearmanr(-mape, hit)
    assert abs(rho) < 0.35
