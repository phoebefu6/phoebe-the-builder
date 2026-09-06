"""Assertions for every claim the README makes.

The published group-sequential constants are checked exactly (they are a
numerical result, not a simulation); everything measured on simulated traffic is
checked with a tolerance wide enough for the Monte Carlo error at that sample
size and narrow enough to fail if the claim is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
from sequential import (
    Trial,
    bonferroni_bounds,
    crossing_probability,
    first_crossing,
    msprt_crossing,
    msprt_statistic,
    naive_bounds,
    obf_bounds,
    pocock_bounds,
    score,
    score_with_stop,
    simulate,
    with_futility,
)

ALPHA = 0.05
P0 = 0.10
P1 = 0.11
N_MAX = 20_000
M = 40_000
SEED = 1234

PUB_POCOCK = {2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413, 10: 2.555, 20: 2.672}
PUB_OBF_FINAL = {2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040, 10: 2.087}
PUB_OBF5 = [4.562, 3.226, 2.634, 2.281, 2.040]


def looks_for(k: int, n_max: int = N_MAX) -> np.ndarray:
    return np.linspace(n_max / k, n_max, k).astype(np.int64)


# --------------------------------------------------------------------------
# The recursion itself
# --------------------------------------------------------------------------


def test_single_look_recovers_the_fixed_horizon_alpha():
    spent, _ = crossing_probability([1.959964], step=0.0025)
    assert spent == pytest.approx(0.05, abs=1e-4)


def test_crossing_probability_is_monotone_in_the_boundary():
    tight, _ = crossing_probability([2.0, 2.0, 2.0], step=0.005)
    loose, _ = crossing_probability([1.8, 1.8, 1.8], step=0.005)
    assert loose > tight


@pytest.mark.parametrize("k", sorted(PUB_POCOCK))
def test_pocock_matches_published_table(k):
    solved = pocock_bounds(k, ALPHA, step=0.0025)[0]
    assert solved == pytest.approx(PUB_POCOCK[k], abs=0.002)


@pytest.mark.parametrize("k", sorted(PUB_OBF_FINAL))
def test_obf_final_bound_matches_published_table(k):
    solved = obf_bounds(k, ALPHA, step=0.0025)[-1]
    assert solved == pytest.approx(PUB_OBF_FINAL[k], abs=0.002)


def test_obf_five_look_boundary_matches_published_shape():
    solved = obf_bounds(5, ALPHA, step=0.0025)
    assert np.allclose(solved, PUB_OBF5, atol=0.004)


def test_solved_boundaries_spend_exactly_alpha():
    for builder in (pocock_bounds, obf_bounds):
        spent, each = crossing_probability(builder(5, ALPHA, step=0.0025), step=0.0025)
        assert spent == pytest.approx(ALPHA, abs=2e-3)
        assert sum(each) == pytest.approx(spent, abs=1e-12)


def test_obf_spends_almost_nothing_at_the_first_look():
    _, each = crossing_probability(obf_bounds(5, ALPHA, step=0.0025), step=0.0025)
    assert each[0] < 0.001
    assert each[-1] > 10 * each[0]


def test_pocock_boundary_is_flat_and_obf_is_decreasing():
    p = pocock_bounds(5, ALPHA, step=0.005)
    o = obf_bounds(5, ALPHA, step=0.005)
    assert np.allclose(p, p[0])
    assert np.all(np.diff(o) < 0)


def test_bonferroni_is_stricter_than_pocock_at_every_k():
    for k in (2, 5, 10, 20, 50):
        assert bonferroni_bounds(k, ALPHA)[0] > pocock_bounds(k, ALPHA, step=0.005)[0]


# --------------------------------------------------------------------------
# Section 2: the measured cost of an uncorrected peek
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def null_20():
    return simulate(looks_for(20), P0, P0, M, SEED)


@pytest.fixture(scope="module")
def alt_20():
    return simulate(looks_for(20), P0, P1, M, SEED + 1)


def test_one_look_is_calibrated():
    t = simulate(looks_for(1), P0, P0, M, SEED + 2)
    fpr = float((first_crossing(t.z, naive_bounds(1, ALPHA)) >= 0).mean())
    assert fpr == pytest.approx(ALPHA, abs=0.004)


def test_naive_false_positive_rate_grows_with_the_number_of_looks():
    rates = []
    for k in (1, 5, 20):
        t = simulate(looks_for(k), P0, P0, M, SEED + 10 + k)
        rates.append(float((first_crossing(t.z, naive_bounds(k, ALPHA)) >= 0).mean()))
    assert rates[0] < rates[1] < rates[2]
    assert rates[1] > 0.12  # five looks: roughly triple nominal
    assert rates[2] > 0.22  # twenty looks: roughly five times nominal


def test_a_daily_peek_is_about_five_times_nominal(null_20):
    fpr = float((first_crossing(null_20.z, naive_bounds(20, ALPHA)) >= 0).mean())
    assert 4.0 * ALPHA < fpr < 6.0 * ALPHA


def test_continuous_monitoring_keeps_climbing_with_more_traffic():
    looks = np.arange(500, 100_001, 500, dtype=np.int64)
    t = simulate(looks, P0, P0, 8_000, SEED + 3)
    ever = np.maximum.accumulate(np.abs(t.z) >= 1.959964, axis=1)
    early = float(ever[:, int(20_000 / 500) - 1].mean())
    late = float(ever[:, -1].mean())
    assert late > early > ALPHA
    assert late > 0.35


# --------------------------------------------------------------------------
# Section 3: the corrections
# --------------------------------------------------------------------------


def test_pocock_and_obf_are_calibrated_on_bernoulli_traffic(null_20):
    for bounds in (pocock_bounds(20, ALPHA, step=0.005), obf_bounds(20, ALPHA, step=0.005)):
        fpr = float((first_crossing(null_20.z, bounds) >= 0).mean())
        assert fpr == pytest.approx(ALPHA, abs=0.006)


def test_bonferroni_undershoots_its_alpha(null_20):
    fpr = float((first_crossing(null_20.z, bonferroni_bounds(20, ALPHA)) >= 0).mean())
    assert fpr < 0.5 * ALPHA


def test_naive_peeking_buys_power_with_the_same_defect_that_inflates_its_fpr(alt_20):
    fixed = Trial(alt_20.looks[-1:], alt_20.z[:, -1:], alt_20.diff[:, -1:], alt_20.se[:, -1:], P0, P1)
    pw_fixed = float((first_crossing(fixed.z, naive_bounds(1, ALPHA)) >= 0).mean())
    pw_naive = float((first_crossing(alt_20.z, naive_bounds(20, ALPHA)) >= 0).mean())
    assert pw_naive > pw_fixed  # this is why people do it
    assert pw_fixed == pytest.approx(0.90, abs=0.02)


def test_sequential_designs_use_less_traffic_than_the_fixed_horizon(alt_20):
    o_p = score(alt_20, first_crossing(alt_20.z, pocock_bounds(20, ALPHA, step=0.005)), "pocock")
    o_o = score(alt_20, first_crossing(alt_20.z, obf_bounds(20, ALPHA, step=0.005)), "obf")
    assert o_p.expected_n < o_o.expected_n < N_MAX
    assert o_p.expected_n < 0.7 * N_MAX


def test_obf_keeps_more_power_than_pocock_and_pocock_stops_sooner(alt_20):
    o_p = score(alt_20, first_crossing(alt_20.z, pocock_bounds(20, ALPHA, step=0.005)), "pocock")
    o_o = score(alt_20, first_crossing(alt_20.z, obf_bounds(20, ALPHA, step=0.005)), "obf")
    assert o_o.reject_rate > o_p.reject_rate
    assert o_p.expected_n < o_o.expected_n


# --------------------------------------------------------------------------
# Section 4: the estimate at the stopping look
# --------------------------------------------------------------------------


def test_the_fixed_horizon_estimate_is_nearly_unbiased(alt_20):
    fixed = Trial(alt_20.looks[-1:], alt_20.z[:, -1:], alt_20.diff[:, -1:], alt_20.se[:, -1:], P0, P1)
    o = score(fixed, first_crossing(fixed.z, naive_bounds(1, ALPHA)), "fixed")
    assert abs(o.est_bias) < 0.10


def test_every_early_stopping_rule_overstates_the_effect(alt_20):
    for name, bounds in (
        ("naive", naive_bounds(20, ALPHA)),
        ("pocock", pocock_bounds(20, ALPHA, step=0.005)),
        ("obf", obf_bounds(20, ALPHA, step=0.005)),
    ):
        o = score(alt_20, first_crossing(alt_20.z, bounds), name)
        assert o.est_bias > 0.10, name


def test_obf_overstates_less_than_pocock_although_both_are_valid(alt_20):
    o_p = score(alt_20, first_crossing(alt_20.z, pocock_bounds(20, ALPHA, step=0.005)), "pocock")
    o_o = score(alt_20, first_crossing(alt_20.z, obf_bounds(20, ALPHA, step=0.005)), "obf")
    assert o_o.est_bias < o_p.est_bias


def test_the_naive_interval_undercovers_at_the_stopping_look(alt_20):
    o = score(alt_20, first_crossing(alt_20.z, pocock_bounds(20, ALPHA, step=0.005)), "pocock")
    assert o.ci_coverage < 0.94


def test_the_overstatement_grows_as_the_effect_gets_smaller():
    biases = []
    for rel in (0.20, 0.05, 0.02):
        t = simulate(looks_for(20), P0, P0 * (1 + rel), M, SEED + 40 + int(rel * 100))
        biases.append(score(t, first_crossing(t.z, pocock_bounds(20, ALPHA, step=0.005)), "p").est_bias)
    assert biases[0] < biases[1] < biases[2]
    assert biases[-1] > 3.0  # a 2% true lift is reported several times too large


# --------------------------------------------------------------------------
# Section 5: schedules, and mSPRT
# --------------------------------------------------------------------------


def test_msprt_is_a_martingale_style_bound_at_every_schedule():
    for k, sims in ((20, M), (40, M), (200, 8_000)):
        t = simulate(looks_for(k), P0, P0, sims, SEED + 60 + k)
        fpr = float((msprt_crossing(t, P1 - P0, ALPHA) >= 0).mean())
        assert fpr <= ALPHA, (k, fpr)


def test_msprt_statistic_starts_small_and_is_finite():
    t = simulate(looks_for(20), P0, P1, 200, SEED + 7)
    lam = msprt_statistic(t, P1 - P0)
    assert np.all(np.isfinite(lam))
    assert lam[:, 0].mean() < 1.0 / ALPHA


def test_a_pocock_constant_reused_on_a_denser_schedule_leaks_alpha():
    c = float(pocock_bounds(20, ALPHA, step=0.005)[0])
    t = simulate(looks_for(40), P0, P0, M, SEED + 8)
    fpr = float((first_crossing(t.z, np.full(40, c)) >= 0).mean())
    assert fpr > ALPHA + 0.006


def test_the_obf_shape_reindexed_by_information_fraction_does_not():
    c = float(obf_bounds(20, ALPHA, step=0.005)[-1])
    bounds = c / np.sqrt(np.arange(1, 41) / 40)
    t = simulate(looks_for(40), P0, P0, M, SEED + 9)
    fpr = float((first_crossing(t.z, bounds) >= 0).mean())
    assert fpr == pytest.approx(ALPHA, abs=0.007)


def test_msprt_pays_for_its_generality_in_power(alt_20):
    o_obf = score(alt_20, first_crossing(alt_20.z, obf_bounds(20, ALPHA, step=0.005)), "obf")
    pw_ms = float((msprt_crossing(alt_20, P1 - P0, ALPHA) >= 0).mean())
    assert pw_ms < o_obf.reject_rate - 0.15


def test_a_badly_chosen_tau_costs_most_of_the_power(alt_20):
    pw = {tau: float((msprt_crossing(alt_20, tau, ALPHA) >= 0).mean()) for tau in (0.002, 0.010)}
    assert pw[0.002] < 0.2 < pw[0.010]


# --------------------------------------------------------------------------
# Section 6: futility
# --------------------------------------------------------------------------


def _futility_bounds(k: int = 20) -> np.ndarray:
    fut = np.full(k, -np.inf)
    fut[k // 2 - 1 :] = 0.0
    return fut


def test_futility_cannot_add_a_false_positive(null_20):
    bounds = obf_bounds(20, ALPHA, step=0.005)
    plain = float((first_crossing(null_20.z, bounds) >= 0).mean())
    r, s = with_futility(null_20.z, bounds, _futility_bounds(), signed=True)
    assert float((r >= 0).mean()) <= plain + 1e-12


def test_futility_returns_most_of_an_empty_experiment(null_20):
    bounds = obf_bounds(20, ALPHA, step=0.005)
    r0, s0 = with_futility(null_20.z, bounds, None, signed=True)
    r1, s1 = with_futility(null_20.z, bounds, _futility_bounds(), signed=True)
    o0 = score_with_stop(null_20, r0, s0, "plain")
    o1 = score_with_stop(null_20, r1, s1, "futility")
    assert o1.expected_n < 0.8 * o0.expected_n


def test_futility_costs_almost_no_power(alt_20):
    bounds = obf_bounds(20, ALPHA, step=0.005)
    r0, s0 = with_futility(alt_20.z, bounds, None, signed=True)
    r1, s1 = with_futility(alt_20.z, bounds, _futility_bounds(), signed=True)
    p0 = score_with_stop(alt_20, r0, s0, "plain").reject_rate
    p1 = score_with_stop(alt_20, r1, s1, "futility").reject_rate
    assert p0 - p1 < 0.01


# --------------------------------------------------------------------------
# Section 7: the improvised correction
# --------------------------------------------------------------------------


def test_bonferroni_loses_real_power_and_the_gap_widens_with_k():
    lost = []
    for k in (2, 20):
        t = simulate(looks_for(k), P0, P1, M, SEED + 80 + k)
        pw_p = float((first_crossing(t.z, pocock_bounds(k, ALPHA, step=0.005)) >= 0).mean())
        pw_b = float((first_crossing(t.z, bonferroni_bounds(k, ALPHA)) >= 0).mean())
        lost.append(pw_p - pw_b)
    assert lost[0] < lost[1]
    assert lost[1] > 0.08


def test_peeking_less_often_is_a_weak_lever():
    t3 = simulate(looks_for(3), P0, P0, M, SEED + 90)
    fpr3 = float((first_crossing(t3.z, naive_bounds(3, ALPHA)) >= 0).mean())
    assert fpr3 > 1.5 * ALPHA  # weekly instead of daily is still not honest


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_simulation_increments_are_binomial_and_accumulate():
    t = simulate([100, 200, 400], P0, P0, 5_000, SEED + 99)
    assert t.z.shape == (5_000, 3)
    assert np.all(np.isfinite(t.z))
    assert t.se[:, 2].mean() < t.se[:, 0].mean()  # more traffic, tighter interval


def test_first_crossing_reports_the_earliest_crossing_only():
    stat = np.array([[0.0, 3.0, 5.0], [0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    idx = first_crossing(stat, np.full(3, 2.0))
    assert list(idx) == [1, -1, 0]
