"""Assertions behind every claim in the README.

These are not smoke tests.  Each one pins a number the README quotes, or a
closed form the README derives, so that a change to the engine breaks the
document rather than silently rewriting it.
"""

from __future__ import annotations

import interference as I
import numpy as np
import pytest

PC, PT = 0.10, 0.13


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- mechanics
def test_rationing_never_serves_more_than_supply():
    r = rng(1)
    for supply in (0, 1, 50, 500):
        z = (r.random(2000) < 0.5).astype(int)
        y = I.rationed_outcomes(z, 0.5, 0.9, supply, r)
        assert y.sum() <= supply


def test_rationing_serves_every_attempt_when_supply_is_slack():
    r = rng(2)
    z = np.zeros(1000, dtype=int)
    y = I.rationed_outcomes(z, 0.2, 0.2, 10**9, r)
    # every attempt served, so the mean is the attempt rate
    assert abs(y.mean() - 0.2) < 0.04


def test_local_supply_is_respected_per_group():
    r = rng(3)
    group = np.repeat(np.arange(10), 200)
    y = I.rationed_outcomes(np.ones(2000, dtype=int), PC, 0.9, 0, r, group=group, supply_per_group=7)
    for g in range(10):
        assert y[group == g].sum() <= 7


def test_within_group_split_treats_exactly_half_of_each_group():
    r = rng(4)
    group = np.repeat(np.arange(25), 8)
    z = I.assign_within_group(group, r)
    for g in range(25):
        assert z[group == g].sum() == 4


def test_cluster_assignment_is_constant_within_a_group():
    r = rng(5)
    group = np.repeat(np.arange(30), 10)
    z = I.assign_by_group(group, r)
    for g in range(30):
        assert len(set(z[group == g].tolist())) == 1
    assert z.sum() == 15 * 10


# ------------------------------------------------- market closed forms
def test_global_effect_matches_its_closed_form():
    """min(p_t, S/n) - min(p_c, S/n): the whole market is capped by supply."""
    r = rng(6)
    n = 20_000
    for supply in (4000, 2400, 2000, 1400):
        cf = min(PT, supply / n) - min(PC, supply / n)
        got = I.market_global_effect(n, PC, PT, supply, r, reps=120)
        assert abs(got - cf) < 0.0015, (supply, cf, got)


def test_split_estimate_matches_its_closed_form():
    """Both arms face the same rationing factor, so it multiplies the difference."""
    r = rng(7)
    n = 20_000
    for supply in (2000, 1600):
        cf = (PT - PC) * min(1.0, supply / (n * (PT + PC) / 2))
        got = np.mean([I.market_split_estimate(n, PC, PT, supply, r)[0] for _ in range(200)])
        assert abs(got - cf) < 0.0010, (supply, cf, got)


def test_a_saturated_market_has_exactly_zero_global_effect():
    r = rng(8)
    got = I.market_global_effect(20_000, PC, PT, 1500, r, reps=60)
    assert got == pytest.approx(0.0, abs=1e-12)


def test_split_is_significant_and_wrong_in_a_saturated_market():
    """The headline of the build: p < 0.05 every time, ~100% of it bias."""
    r = rng(9)
    n, supply = 20_000, 2_000
    est, ses = [], []
    for _ in range(120):
        e, s = I.market_split_estimate(n, PC, PT, supply, r)
        est.append(e)
        ses.append(s)
    est, ses = np.array(est), np.array(ses)
    truth = I.market_global_effect(n, PC, PT, supply, r, reps=120)
    assert np.mean(np.abs(est / ses) > 1.96) > 0.95
    assert est.mean() > 0.020
    assert truth < 0.002
    assert (est.mean() - truth) / est.mean() > 0.90


def test_bias_is_flat_in_n_while_the_se_falls():
    r = rng(10)
    biases, ses = [], []
    for n in (25_000, 200_000):
        supply = int(n * PC)
        truth = I.market_global_effect(n, PC, PT, supply, r, reps=40)
        e, s = [], []
        for _ in range(120):
            a, b = I.market_split_estimate(n, PC, PT, supply, r)
            e.append(a)
            s.append(b)
        biases.append(np.mean(e) - truth)
        ses.append(np.mean(s))
    assert abs(biases[1] - biases[0]) < 0.002       # bias unchanged
    assert ses[0] / ses[1] > 2.3                    # SE fell by ~sqrt(8)


def test_tightness_is_supply_per_expected_attempt():
    assert I.tightness(20_000, 0.10, 2_000) == pytest.approx(1.0)
    assert I.tightness(20_000, 0.10, 2_600) == pytest.approx(1.3)


# ---------------------------------------------- spillover closed forms
def test_spillover_split_recovers_the_direct_effect_only():
    r = rng(11)
    m, groups, tau, gamma = 20, 300, 1.0, 0.5
    group = np.repeat(np.arange(groups), m)
    est = []
    for _ in range(300):
        z = I.assign_within_group(group, r)
        est.append(I.user_estimate(I.spillover_outcomes(z, group, tau, gamma, 1.0, r), z)[0])
    assert np.mean(est) == pytest.approx(tau - gamma / (m - 1), abs=0.01)


def test_spillover_bias_closed_form_is_exact_and_has_no_n_in_it():
    r = rng(12)
    m, tau, gamma = 20, 1.0, 0.5
    truth = tau + gamma
    for groups in (100, 800):
        group = np.repeat(np.arange(groups), m)
        est = []
        for _ in range(300):
            z = I.assign_within_group(group, r)
            est.append(I.user_estimate(I.spillover_outcomes(z, group, tau, gamma, 1.0, r), z)[0])
        assert np.mean(est) - truth == pytest.approx(I.spillover_split_bias_closed_form(gamma, m), abs=0.02)


def test_the_two_worlds_have_opposite_signs_of_error():
    """Marketplace overstates, peer network understates - same estimator."""
    r = rng(13)
    market_est = np.mean([I.market_split_estimate(20_000, PC, PT, 2_000, r)[0] for _ in range(80)])
    market_truth = I.market_global_effect(20_000, PC, PT, 2_000, r, reps=80)
    m, groups, tau, gamma = 20, 200, 1.0, 0.5
    group = np.repeat(np.arange(groups), m)
    peer_est = np.mean([
        I.user_estimate(I.spillover_outcomes((z := I.assign_within_group(group, r)), group, tau, gamma, 1.0, r), z)[0]
        for _ in range(200)
    ])
    assert market_est - market_truth > 0
    assert peer_est - (tau + gamma) < 0


def test_cluster_randomisation_is_unbiased_when_interference_is_contained():
    r = rng(14)
    m, groups, tau, gamma = 20, 300, 1.0, 0.5
    group = np.repeat(np.arange(groups), m)
    est = []
    for _ in range(300):
        z = I.assign_by_group(group, r)
        est.append(I.cluster_estimate(I.spillover_outcomes(z, group, tau, gamma, 1.0, r), z, group)[0])
    assert np.mean(est) == pytest.approx(tau + gamma, abs=0.02)


def test_group_effect_cancels_in_a_within_group_split_and_not_in_a_cluster_test():
    r = rng(15)
    m, groups = 20, 300
    group = np.repeat(np.arange(groups), m)
    sp, cl = [], []
    for _ in range(400):
        z = I.assign_within_group(group, r)
        sp.append(I.user_estimate(I.spillover_outcomes(z, group, 1.0, 0.0, 1.0, r, group_sd=0.6), z)[0])
        z2 = I.assign_by_group(group, r)
        cl.append(I.cluster_estimate(I.spillover_outcomes(z2, group, 1.0, 0.0, 1.0, r, group_sd=0.6), z2, group)[0])
    de = np.var(cl) / np.var(sp)
    derived = 1 + m * 0.6**2 / 1.0**2
    textbook = 1 + (m - 1) * (0.36 / 1.36)
    assert de == pytest.approx(derived, rel=0.20)
    assert de > textbook * 1.10          # the textbook number really is too small


# ------------------------------------------------------ the guard test
def test_dose_response_check_is_calibrated_under_no_interference():
    r = rng(16)
    p = np.array([I.dose_response_check(20_000, PC, PT, 10**9, r)["p"] for _ in range(400)])
    assert 0.02 < np.mean(p < 0.05) < 0.09


def test_dose_response_check_is_nearly_blind_at_experiment_scale():
    r = rng(17)
    p = np.array([I.dose_response_check(20_000, PC, PT, 2_000, r)["p"] for _ in range(300)])
    assert np.mean(p < 0.05) < 0.15      # 100% bias, and it barely notices


def test_dose_response_check_does_eventually_work():
    r = rng(18)
    p = np.array([I.dose_response_check(1_000_000, PC, PT, 100_000, r)["p"] for _ in range(60)])
    assert np.mean(p < 0.05) > 0.45      # it is a valid test, just an expensive one


# -------------------------------------------------------- switchback
def test_switchback_attenuates_by_exactly_the_carryover():
    r = rng(19)
    for c in (0.0, 0.10, 0.30):
        e = [I.switchback_run(400, 1.0, c, 1.0, r, alternating=False)[0] for _ in range(400)]
        assert np.mean(e) == pytest.approx(1.0 - c, abs=0.02)


def test_strict_alternation_doubles_the_carryover_bias():
    r = rng(20)
    for c in (0.10, 0.20):
        e = [I.switchback_run(400, 1.0, c, 1.0, r, alternating=True)[0] for _ in range(400)]
        assert np.mean(e) == pytest.approx(1.0 - 2 * c, abs=0.02)


def test_burn_in_at_least_the_carryover_removes_the_bias():
    r = rng(21)
    e = [I.switchback_run(400, 1.0, 0.2, 1.0, r, burn_in=0.2)[0] for _ in range(400)]
    assert np.mean(e) == pytest.approx(1.0, abs=0.02)


def test_burn_in_beyond_the_carryover_only_costs_variance():
    r = rng(22)
    v = []
    for b in (0.20, 0.60):
        e = np.array([I.switchback_run(400, 1.0, 0.2, 1.0, r, burn_in=b)[0] for _ in range(500)])
        assert e.mean() == pytest.approx(1.0, abs=0.03)
        v.append(e.var())
    assert v[1] > v[0] * 1.5


def test_switchback_closed_form_matches_the_simulation():
    r = rng(23)
    for alt in (False, True):
        for c, b in ((0.25, 0.0), (0.25, 0.10), (0.10, 0.30)):
            pred = I.switchback_bias_closed_form(1.0, c, alt, burn_in=b)
            got = np.mean([I.switchback_run(600, 1.0, c, 1.0, r, alternating=alt, burn_in=b)[0] for _ in range(300)])
            assert got == pytest.approx(pred, abs=0.03), (alt, c, b, pred, got)


def test_burn_in_of_one_is_rejected():
    r = rng(24)
    with pytest.raises(ValueError):
        I.switchback_run(100, 1.0, 0.1, 1.0, r, burn_in=1.0)
