"""Tests for the arithmetic. The closed forms are checked against their definitions and
against simulation; the worlds are checked for the property that makes them worth having.
"""

from __future__ import annotations

import novelty as nv
import numpy as np
import pytest

TAU0, TAU_INF, LAM, T = 0.10, 0.02, 7.0, 28


# ----------------------------------------------------------------- the exact construction
def test_cohort_profile_mean_equals_tau0():
    """The whole point of section 5: a world with no decay whose average effect is tau0."""
    m = nv.cohort_profile(T, TAU0, TAU_INF, LAM)
    assert abs(m.mean() - TAU0) < 1e-12


def test_cohort_profile_prefix_means_reproduce_the_decay_curve():
    m = nv.cohort_profile(T, TAU0, TAU_INF, LAM)
    for t in range(T):
        prefix = m[: T - t]           # cohorts old enough to appear at tenure t
        target = nv.tau_decay(t, TAU0, TAU_INF, LAM)
        assert abs(prefix.mean() - target) < 1e-12


def test_cohort_profile_has_no_tenure_dependence_by_construction():
    """m is indexed by cohort only - there is no tenure argument to depend on."""
    m = nv.cohort_profile(T, TAU0, TAU_INF, LAM)
    assert m.shape == (T,)


# ----------------------------------------------------------------- attenuation
@pytest.mark.parametrize("mult", [1, 2, 4, 8, 12])
def test_uniform_enrollment_weights_novelty_up(mult):
    Tw = int(mult * LAM)
    assert nv.attenuation(Tw, LAM, "uniform") > nv.attenuation(Tw, LAM, "bigbang")


def test_attenuation_goes_to_one_for_a_slow_decay():
    for enroll in ("bigbang", "uniform"):
        assert nv.attenuation(28, 1e6, enroll) == pytest.approx(1.0, abs=1e-4)


def test_attenuation_definition_matches_a_direct_weighted_mean():
    t = np.arange(T)
    r = np.exp(-t / LAM)
    assert nv.attenuation(T, LAM, "bigbang") == pytest.approx(r.mean())
    w = (T - t).astype(float)
    assert nv.attenuation(T, LAM, "uniform") == pytest.approx((w * r).sum() / w.sum())


def test_continuous_form_is_within_ten_percent_of_the_daily_sum():
    for enroll in ("bigbang", "uniform"):
        c = nv.attenuation_continuous(4 * LAM, LAM, enroll)
        d = nv.attenuation(int(4 * LAM), LAM, enroll)
        assert abs(c / d - 1) < 0.10


def test_reported_lift_matches_the_attenuation_form():
    rng = np.random.default_rng(7)
    got = []
    for _ in range(40):
        d = nv.panel_decay(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
        got.append(nv.pooled_lift(d)[0])
    pred = TAU_INF + (TAU0 - TAU_INF) * nv.attenuation(T, LAM, "uniform")
    assert abs(np.mean(got) - pred) < 0.003


# ----------------------------------------------------------------- the holdout arithmetic
def test_mde_ratio_is_one_at_a_balanced_split():
    assert nv.mde_ratio(0.5) == pytest.approx(1.0)


def test_mde_ratio_closed_form():
    assert nv.mde_ratio(0.05) == pytest.approx(1 / (2 * np.sqrt(0.05 * 0.95)))


def test_mde_ratio_matches_the_simulated_mde():
    assert nv.mde(100000, 0.05) / nv.mde(100000, 0.5) == pytest.approx(nv.mde_ratio(0.05))


def test_same_day_launches_are_unidentified_not_merely_imprecise():
    assert nv.holdout_attribution(np.full(6, 60), 240, 0.01) is None


def test_spaced_launches_are_identified():
    a = nv.holdout_attribution(30 + np.arange(6) * 30, 240, 0.01)
    assert a is not None and np.all(np.isfinite(a["se_per_launch"]))


def test_attribution_se_rises_monotonically_as_spacing_shrinks():
    ses = [float(np.mean(nv.holdout_attribution(30 + np.arange(6) * g, 240, 0.01)["se_per_launch"]))
           for g in (30, 14, 7, 3, 1)]
    assert ses == sorted(ses)


# ----------------------------------------------------------------- the worlds
def test_decay_world_recovers_its_own_tenure_curve():
    rng = np.random.default_rng(11)
    acc = np.zeros(T)
    reps = 60
    for _ in range(reps):
        d = nv.panel_decay(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
        c = nv.lift_by_tenure(d, min_arm=2)
        acc[c["tenure"].astype(int)] += c["lift"]
    acc /= reps
    for t in (0, 3, 7, 14):
        assert abs(acc[t] - nv.tau_decay(t, TAU0, TAU_INF, LAM)) < 0.012


def test_the_two_worlds_are_separated_by_the_within_cohort_slope_and_not_by_the_guard():
    rng = np.random.default_rng(13)
    d = nv.panel_decay(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    m = nv.panel_mixshift(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    # both aggregate curves decline, so the guard fires on both
    assert nv.guard_early_vs_late(d)["diff"] > 0
    assert nv.guard_early_vs_late(m)["diff"] > 0
    # only the decay world bends its own cohorts
    assert nv.within_cohort_slope(d)["z"] < -2.0
    assert abs(nv.within_cohort_slope(m)["z"]) < 2.5


def test_mixshift_long_run_truth_is_five_times_the_decay_worlds():
    rng = np.random.default_rng(17)
    m = nv.panel_mixshift(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    d = nv.panel_decay(rng, n_users=8400, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    assert m["truth_long_run"] / d["truth_long_run"] == pytest.approx(5.0, abs=1e-9)


def test_survivor_world_lift_grows_with_tenure_on_a_constant_effect():
    rng = np.random.default_rng(19)
    d = nv.panel_survivor(rng, n_users=8400, T=T, tau=0.05)
    c = nv.lift_by_tenure(d)
    early = c["lift"][c["tenure"] < 5].mean()
    late = c["lift"][c["tenure"] > 20].mean()
    assert late > early + 0.1
    assert d["truth_long_run"] == 0.05


def test_guard_is_calibrated_on_a_flat_effect():
    rng = np.random.default_rng(23)
    fires = 0
    reps = 200
    for _ in range(reps):
        d = nv.panel_decay(rng, n_users=4200, T=T, tau0=0.05, tau_inf=0.05, lam=LAM)
        fires += int(nv.guard_early_vs_late(d)["p"] < 0.05)
    assert 0.01 < fires / reps < 0.11


def test_share_weighted_identity_closes():
    rng = np.random.default_rng(29)
    d = nv.panel_decay(rng, n_users=4200, T=T, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    r = nv.share_weighted_identity(d)
    assert abs(r["gap"]) < 1e-12          # exact per-arm algebra
    assert abs(r["naive_gap"]) < 0.01     # single-share rollup: close, not exact


def test_fit_recovers_the_asymptote_when_the_window_is_long_enough():
    rng = np.random.default_rng(31)
    d = nv.panel_decay(rng, n_users=8400, T=84, tau0=TAU0, tau_inf=TAU_INF, lam=LAM)
    f = nv.fit_exponential(nv.lift_by_tenure(d, min_arm=2), lam0=LAM)
    assert f is not None
    assert abs(f["tau_inf"] - TAU_INF) < 0.02


def test_pooled_lift_is_unbiased_on_a_zero_effect_world():
    rng = np.random.default_rng(37)
    got = [nv.pooled_lift(nv.panel_decay(rng, n_users=4200, T=T, tau0=0.0, tau_inf=0.0))[0]
           for _ in range(80)]
    assert abs(np.mean(got)) < 0.004
