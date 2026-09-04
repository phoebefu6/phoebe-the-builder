"""Assertions for every number evidence.py and the README claim.

The first group is the one that matters: a variance-reduction harness whose
null is not calibrated cannot measure a reduction, only report one. Those tests
fail if the estimators stop being unbiased or the intervals stop covering.
"""

from __future__ import annotations

import numpy as np
import pytest

import cuped

W = cuped.World()


# ------------------------------------------------------- the null comes first


@pytest.mark.parametrize("name", list(cuped.ADJUSTERS))
def test_every_adjuster_is_size_correct_under_no_effect(name):
    r = np.random.default_rng(1)
    w = cuped.World(true_rel_lift=0.0)
    d = cuped.simulate(w, 4_000, r)
    est, se = cuped.ADJUSTERS[name](d)
    s = cuped.score(est, se, 0.0)
    assert 0.042 < s["reject_rate"] < 0.060, (name, s["reject_rate"])
    assert 0.940 < s["coverage"] < 0.960, (name, s["coverage"])


@pytest.mark.parametrize("name", list(cuped.ADJUSTERS))
def test_every_adjuster_is_unbiased_with_an_effect(name):
    r = np.random.default_rng(2)
    d = cuped.simulate(W, 4_000, r)
    est, se = cuped.ADJUSTERS[name](d)
    s = cuped.score(est, se, W.true_effect)
    assert abs(s["bias"]) < 0.006, (name, s["bias"])


def test_the_world_has_the_correlation_it_claims():
    r = np.random.default_rng(3)
    d = cuped.simulate(W, 40, r)
    rhos = [np.corrcoef(d["pre_c"][i], d["post_c"][i])[0, 1] for i in range(40)]
    assert abs(float(np.mean(rhos)) - W.rho) < 0.02, np.mean(rhos)


# ------------------------------------------------ section 1: the whole result


def test_theta_star_is_rho_times_the_sd_ratio():
    assert abs(cuped.theta_star(W) - 0.60) < 1e-12
    w = cuped.World(sd_pre=8.0, rho=0.40)
    assert abs(cuped.theta_star(w) - 0.40 * 4.0 / 8.0) < 1e-12


def test_measured_reduction_matches_rho_squared():
    r = np.random.default_rng(4)
    d = cuped.simulate(cuped.World(true_rel_lift=0.0), 6_000, r)
    b, _ = cuped.adj_none(d)
    c, _ = cuped.adj_cuped(d)
    pt, se = cuped.reduction_with_mc(c, b)
    assert abs(pt - cuped.variance_reduction(W.rho)) < 3 * se, (pt, se)
    assert 0.33 < pt < 0.39


def test_cuped_raises_power_on_identical_data():
    r = np.random.default_rng(5)
    d = cuped.simulate(W, 4_000, r)
    p_none = cuped.score(*cuped.adj_none(d), W.true_effect)["reject_rate"]
    p_cuped = cuped.score(*cuped.adj_cuped(d), W.true_effect)["reject_rate"]
    assert 0.46 < p_none < 0.52, p_none
    assert 0.64 < p_cuped < 0.71, p_cuped
    assert p_cuped - p_none > 0.15


# ------------------------------------------- section 2: the saving is rho^2


@pytest.mark.parametrize("rho,mult", [(0.3, 0.910), (0.5, 0.750), (0.6, 0.640), (0.8, 0.360)])
def test_sample_size_multiplier_is_one_minus_rho_squared(rho, mult):
    assert abs(cuped.sample_size_multiplier(rho) - mult) < 1e-9


def test_halving_the_test_needs_rho_point_seven_one():
    assert abs(cuped.rho_for_saving(0.5) - 0.70710678) < 1e-7
    assert abs(cuped.sample_size_multiplier(cuped.rho_for_saving(0.5)) - 0.5) < 1e-12


# ----------------------------------- section 3: theta = 1 can double variance


@pytest.mark.parametrize(
    "sd_pre,sd_post,rho,ratio",
    [(4.0, 4.0, 0.60, 0.80), (4.0, 4.0, 0.30, 1.40), (6.0, 4.0, 0.40, 2.05), (8.0, 4.0, 0.40, 3.40)],
)
def test_unit_theta_variance_ratio_closed_form(sd_pre, sd_post, rho, ratio):
    assert abs(cuped.variance_ratio_unit_theta(rho, sd_pre, sd_post) - ratio) < 1e-9


def test_unit_theta_hurts_exactly_when_sd_pre_exceeds_two_rho_sd_post():
    # the boundary itself
    assert abs(cuped.variance_ratio_unit_theta(0.5, 4.0, 4.0) - 1.0) < 1e-12
    assert cuped.variance_ratio_unit_theta(0.51, 4.0, 4.0) < 1.0
    assert cuped.variance_ratio_unit_theta(0.49, 4.0, 4.0) > 1.0


def test_unit_theta_measured_agrees_with_the_closed_form():
    r = np.random.default_rng(6)
    w = cuped.World(sd_pre=6.0, rho=0.40, true_rel_lift=0.0)
    d = cuped.simulate(w, 6_000, r)
    b, _ = cuped.adj_none(d)
    u, _ = cuped.adj_diff_in_diff(d)
    ratio = (u.std(ddof=1) / b.std(ddof=1)) ** 2
    _, se = cuped.reduction_with_mc(u, b)
    assert abs(ratio - 2.05) < 3 * se, (ratio, se)
    assert ratio > 1.8  # it really does nearly double the variance


def test_fitting_theta_still_helps_where_unit_theta_hurts():
    r = np.random.default_rng(7)
    w = cuped.World(sd_pre=8.0, rho=0.40, true_rel_lift=0.0)
    d = cuped.simulate(w, 4_000, r)
    b, _ = cuped.adj_none(d)
    u, _ = cuped.adj_diff_in_diff(d)
    c, _ = cuped.adj_cuped(d)
    assert (u.std(ddof=1) / b.std(ddof=1)) ** 2 > 3.0
    assert cuped.reduction_with_mc(c, b)[0] > 0.12


# --------------------------- section 4: mean-imputation breaks even at f = 0.5


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7, 0.9])
def test_impute_breakeven_is_half_regardless_of_rho(rho):
    assert abs(cuped.reduction_mean_impute(rho, 0.5)) < 1e-12
    assert cuped.reduction_mean_impute(rho, 0.45) > 0
    assert cuped.reduction_mean_impute(rho, 0.55) < 0
    assert cuped.impute_breakeven_share() == 0.5


@pytest.mark.parametrize("f,expected", [(0.0, 0.49), (0.2, 0.3675), (0.4, 0.163333),
                                        (0.6, -0.245), (0.8, -1.47)])
def test_impute_formula_values(f, expected):
    assert abs(cuped.reduction_mean_impute(0.7, f) - expected) < 1e-5


def test_mean_imputation_measured_follows_the_derived_formula_not_the_textbook_one():
    r = np.random.default_rng(8)
    f = 0.60
    w = cuped.World(rho=0.7, new_user_share=f, true_rel_lift=0.0)
    d = cuped.simulate(w, 3_000, r)
    b, _ = cuped.adj_none(d)
    i_, _ = cuped.adj_cuped(d)
    pt, se = cuped.reduction_with_mc(i_, b)
    derived = cuped.reduction_mean_impute(0.7, f)      # -0.245
    textbook = cuped.reduction_stratified(0.7, f)      # +0.196
    assert abs(pt - derived) < 4 * se, (pt, derived, se)
    assert abs(pt - textbook) > 8 * se, (pt, textbook, se)
    assert pt < 0  # a variance INCREASE


def test_stratifying_recovers_the_textbook_promise():
    r = np.random.default_rng(9)
    for f in (0.4, 0.6, 0.8):
        w = cuped.World(rho=0.7, new_user_share=f, true_rel_lift=0.0)
        d = cuped.simulate(w, 3_000, r)
        b, _ = cuped.adj_none(d)
        st, _ = cuped.adj_cuped_stratified(d)
        pt, se = cuped.reduction_with_mc(st, b)
        assert pt > 0, (f, pt)
        assert abs(pt - cuped.reduction_stratified(0.7, f)) < 5 * max(se, 0.01), (f, pt, se)


def test_imputation_is_a_no_op_when_nobody_is_missing():
    r = np.random.default_rng(10)
    d = cuped.simulate(cuped.World(), 5, r)
    assert np.array_equal(cuped._impute(d["pre_c"], d["new_c"]), d["pre_c"])


# ------------------------------------------ section 5: heavy tails


@pytest.mark.parametrize("sigma,rho", [(0.5, 0.7795), (1.0, 0.7132), (1.5, 0.5949), (2.0, 0.4391)])
def test_lognormal_pearson_closed_form(sigma, rho):
    assert abs(cuped.lognormal_pearson_rho(0.80, sigma) - rho) < 5e-4


def test_tail_weight_erodes_the_correlation_cuped_can_use():
    light = cuped.lognormal_pearson_rho(0.80, 0.5)
    heavy = cuped.lognormal_pearson_rho(0.80, 2.0)
    assert light > 0.77 and heavy < 0.45
    assert light ** 2 / heavy ** 2 > 3.0   # more than 3x the reduction, same log-scale rho


def test_sample_correlation_is_biased_up_and_unstable_on_a_heavy_tail():
    r = np.random.default_rng(11)
    w = cuped.World(rho=0.80, lognormal=True, log_sigma=2.0, true_rel_lift=0.0)
    d = cuped.simulate(w, 300, r)
    samp = np.array([np.corrcoef(d["pre_c"][i], d["post_c"][i])[0, 1] for i in range(300)])
    pop = cuped.lognormal_pearson_rho(0.80, 2.0)
    assert samp.mean() > pop * 1.10, (samp.mean(), pop)
    assert samp.std(ddof=1) > 0.08, samp.std(ddof=1)


# --------------------------- section 6: a covariate from after assignment


def test_post_assignment_covariate_destroys_the_effect():
    r = np.random.default_rng(12)
    d = cuped.simulate(W, 4_000, r, effect_on_pre=True)
    none = cuped.score(*cuped.adj_none(d), W.true_effect)
    bad = cuped.score(*cuped.adj_cuped(d), W.true_effect)
    assert abs(none["bias"]) < 0.006                      # unadjusted is fine
    assert bad["bias"] / W.true_effect < -0.45             # CUPED loses over 45% of it
    assert bad["coverage"] < 0.80, bad["coverage"]         # and the interval stops covering
    assert bad["reject_rate"] < none["reject_rate"]        # power falls, not rises


def test_pre_period_covariate_is_the_only_difference():
    """Identical call, identical adjuster - only the covariate's timestamp changes."""
    r = np.random.default_rng(13)
    good = cuped.score(*cuped.adj_cuped(cuped.simulate(W, 3_000, r, effect_on_pre=False)),
                       W.true_effect)
    bad = cuped.score(*cuped.adj_cuped(cuped.simulate(W, 3_000, r, effect_on_pre=True)),
                      W.true_effect)
    assert abs(good["bias"]) < 0.006
    assert abs(bad["bias"]) > 0.09
    assert good["coverage"] > 0.93 and bad["coverage"] < 0.80


# ------------------------------- section 7: the free worries


def test_per_arm_theta_is_not_a_problem_even_with_a_multiplicative_effect():
    r = np.random.default_rng(14)
    w = cuped.World(rho=0.6, multiplicative=True, true_rel_lift=0.10)
    d = cuped.simulate(w, 3_000, r)
    a = cuped.score(*cuped.adj_cuped(d), w.mean * 0.10)
    b = cuped.score(*cuped.adj_cuped_per_arm(d), w.mean * 0.10)
    assert abs(a["mean_est"] - b["mean_est"]) < 1e-4, (a["mean_est"], b["mean_est"])
    assert abs(a["sd_est"] - b["sd_est"]) < 1e-4


def test_theta_estimation_costs_almost_nothing_at_tiny_n():
    r = np.random.default_rng(15)
    w = cuped.World(per_arm=20, rho=0.6, true_rel_lift=0.0)
    d = cuped.simulate(w, 6_000, r)
    b, sb = cuped.adj_none(d)
    c, sc = cuped.adj_cuped(d)
    size_b = cuped.score(b, sb, 0.0)["reject_rate"]
    size_c = cuped.score(c, sc, 0.0)["reject_rate"]
    assert size_c - size_b < 0.020, (size_b, size_c)     # tiny inflation
    assert size_c < 0.075
    assert cuped.reduction_with_mc(c, b)[0] > 0.30        # and it already works


def test_theta_cost_has_gone_by_a_hundred_per_arm():
    r = np.random.default_rng(16)
    w = cuped.World(per_arm=100, rho=0.6, true_rel_lift=0.0)
    d = cuped.simulate(w, 6_000, r)
    s = cuped.score(*cuped.adj_cuped(d), 0.0)
    assert 0.042 < s["reject_rate"] < 0.058, s["reject_rate"]
    assert 0.943 < s["coverage"] < 0.960, s["coverage"]


# ---------------------- section 8: the bias it fixes and the bias it cannot


def test_cuped_repairs_composition_damage_the_covariate_can_see():
    r = np.random.default_rng(17)
    w = cuped.World(rho=0.6, drop_low_pre=0.10)
    d = cuped.simulate(w, 2_000, r)
    none = cuped.score(*cuped.adj_none(d), W.true_effect)
    fixed = cuped.score(*cuped.adj_cuped(d), W.true_effect)
    assert none["bias"] / W.true_effect > 1.5            # +150% or worse unadjusted
    assert abs(fixed["bias"] / W.true_effect) < 0.15     # essentially gone


def test_cuped_is_blind_to_composition_damage_it_cannot_see():
    r = np.random.default_rng(18)
    w = cuped.World(rho=0.6, drop_low_residual=0.10)
    d = cuped.simulate(w, 2_000, r)
    none = cuped.score(*cuped.adj_none(d), W.true_effect)
    still = cuped.score(*cuped.adj_cuped(d), W.true_effect)
    assert none["bias"] / W.true_effect > 2.0
    assert abs(still["bias"] - none["bias"]) / abs(none["bias"]) < 0.05   # unchanged


# ---------------------------------------------- library hygiene


def test_unknown_mechanism_is_refused():
    with pytest.raises(ValueError):
        cuped._cuped_core({"post_c": np.zeros((1, 2)), "post_t": np.zeros((1, 2)),
                           "pre_c": np.zeros((1, 2)), "pre_t": np.zeros((1, 2)),
                           "new_c": np.zeros((1, 2), bool), "new_t": np.zeros((1, 2), bool)},
                          "wishful")


def test_the_error_estimate_scales_like_one_over_root_t():
    """The bootstrap SE has to behave like a standard error, or it is decoration."""
    r = np.random.default_rng(19)
    w = cuped.World(sd_pre=6.0, rho=0.40, true_rel_lift=0.0)
    ses = {}
    for t in (2_000, 8_000):
        d = cuped.simulate(w, t, r)
        b, _ = cuped.adj_none(d)
        u, _ = cuped.adj_diff_in_diff(d)
        ses[t] = cuped.reduction_with_mc(u, b)[1]
    # 4x the trials should halve it; allow a wide band, it is itself noisy
    assert 1.5 < ses[2_000] / ses[8_000] < 2.8, ses


def test_measured_ratio_lands_on_the_closed_form_at_high_trial_counts():
    """The claim that resolved the section-3 gap: it was Monte Carlo noise.

    Held to 1.5% at T=32,000, which the T=6,000 run in evidence.py is not.
    """
    r = np.random.default_rng(20)
    w = cuped.World(sd_pre=6.0, rho=0.40, true_rel_lift=0.0)
    bs, us = [], []
    for _ in range(8):
        d = cuped.simulate(w, 4_000, r)
        b, _ = cuped.adj_none(d)
        u, _ = cuped.adj_diff_in_diff(d)
        bs.append(b)
        us.append(u)
    b = np.concatenate(bs)
    u = np.concatenate(us)
    ratio = (u.std(ddof=1) / b.std(ddof=1)) ** 2
    assert abs(ratio - 2.05) / 2.05 < 0.015, ratio


def test_reduction_with_mc_declines_to_guess_on_too_few_trials():
    assert np.isnan(cuped.reduction_with_mc(np.zeros(50) + 1, np.arange(50.0) + 1)[1])
