"""Assertions behind every number in the README. Run: python -m pytest test_hte.py"""

from __future__ import annotations

import hte
import numpy as np
import pytest
from scipy import stats

# --------------------------------------------------------------------- estimator basics


def test_diff_means_is_unbiased_and_its_se_is_right():
    rng = np.random.default_rng(0)
    ests = []
    for _ in range(400):
        d = hte.trial_partition(rng, n=4000, k=1, ate=0.20)
        ests.append(hte.diff_means(d["y"], d["w"])[0])
    assert abs(np.mean(ests) - 0.20) < 4 * np.std(ests, ddof=1) / np.sqrt(400)
    assert abs(np.std(ests, ddof=1) - hte.segment_se(4000, 1)) < 0.004


def test_diff_means_refuses_an_empty_arm_instead_of_dividing_by_zero():
    y, w = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])
    est, se = hte.diff_means(y, w)
    assert np.isnan(est) and np.isinf(se)


def test_segment_se_is_sqrt_k_times_the_experiment_se():
    assert hte.segment_se(8000, 20) == pytest.approx(np.sqrt(20) * hte.segment_se(8000, 1), rel=1e-12)
    assert hte.segment_se(8000, 20) == pytest.approx(0.1, rel=1e-12)


def test_scan_matches_diff_means_group_by_group():
    rng = np.random.default_rng(1)
    d = hte.trial_partition(rng, n=2000, k=4, ate=0.1)
    g = hte.partition_matrix(d["seg"], 4)
    res = hte.scan(d["y"], d["w"], g)
    for j in range(4):
        m = d["seg"] == j
        assert res["est"][j] == pytest.approx(hte.diff_means(d["y"][m], d["w"][m])[0], rel=1e-12)
    assert g.sum(axis=1).max() == 1 and g.sum(axis=1).min() == 1


def test_scan_returns_p_one_for_a_group_with_no_control_users():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.array([1, 1, 0, 1])
    groups = np.array([[True, False], [True, False], [False, True], [True, False]])
    res = hte.scan(y, w, groups)
    assert res["p"][0] == 1.0


# --------------------------------------------------------------------- multiplicity maths


def test_bonferroni_scales_and_caps_at_one():
    p = np.array([0.001, 0.02, 0.5])
    assert hte.bonferroni(p).tolist() == pytest.approx([0.003, 0.06, 1.0])


def test_bh_is_monotone_and_never_below_the_raw_p():
    rng = np.random.default_rng(2)
    p = np.sort(rng.random(50))
    adj = hte.bh(p)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)
    assert np.all(adj >= p - 1e-12)
    assert hte.bh(np.array([0.01, 0.02, 0.03]))[2] == pytest.approx(0.03)


def test_fwer_arithmetic_has_no_data_in_it():
    assert hte.fwer_independent(20, 0.025) == pytest.approx(1.0 - 0.975**20)
    assert hte.k_effective(hte.fwer_independent(13, 0.025), 0.025) == pytest.approx(13.0, rel=1e-9)


def test_harm_alarm_closed_form_reduces_to_the_quoted_number_at_a_zero_effect():
    assert hte.harm_alarm_closed_form(20, 8000, 0.0) == pytest.approx(hte.fwer_independent(20, 0.025), rel=1e-9)
    assert hte.harm_alarm_closed_form(20, 8000, 0.0) == pytest.approx(0.3973, abs=1e-4)


def test_harm_alarm_closed_form_matches_simulation_when_the_effect_is_real():
    rng = np.random.default_rng(3)
    fires = 0
    for _ in range(1200):
        d = hte.trial_partition(rng, n=8000, k=20, ate=0.05)
        fires += int(hte.harm_flag(hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))))
    predicted = hte.harm_alarm_closed_form(20, 8000, 0.05)
    assert abs(fires / 1200 - predicted) < 4 * np.sqrt(predicted * (1 - predicted) / 1200)
    assert predicted < 0.5 * hte.fwer_independent(20, 0.025)  # the naive form is 3x too big here


def test_expected_extreme_z_is_symmetric_and_agrees_with_simulation():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(20000, 20)).min(axis=1)
    assert hte.expected_extreme_z(20, "min") == pytest.approx(-hte.expected_extreme_z(20, "max"), rel=1e-9)
    assert hte.expected_extreme_z(20, "min") == pytest.approx(float(draws.mean()), abs=0.02)
    assert hte.expected_extreme_z(1, "min") == pytest.approx(0.0, abs=1e-9)


def test_the_winners_curse_closed_form_predicts_the_worst_segment():
    rng = np.random.default_rng(5)
    worst = []
    for _ in range(600):
        d = hte.trial_partition(rng, n=8000, k=20, ate=0.05)
        worst.append(hte.scan(d["y"], d["w"], hte.partition_matrix(d["seg"], 20))["est"].min())
    predicted = 0.05 + hte.segment_se(8000, 20) * hte.expected_extreme_z(20, "min")
    assert abs(np.mean(worst) - predicted) < 4 * np.std(worst, ddof=1) / np.sqrt(600)
    assert np.mean(worst) < 0.0  # a positive effect everywhere, reported as harm


def test_mde_is_the_effect_a_test_finds_eighty_percent_of_the_time():
    rng = np.random.default_rng(6)
    delta = hte.mde(2000)
    hits = 0
    for _ in range(500):
        d = hte.trial_partition(rng, n=4000, k=1, ate=delta)
        est, se = hte.diff_means(d["y"], d["w"])
        hits += int(abs(est / se) > stats.norm.isf(0.025))
    assert abs(hits / 500 - 0.80) < 0.05


# --------------------------------------------------------------------- the CATE tree


def test_transformed_outcome_is_an_unbiased_effect_signal():
    rng = np.random.default_rng(7)
    d = hte.trial_partition(rng, n=200000, k=1, ate=0.20)
    assert hte.transformed_outcome(d["y"], d["w"]).mean() == pytest.approx(0.20, abs=0.02)


def test_the_tree_splits_on_the_covariate_that_carries_the_heterogeneity():
    rng = np.random.default_rng(8)
    picks = []
    for _ in range(20):
        c = hte.trial_continuous(rng, n=6000, ate=0.05, beta=0.60)
        picks.append(hte.causal_tree(c["x"], c["w"], c["y"], depth=1, min_leaf=200).feat)
    assert picks.count(0) >= 18


def test_leaf_bookkeeping_is_consistent():
    rng = np.random.default_rng(9)
    c = hte.trial_continuous(rng, n=3000, beta=0.5)
    tree = hte.causal_tree(c["x"], c["w"], c["y"], depth=2, min_leaf=100)
    lid = hte.leaf_of(tree, c["x"])
    assert set(np.unique(lid)) <= set(range(hte.n_leaves(tree)))
    eff = hte.leaf_effects(tree, c["x"], c["w"], c["y"])
    assert eff["n"].sum() == c["x"].shape[0]


def test_the_honest_half_is_unbiased_where_the_fitting_half_is_not():
    rng = np.random.default_rng(10)
    ins, hon = [], []
    for _ in range(120):
        c = hte.trial_continuous(rng, n=4000, ate=0.05, beta=0.0)
        h = hte.honest_fit(rng, c["x"], c["w"], c["y"], depth=2, min_leaf=100)
        j = int(np.argmin(h["in_sample"]["est"]))
        ins.append(h["in_sample"]["est"][j])
        hon.append(h["honest"]["est"][j])
    assert np.mean(ins) < -0.15  # a subgroup that does not exist, reported as badly harmed
    assert abs(np.mean(hon) - 0.05) < 4 * np.std(hon, ddof=1) / np.sqrt(120)


def test_in_sample_intervals_do_not_cover_and_honest_ones_do():
    rng = np.random.default_rng(11)
    cov_i, cov_h = [], []
    for _ in range(150):
        c = hte.trial_continuous(rng, n=4000, ate=0.05, beta=0.0)
        h = hte.honest_fit(rng, c["x"], c["w"], c["y"], depth=2, min_leaf=100)
        j = int(np.argmin(h["in_sample"]["est"]))
        cov_i.append(hte.ci_covers(h["in_sample"]["est"][j], h["in_sample"]["se"][j], 0.05))
        cov_h.append(hte.ci_covers(h["honest"]["est"][j], h["honest"]["se"][j], 0.05))
    assert np.mean(cov_i) < 0.55
    assert np.mean(cov_h) > 0.88


# --------------------------------------------------------------------- the algebraic guard


def test_consistency_gap_is_an_identity_for_a_pre_assignment_partition():
    rng = np.random.default_rng(12)
    gaps = []
    for _ in range(120):
        d = hte.trial_partition(rng, n=20000, k=4, ate=0.05)
        gaps.append(hte.consistency_gap(d["y"], d["w"], hte.partition_matrix(d["seg"], 4)))
    assert max(abs(g) for g in gaps) < 0.2  # in units of the overall SE
    assert np.std(gaps, ddof=1) < 0.05


def test_consistency_gap_screams_on_a_post_assignment_gate():
    rng = np.random.default_rng(13)
    gaps = []
    for _ in range(60):
        d = hte.trial_post_gate(rng, n=20000, ate=0.05, gate_effect=0.30)
        gaps.append(hte.consistency_gap(d["y"], d["w"], np.stack([d["engaged"], ~d["engaged"]], axis=1)))
    assert np.mean(gaps) < -5.0
    assert max(gaps) < -3.0  # every single run, not on average


def test_the_post_gate_world_has_no_real_heterogeneity_and_still_looks_like_it_does():
    rng = np.random.default_rng(14)
    e_eff, n_eff, overall = [], [], []
    for _ in range(120):
        d = hte.trial_post_gate(rng, n=20000, ate=0.05, gate_effect=0.30)
        e = d["engaged"]
        e_eff.append(hte.diff_means(d["y"][e], d["w"][e])[0])
        n_eff.append(hte.diff_means(d["y"][~e], d["w"][~e])[0])
        overall.append(hte.diff_means(d["y"], d["w"])[0])
    assert np.mean(overall) == pytest.approx(0.05, abs=0.01)
    assert np.mean(e_eff) < 0 and np.mean(n_eff) < 0  # both parts negative, the whole positive


def test_the_worlds_are_what_they_claim_to_be():
    rng = np.random.default_rng(15)
    d = hte.trial_partition(rng, n=10000, k=10, ate=0.05, delta=0.30, harmed=3)
    assert sorted(np.unique(d["tau"])) == pytest.approx([-0.25, 0.05])
    assert np.all(d["tau"][d["seg"] == 3] < 0)
    c = hte.trial_continuous(rng, n=5000, ate=0.05, beta=0.25)
    assert c["tau"].min() < 0 < c["tau"].max()  # the effect changes sign
    s = hte.trial_slices(rng, n=20000, n_slices=6, rho=0.9)
    corr = np.corrcoef(s["slices"].astype(float).T)
    assert corr[np.triu_indices(6, 1)].mean() > 0.5
