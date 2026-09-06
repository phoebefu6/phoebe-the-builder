"""Assertions for every number the README and evidence.py claim.

The first group is the one that matters most: a simulator whose healthy world
does not produce a calibrated null cannot measure power, and a broken null
looks exactly like a strong detector. Those tests are written to FAIL if the
assignment stops being randomised between arms.
"""

from __future__ import annotations

import numpy as np
import pytest
import srm
from scipy import stats

W = srm.World()
SEED = 20260903


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------- the null


def test_healthy_world_is_a_calibrated_null_at_005():
    """The whole harness rests on this. 0.052 measured, nominal 0.05."""
    r = np.random.default_rng(1)
    d = srm.simulate(W, "healthy", 0.0, 20_000, r)
    fpr = float((srm.vector_p_chi2(d["n_ctrl"], d["n_trt"]) < 0.05).mean())
    assert 0.045 < fpr < 0.056, fpr


def test_healthy_world_is_a_calibrated_null_at_00005():
    r = np.random.default_rng(2)
    d = srm.simulate(W, "healthy", 0.0, 40_000, r)
    fpr = float((srm.vector_p_chi2(d["n_ctrl"], d["n_trt"]) < 0.0005).mean())
    assert 0.0001 < fpr < 0.0015, fpr


def test_arm_counts_are_random_not_quota():
    """If each arm were handed exactly per_arm users the null would be degenerate."""
    r = np.random.default_rng(3)
    d = srm.simulate(W, "healthy", 0.0, 3_000, r)
    sd = float(np.std(d["n_ctrl"]))
    expected = np.sqrt(2 * W.per_arm * 0.25)  # sd of Binomial(2m, 0.5)
    assert 0.9 * expected < sd < 1.1 * expected, (sd, expected)


def test_per_segment_null_is_calibrated():
    r = np.random.default_rng(4)
    d = srm.simulate_segmented(W.per_arm, srm.DEFAULT_SEGMENTS, None, 0.0, 8_000, r)
    ps = np.vstack([srm.vector_p_chi2(d["n_ctrl_seg"][i], d["n_trt_seg"][i]) for i in range(3)])
    assert 0.044 < float((ps < 0.05).mean()) < 0.057
    # and the any-of-three rate is the multiplicity arithmetic, not a bug
    any_of_3 = float((ps.min(axis=0) < 0.05).mean())
    assert abs(any_of_3 - (1 - 0.95 ** 3)) < 0.012, any_of_3


# ---------------------------------------- section 1: the tests are the same


def test_published_chi_square_critical_values():
    assert abs(srm.chi2_critical(0.05) - 3.8415) < 5e-4
    assert abs(srm.chi2_critical(0.0005) - 12.1157) < 5e-4


def test_z_squared_is_the_chi_square_statistic_exactly():
    n, dev = 200_000, 0.003
    a = int(round(n * (0.5 + dev)))
    z = (a / n - 0.5) / np.sqrt(0.25 / n)
    assert abs(srm.chi2_stat(a, n - a) - z * z) < 1e-9


def test_chi_square_tracks_the_exact_binomial():
    worst = 0.0
    for n in (100, 1_000, 10_000, 200_000):
        for dev in (0.0, 0.002, 0.005, 0.01, 0.02):
            a = int(round(n * (0.5 + dev)))
            pc = max(srm.p_chi2(a, n - a), 1e-300)
            pe = max(srm.p_binom_exact(a, n - a), 1e-300)
            worst = max(worst, abs(np.log10(pc) - np.log10(pe)))
    assert worst < 0.05, worst
    assert 10 ** worst < 1.12


def test_five_tests_almost_never_disagree():
    """6 of 4,000 in evidence.py. A single-seed 'identical' claim was seed luck -
    Yates and the exact test do differ, at the boundary, ~0.15% of the time."""
    disagree = checked = 0
    for seed in range(8):
        r = np.random.default_rng(500 + seed)
        d = srm.simulate(W, "mcar_loss", 0.015, 500, r)
        for i in range(500):
            a, b = int(d["n_ctrl"][i]), int(d["n_trt"][i])
            verdicts = {
                srm.p_chi2(a, b) < 0.0005,
                srm.p_chi2_yates(a, b) < 0.0005,
                srm.p_g_test(a, b) < 0.0005,
                srm.p_normal_z(a, b) < 0.0005,
                srm.p_binom_exact(a, b) < 0.0005,
            }
            checked += 1
            disagree += len(verdicts) > 1
    assert checked == 4_000
    assert disagree == 6, disagree
    assert disagree / checked < 0.002


# ------------------------------- section 2: a ratio is not a finding


@pytest.mark.parametrize(
    "n,lo,hi",
    [(1_000, 0.6, 0.7), (10_000, 0.15, 0.18), (100_000, 5e-6, 2e-5), (1_000_000, 1e-45, 1e-43)],
)
def test_the_same_split_spans_44_orders_of_magnitude(n, lo, hi):
    a = int(round(n * 0.493))
    p = srm.p_chi2(a, n - a)
    assert lo <= p <= hi, (n, p)


def test_crossing_points_for_493_507():
    def cross(alpha: float) -> int:
        lo, hi = 100, 50_000_000
        for _ in range(60):
            mid = (lo + hi) // 2
            a = int(round(mid * 0.493))
            if srm.p_chi2(a, mid - a) < alpha:
                hi = mid
            else:
                lo = mid
        return hi

    assert abs(cross(0.05) - 19_575) <= 40
    assert abs(cross(0.0005) - 61_856) <= 80


# ---------------------- section 3: "within 1%" names two different rules


def test_within_one_percent_names_two_rules_four_times_apart():
    encoded = 1.01 / 2.01 - 0.5
    assert abs(encoded - 0.0024876) < 1e-6
    assert abs(0.01 / encoded - 4.02) < 0.02


def test_share_eyeball_is_inert_and_ratio_eyeball_is_uncontrolled():
    r = np.random.default_rng(6)
    healthy = srm.simulate(W, "healthy", 0.0, 6_000, r)
    broken = srm.simulate(W, "mcar_loss", 0.015, 6_000, r)

    assert srm.flag_rate(broken["n_ctrl"], broken["n_trt"], srm.eyeball_abs, 0.5) == 0.0
    # inert even where the p-value is exactly zero
    assert srm.eyeball_abs(4_930_000, 5_070_000) == 1.0
    assert srm.p_chi2(4_930_000, 5_070_000) < 1e-300

    fp = srm.flag_rate(healthy["n_ctrl"], healthy["n_trt"], srm.eyeball_ratio, 0.5)
    assert 0.018 < fp < 0.032, fp  # ~48x the 0.0005 it stands in for
    assert srm.flag_rate(broken["n_ctrl"], broken["n_trt"], srm.eyeball_ratio, 0.5) > 0.85


# --------------- section 4: the check is more sensitive than the experiment


@pytest.mark.parametrize("per_arm", [5_000, 25_000, 100_000, 1_000_000])
def test_sensitivity_ratio_is_a_constant_of_the_design(per_arm):
    mde = srm.mde_rel_lift(per_arm, W.base_rate, 0.05)
    rel_dev = srm.mdd_share(2 * per_arm, 0.05) / 0.5
    assert 5.95 < mde / rel_dev < 6.10, (per_arm, mde / rel_dev)


def test_the_ratio_survives_a_hundredfold_stricter_alpha():
    mde = srm.mde_rel_lift(100_000, W.base_rate, 0.05)
    rel_dev = srm.mdd_share(200_000, 0.0005) / 0.5
    assert 3.85 < mde / rel_dev < 3.95


def test_mdd_and_mde_match_the_textbook_forms():
    """Independent check: MDD should be (z_a + z_b) * se, near enough."""
    n = 200_000
    se = np.sqrt(0.25 / n)
    approx = (stats.norm.isf(0.025) + stats.norm.isf(0.20)) * se
    assert abs(srm.mdd_share(n, 0.05) - approx) / approx < 0.02


# ------------ section 5: sensitive, and still not sensitive enough


@pytest.mark.parametrize(
    "per_arm,alpha,loss,bias",
    [
        (5_000, 0.0005, 0.0829, 1.518),
        (25_000, 0.0005, 0.0379, 0.662),
        (100_000, 0.0005, 0.0191, 0.328),
        (1_000_000, 0.0005, 0.0061, 0.103),
    ],
)
def test_detectable_loss_already_carries_material_bias(per_arm, alpha, loss, bias):
    got_loss = srm.loss_for_share_deviation(srm.mdd_share(2 * per_arm, alpha))
    assert abs(got_loss - loss) < 0.0006, got_loss
    rate = min(got_loss / W.low_share, 1.0)
    got_bias = (srm.analytic_est_lift(W, "selective_loss", rate) - W.true_rel_lift) / W.true_rel_lift
    assert abs(got_bias - bias) < 0.02, got_bias


def test_bias_is_scale_free_while_detection_is_not():
    """The same mechanism carries the same bias at every sample size."""
    b = [srm.analytic_est_lift(srm.World(per_arm=m), "selective_loss", 0.05) for m in
         (5_000, 100_000, 5_000_000)]
    assert max(b) - min(b) < 1e-12
    d = [srm.mdd_share(2 * m, 0.0005) for m in (5_000, 100_000, 5_000_000)]
    assert d[0] / d[2] > 25


# -------------------- section 6: a passing check is not evidence


def test_balanced_selective_loss_leaves_the_split_exactly_even():
    assert srm.expected_share(W, "balanced_selective", 0.05) == 0.5


def test_balanced_selective_loss_flags_at_the_null_while_biasing_the_effect():
    r = np.random.default_rng(7)
    d = srm.simulate(W, "balanced_selective", 0.05, 6_000, r)
    p = srm.vector_p_chi2(d["n_ctrl"], d["n_trt"])
    flag = float((p < 0.05).mean())
    assert 0.043 < flag < 0.060, flag  # the null, not a weak signal
    assert float((p < 0.0005).mean()) < 0.002

    est = float(d["est_rel_lift"].mean())
    bias = (est - W.true_rel_lift) / W.true_rel_lift
    assert 0.22 < bias < 0.29, bias


def test_analytic_bias_matches_the_simulation():
    r = np.random.default_rng(8)
    for mech, rate in (("healthy", 0.0), ("mcar_loss", 0.015),
                       ("selective_loss", 0.05), ("balanced_selective", 0.05)):
        d = srm.simulate(W, mech, rate, 6_000, r)
        got = float(d["est_rel_lift"].mean())
        want = srm.analytic_est_lift(W, mech, rate)
        assert abs(got - want) < 0.004, (mech, got, want)


# ----------- section 7: the verdict says nothing about the harm


def test_identical_count_loss_identical_flag_rate_opposite_consequences():
    r = np.random.default_rng(9)
    assert (srm.count_loss_of(W, "mcar_loss", 0.015)
            == pytest.approx(srm.count_loss_of(W, "selective_loss", 0.05)))
    a = srm.simulate(W, "mcar_loss", 0.015, 6_000, r)
    b = srm.simulate(W, "selective_loss", 0.05, 6_000, r)
    fa = float((srm.vector_p_chi2(a["n_ctrl"], a["n_trt"]) < 0.05).mean())
    fb = float((srm.vector_p_chi2(b["n_ctrl"], b["n_trt"]) < 0.05).mean())
    assert abs(fa - fb) < 0.02, (fa, fb)

    bias_a = (float(a["est_rel_lift"].mean()) - W.true_rel_lift) / W.true_rel_lift
    bias_b = (float(b["est_rel_lift"].mean()) - W.true_rel_lift) / W.true_rel_lift
    assert abs(bias_a) < 0.02
    assert 0.22 < bias_b < 0.29


# -------------------- section 8: segments and the threshold


def test_segment_confined_break_is_missed_by_the_aggregate_check():
    r = np.random.default_rng(10)
    d = srm.simulate_segmented(W.per_arm, srm.DEFAULT_SEGMENTS, "safari", 0.06, 3_000, r)
    agg = float((srm.vector_p_chi2(d["n_ctrl"], d["n_trt"]) < 0.0005).mean())
    ps = np.vstack([srm.vector_p_chi2(d["n_ctrl_seg"][i], d["n_trt_seg"][i]) for i in range(3)])
    bonf = float((ps.min(axis=0) < 0.0005 / 3).mean())
    assert agg < 0.12, agg
    assert bonf > 0.90, bonf
    assert bonf / max(agg, 1e-9) > 7


def test_bonferroni_segment_sweep_costs_nothing_on_a_healthy_world():
    r = np.random.default_rng(11)
    d = srm.simulate_segmented(W.per_arm, srm.DEFAULT_SEGMENTS, None, 0.0, 12_000, r)
    ps = np.vstack([srm.vector_p_chi2(d["n_ctrl_seg"][i], d["n_trt_seg"][i]) for i in range(3)])
    assert float((ps.min(axis=0) < 0.0005 / 3).mean()) < 0.0015
    assert float((ps.min(axis=0) < 0.05).mean()) > 0.11


@pytest.mark.parametrize("looks,lo,hi", [(1, 0.040, 0.058), (5, 0.115, 0.155), (20, 0.220, 0.290)])
def test_daily_srm_checking_inflates_its_own_alpha(looks, lo, hi):
    r = np.random.default_rng(12 + looks)
    got = srm.sequential_srm_fpr(200_000, looks, 0.05, 4_000, r)
    assert lo < got < hi, got


def test_the_strict_threshold_absorbs_daily_checking():
    r = np.random.default_rng(99)
    got = srm.sequential_srm_fpr(200_000, 20, 0.0005, 6_000, r)
    assert 0.0015 < got < 0.008, got  # inflated, but two orders below the reflex


# -------------------- library hygiene


def test_unknown_mechanism_is_refused():
    with pytest.raises(ValueError):
        srm.simulate(W, "wishful", 0.1, 10, np.random.default_rng(0))


def test_world_base_rate_is_exactly_ten_percent():
    assert abs(W.base_rate - 0.10) < 1e-12


def test_loss_for_share_deviation_inverts_expected_share():
    for loss in (0.0, 0.005, 0.02, 0.10):
        dev = srm.expected_share(W, "mcar_loss", loss) - 0.5
        assert abs(srm.loss_for_share_deviation(dev) - loss) < 1e-9
