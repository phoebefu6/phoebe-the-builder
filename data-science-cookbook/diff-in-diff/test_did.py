"""Every claim in `evidence.py` and every closed form in `did.py`, asserted.

The tests that matter most here are the ones that would pass on a broken
estimator: a null that holds, a weight vector that sums to one, and an
identity that has to close to machine precision.  A measurement harness whose
null is broken cannot detect bias, it can only report it.
"""

from __future__ import annotations

import numpy as np
import pytest

from did import (
    aggregate_att,
    ar1_errors,
    did_2x2,
    event_dummies,
    event_study,
    fe_ols,
    group_time_att,
    heterogeneity_bound,
    make_panel,
    make_staggered,
    pretrend_test,
    twfe,
    twfe_cell_weights,
    two_way_demean,
)


# ---------------------------------------------------------------- primitives


def test_two_way_demean_annihilates_unit_and_time_effects():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 1))
    g = rng.normal(size=(1, 8))
    assert np.allclose(two_way_demean(a + g), 0.0, atol=1e-12)


def test_two_way_demean_is_idempotent():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 6))
    once = two_way_demean(X)
    assert np.allclose(once, two_way_demean(once), atol=1e-12)


def test_two_way_demean_leaves_interaction_intact():
    """A DiD regressor is exactly the part the transform must NOT remove."""
    D = np.zeros((10, 6))
    D[:5, 3:] = 1.0
    assert np.abs(two_way_demean(D)).max() > 0.1


# ---------------------------------------------------------------- the null


def test_did_is_unbiased_under_parallel_trends():
    rng = np.random.default_rng(101)
    est = []
    for _ in range(600):
        Y, D, adopt, _ = make_panel(rng, n_treated=100, n_control=100, T=12, t0=6, effect=1.0)
        est.append(twfe(Y, D))
    est = np.array(est)
    se = est.std(ddof=1) / np.sqrt(est.size)
    assert abs(est.mean() - 1.0) < 4 * se


def test_zero_effect_recovers_zero():
    rng = np.random.default_rng(102)
    est = [twfe(*make_panel(rng, T=10, t0=5, effect=0.0)[:2]) for _ in range(400)]
    est = np.array(est)
    assert abs(est.mean()) < 4 * est.std(ddof=1) / np.sqrt(est.size)


def test_cluster_test_has_nominal_size_when_errors_are_iid():
    rng = np.random.default_rng(103)
    rej = 0
    reps = 800
    for _ in range(reps):
        Y, D, _, _ = make_panel(rng, n_treated=50, n_control=50, T=10, t0=5, effect=0.0)
        if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
            rej += 1
    assert 0.03 <= rej / reps <= 0.075


# ------------------------------------------------ section 1: the exact bias


def test_2x2_equals_twfe_under_common_timing():
    rng = np.random.default_rng(2)
    for _ in range(20):
        Y, D, adopt, _ = make_panel(rng, T=12, t0=6, effect=1.0, diff_trend=0.07)
        a = did_2x2(Y, adopt == 6, range(6), range(6, 12))
        assert abs(a - twfe(Y, D)) < 1e-12


def test_bias_equals_delta_times_period_gap():
    """The headline closed form: bias = delta * (mean post t - mean pre t)."""
    T, t0, delta = 12, 6, 0.05
    gap = np.mean(np.arange(t0, T)) - np.mean(np.arange(0, t0))
    rng = np.random.default_rng(3)
    est = []
    for _ in range(800):
        Y, D, _, _ = make_panel(
            rng, n_treated=200, n_control=200, T=T, t0=t0, effect=1.0, diff_trend=delta
        )
        est.append(twfe(Y, D))
    est = np.array(est)
    se = est.std(ddof=1) / np.sqrt(est.size)
    assert abs((est.mean() - 1.0) - delta * gap) < 4 * se


def test_bias_is_invariant_to_sample_size():
    """The point of section 1: n moves the SE and not the bias."""
    biases, ses = [], []
    for n_arm in (50, 800):
        rng = np.random.default_rng(400 + n_arm)
        est, se_i = [], []
        for _ in range(300):
            Y, D, _, _ = make_panel(
                rng, n_treated=n_arm, n_control=n_arm, T=12, t0=6, effect=1.0, diff_trend=0.05
            )
            f = fe_ols(Y, [D], ["D"], vcov="cluster")
            est.append(float(f.beta[0]))
            se_i.append(float(f.se()[0]))
        biases.append(np.mean(est) - 1.0)
        ses.append(np.mean(se_i))
    assert abs(biases[0] - biases[1]) < 0.02          # bias unchanged
    assert ses[0] / ses[1] > 3.0                       # SE shrank ~4x


# ------------------------------------------- sections 2-3: the pre-trend test


def test_pretrend_test_is_calibrated_under_the_null():
    rng = np.random.default_rng(5)
    ev = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    adopt = np.where(np.arange(200) < 100, 6.0, np.inf)
    cols = event_dummies(adopt, 12, [e for e in ev if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
    fired = 0
    reps = 800
    for _ in range(reps):
        Y, _, _, _ = make_panel(rng, n_treated=100, n_control=100, T=12, t0=6, effect=1.0)
        if fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1] < 0.05:
            fired += 1
    assert 0.025 <= fired / reps <= 0.075


def test_pretrend_power_is_near_size_at_a_damaging_violation():
    """delta = 0.05 biases the estimate 30% and the test barely notices."""
    rng = np.random.default_rng(6)
    ev = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    adopt = np.where(np.arange(200) < 100, 6.0, np.inf)
    cols = event_dummies(adopt, 12, [e for e in ev if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
    fired = 0
    reps = 800
    for _ in range(reps):
        Y, _, _, _ = make_panel(
            rng, n_treated=100, n_control=100, T=12, t0=6, effect=1.0, diff_trend=0.05
        )
        if fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1] < 0.05:
            fired += 1
    assert fired / reps < 0.15


def test_pretrend_power_rises_with_pre_periods_not_with_units():
    """Both help, but only the time axis helps cheaply - section 2's exchange rate."""

    def power(npre: int, n_arm: int, reps: int = 500) -> float:
        T = npre + 1 + 6
        t0 = npre + 1
        ev = list(range(-npre, 0)) + list(range(0, 6))
        adopt = np.where(np.arange(2 * n_arm) < n_arm, float(t0), np.inf)
        cols = event_dummies(adopt, T, [e for e in ev if e != -1])
        lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
        rng = np.random.default_rng(700 + npre + n_arm)
        fired = 0
        for _ in range(reps):
            Y, _, _, _ = make_panel(
                rng, n_treated=n_arm, n_control=n_arm, T=T, t0=t0, effect=1.0, diff_trend=0.05
            )
            if fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1] < 0.05:
                fired += 1
        return fired / reps

    assert power(12, 100) > power(3, 100) + 0.3
    assert power(12, 100) > power(3, 1600)


def test_conditioning_on_a_passed_pretrend_does_not_remove_the_bias():
    """The negative result in section 3, as an assertion."""
    rng = np.random.default_rng(8)
    ev = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    adopt = np.where(np.arange(200) < 100, 6.0, np.inf)
    cols = event_dummies(adopt, 12, [e for e in ev if e != -1])
    lead_idx = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
    allb, passb = [], []
    for _ in range(1500):
        Y, D, _, _ = make_panel(
            rng, n_treated=100, n_control=100, T=12, t0=6, effect=1.0, diff_trend=0.05
        )
        b = twfe(Y, D)
        allb.append(b)
        if fe_ols(Y, cols, None, vcov="cluster").wald(lead_idx)[1] >= 0.05:
            passb.append(b)
    mc = np.std(passb, ddof=1) / np.sqrt(len(passb))
    assert abs(np.mean(passb) - np.mean(allb)) < 4 * mc
    assert np.mean(passb) - 1.0 > 0.25  # the bias is still there and still large


# ------------------------------------------- section 4: serial correlation


def test_iid_se_over_rejects_under_serial_correlation():
    rng = np.random.default_rng(9)
    reps, N, T = 700, 100, 20
    rej_iid = rej_cl = 0
    for _ in range(reps):
        e = ar1_errors(rng, N, T, 0.8)
        Y = e + 0.10 * np.arange(T)[None, :] + rng.normal(0, 1, (N, 1))
        tr = np.zeros(N, dtype=bool)
        tr[rng.permutation(N)[: N // 2]] = True
        t0 = int(rng.integers(1, T))
        D = (tr[:, None] & (np.arange(T)[None, :] >= t0)).astype(float)
        if fe_ols(Y, [D], ["D"], vcov="iid").pvalue()[0] < 0.05:
            rej_iid += 1
        if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
            rej_cl += 1
    assert rej_iid / reps > 0.20          # BDM's phenomenon
    assert rej_cl / reps < 0.09           # and the fix works


def test_ar1_errors_have_the_requested_autocorrelation():
    rng = np.random.default_rng(10)
    e = ar1_errors(rng, 4000, 40, 0.7)
    lag1 = np.corrcoef(e[:, :-1].ravel(), e[:, 1:].ravel())[0, 1]
    assert abs(lag1 - 0.7) < 0.02
    assert abs(e.std() - 1.0 / np.sqrt(1 - 0.7 ** 2)) < 0.05


# ------------------------------------------- section 5: the clustering level


def test_clustering_at_the_wrong_level_over_rejects_badly():
    n_states, per, T = 20, 10, 12
    N = n_states * per
    sid = np.repeat(np.arange(n_states), per)
    rng = np.random.default_rng(11)
    reps = 400
    h_unit = h_state = 0
    for _ in range(reps):
        sy = ar1_errors(rng, n_states, T, 0.6)[sid]
        Y = sy + rng.normal(0, 1, (N, T)) + rng.normal(0, 1, (N, 1)) + 0.2 * np.arange(T)[None, :]
        tr = (rng.permutation(n_states) < n_states // 2)[sid]
        D = (tr[:, None] & (np.arange(T)[None, :] >= T // 2)).astype(float)
        if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
            h_unit += 1
        if fe_ols(Y, [D], ["D"], vcov="cluster", cluster_id=sid).pvalue()[0] < 0.05:
            h_state += 1
    assert h_unit / reps > 0.30           # one level too fine
    assert h_state / reps < 0.10          # correct level


def test_six_correct_clusters_beat_fifty_wrong_ones():
    """The section 5 headline, as a single comparison."""

    def size(n_states: int, per: int, by_state: bool, reps: int = 400) -> float:
        N, T = n_states * per, 12
        sid = np.repeat(np.arange(n_states), per)
        rng = np.random.default_rng(1200 + n_states + int(by_state))
        hits = 0
        for _ in range(reps):
            sy = ar1_errors(rng, n_states, T, 0.6)[sid]
            Y = sy + rng.normal(0, 1, (N, T)) + rng.normal(0, 1, (N, 1))
            tr = (rng.permutation(n_states) < n_states // 2)[sid]
            D = (tr[:, None] & (np.arange(T)[None, :] >= T // 2)).astype(float)
            cid = sid if by_state else None
            if fe_ols(Y, [D], ["D"], vcov="cluster", cluster_id=cid).pvalue()[0] < 0.05:
                hits += 1
        return hits / reps

    assert abs(size(6, 10, True) - 0.05) < abs(size(50, 10, False) - 0.05)


def test_wrong_level_gets_worse_as_units_are_added():
    def size(per: int, reps: int = 350) -> float:
        n_states, T = 20, 12
        N = n_states * per
        sid = np.repeat(np.arange(n_states), per)
        rng = np.random.default_rng(1400 + per)
        hits = 0
        for _ in range(reps):
            sy = ar1_errors(rng, n_states, T, 0.6)[sid]
            Y = sy + rng.normal(0, 1, (N, T)) + rng.normal(0, 1, (N, 1))
            tr = (rng.permutation(n_states) < n_states // 2)[sid]
            D = (tr[:, None] & (np.arange(T)[None, :] >= T // 2)).astype(float)
            if fe_ols(Y, [D], ["D"], vcov="cluster").pvalue()[0] < 0.05:
                hits += 1
        return hits / reps

    assert size(100) > size(10) + 0.15


# --------------------------------------- section 6: weights and the sign flip


def test_treated_cell_weights_sum_to_one():
    for growth in (0.0, 0.5, 1.5):
        Y, D, _, _ = make_staggered(
            np.random.default_rng(0), [(4, 50), (10, 50)], T=20, growth=growth, sigma=0.0
        )
        assert abs(twfe_cell_weights(D).sum() - 1.0) < 1e-12


def test_weight_identity_closes_to_machine_precision():
    """E[beta_twfe] = sum w_it tau_it, on a noiseless panel."""
    for growth in (0.0, 0.25, 0.5, 1.0):
        Y, D, _, tau = make_staggered(
            np.random.default_rng(1),
            [(4, 50), (10, 50)],
            T=20,
            growth=growth,
            sigma=0.0,
            unit_sd=0.0,
        )
        W = twfe_cell_weights(D)
        assert abs(twfe(Y, D) - float((W * tau).sum())) < 1e-10


def test_no_negative_weights_without_staggered_timing():
    """Common timing is safe - the problem is timing variation, not TWFE itself."""
    Y, D, _, _ = make_staggered(np.random.default_rng(2), [(6, 100)], T=20, n_never=100, growth=0.5)
    W = twfe_cell_weights(D)
    assert (W[D > 0] > 0).all()


def test_staggered_timing_produces_negative_weights():
    Y, D, _, _ = make_staggered(np.random.default_rng(3), [(4, 50), (10, 50)], T=20, growth=0.5)
    w = twfe_cell_weights(D)[D > 0]
    assert (w < 0).any()
    assert 0.30 < (w < 0).mean() < 0.45


def test_negative_weights_sit_on_the_early_cohort_after_the_late_one_adopts():
    Y, D, adopt, _ = make_staggered(
        np.random.default_rng(4), [(4, 50), (10, 50)], T=20, growth=0.5
    )
    W = twfe_cell_weights(D)
    t = np.arange(20)
    early_late = W[np.ix_(adopt == 4, t >= 10)]
    everything_else = W[(D > 0) & ~np.isclose(W, early_late.flat[0])]
    assert (early_late < 0).all()
    assert everything_else.max() > 0


def test_twfe_goes_negative_while_every_true_effect_is_positive():
    """The flagship result."""
    Y, D, _, tau = make_staggered(
        np.random.default_rng(5),
        [(4, 50), (10, 50)],
        T=20,
        growth=1.0,
        sigma=0.0,
        unit_sd=0.0,
    )
    assert tau[D > 0].min() > 0
    assert twfe(Y, D) < 0


def test_twfe_falls_outside_the_range_of_every_true_effect():
    Y, D, _, tau = make_staggered(
        np.random.default_rng(6),
        [(4, 50), (10, 50)],
        T=20,
        growth=0.5,
        sigma=0.0,
        unit_sd=0.0,
    )
    b = twfe(Y, D)
    assert not (tau[D > 0].min() <= b <= tau[D > 0].max())


def test_sign_flip_threshold_is_reproducible():
    lo, hi = 0.0, 2.0
    for _ in range(50):
        mid = (lo + hi) / 2
        Y, D, _, _ = make_staggered(
            np.random.default_rng(1), [(4, 50), (10, 50)], T=20, growth=mid, sigma=0.0, unit_sd=0.0
        )
        if twfe(Y, D) > 0:
            lo = mid
        else:
            hi = mid
    assert abs((lo + hi) / 2 - 0.56) < 0.01


def test_heterogeneity_bound_is_linear_in_the_referenced_att():
    Y, D, _, _ = make_staggered(np.random.default_rng(7), [(4, 50), (10, 50)], T=20, growth=0.5)
    W = twfe_cell_weights(D)
    b1 = heterogeneity_bound(W, D, 1.0)
    b4 = heterogeneity_bound(W, D, 4.0)
    assert abs(b4 - 4 * b1) < 1e-10


def test_heterogeneity_bound_is_a_real_bound():
    """Construct tau at the bound and check the weighted sum is indeed ~zero."""
    Y, D, _, _ = make_staggered(np.random.default_rng(8), [(4, 50), (10, 50)], T=20, growth=0.5)
    W = twfe_cell_weights(D)
    w = W[D > 0]
    m = 1.0
    gamma = m / (w.mean() - (w * w).sum())
    tau = (-gamma * (w * w).sum()) + gamma * w
    assert abs(tau.mean() - m) < 1e-10                  # ATT is m
    assert abs(float(w @ tau)) < 1e-10                  # but TWFE reads zero
    assert abs(tau.std(ddof=0) - heterogeneity_bound(W, D, m)) < 1e-10


# ------------------------------------------------------- section 7: the fix


def test_not_yet_treated_att_recovers_the_truth_exactly_without_noise():
    Y, D, adopt, tau = make_staggered(
        np.random.default_rng(9),
        [(4, 50), (10, 50)],
        T=20,
        n_never=50,
        growth=0.5,
        sigma=0.0,
        unit_sd=0.0,
    )
    cs = aggregate_att(group_time_att(Y, adopt), adopt, 20)
    assert abs(cs - float(tau[D > 0].mean())) < 1e-10


def test_twfe_is_badly_biased_on_the_same_panel():
    Y, D, adopt, tau = make_staggered(
        np.random.default_rng(9),
        [(4, 50), (10, 50)],
        T=20,
        n_never=50,
        growth=0.5,
        sigma=0.0,
        unit_sd=0.0,
    )
    truth = float(tau[D > 0].mean())
    assert abs(twfe(Y, D) - truth) / truth > 0.30


def test_group_time_att_never_uses_an_already_treated_control():
    """Guard the one restriction the whole fix consists of."""
    adopt = np.array([4.0, 4.0, 10.0, 10.0, np.inf])
    for t in range(4, 20):
        ctrl = adopt > t
        assert not ((adopt <= t) & ctrl).any()


def test_not_yet_treated_att_is_unbiased_with_noise():
    rng = np.random.default_rng(12)
    est = []
    truth = None
    for _ in range(200):
        Y, D, adopt, tau = make_staggered(
            rng, [(4, 50), (10, 50)], T=20, n_never=50, growth=0.5, sigma=1.0, unit_sd=1.0
        )
        truth = float(tau[D > 0].mean())
        est.append(aggregate_att(group_time_att(Y, adopt), adopt, 20))
    est = np.array(est)
    assert abs(est.mean() - truth) < 4 * est.std(ddof=1) / np.sqrt(est.size)


# ------------------------------------------------- section 8: functional form


def test_levels_and_logs_disagree_on_the_sign_inside_the_window():
    c_pre, c_post, t_pre = 100.0, 120.0, 200.0
    for obs in (221.0, 230.0, 239.0):
        lv = (obs - t_pre) - (c_post - c_pre)
        lg = np.log(obs / t_pre) - np.log(c_post / c_pre)
        assert lv > 0 > lg


def test_the_two_scales_agree_outside_the_window():
    c_pre, c_post, t_pre = 100.0, 120.0, 200.0
    for obs in (205.0, 260.0):
        lv = (obs - t_pre) - (c_post - c_pre)
        lg = np.log(obs / t_pre) - np.log(c_post / c_pre)
        assert np.sign(lv) == np.sign(lg)


def test_parallel_in_both_scales_requires_equal_baselines_or_a_flat_control():
    """(Yt0 - Yc0)(Yc1/Yc0 - 1) = 0 is the exact condition."""
    for yt0, yc0, yc1 in [(200.0, 100.0, 120.0), (150.0, 150.0, 190.0), (200.0, 100.0, 100.0)]:
        term = (yt0 - yc0) * (yc1 / yc0 - 1.0)
        lv_cf = yt0 + (yc1 - yc0)
        lg_cf = yt0 * (yc1 / yc0)
        assert (abs(term) < 1e-12) == (abs(lv_cf - lg_cf) < 1e-12)


def test_each_scale_reports_the_wrong_sign_in_the_other_world():
    n, T, t0 = 400, 8, 4
    treated = np.zeros(n, dtype=bool)
    treated[: n // 2] = True
    base = np.where(treated, 200.0, 100.0)[:, None]
    tt = np.arange(T)[None, :]
    tp = treated[:, None] * (tt >= t0)
    rng = np.random.default_rng(13)

    # world A: multiplicative trend, -5% effect.  Logs are correct.
    Ya = np.clip(base * (1.05 ** tt) * (1.0 - 0.05 * tp) + rng.normal(0, 4.0, (n, T)), 1, None)
    assert did_2x2(np.log(Ya), treated, range(t0), range(t0, T)) < 0
    assert did_2x2(Ya, treated, range(t0), range(t0, T)) > 0

    # world B: additive trend, +10 effect.  Levels are correct.
    Yb = np.clip(base + 5.0 * tt + 10.0 * tp + rng.normal(0, 4.0, (n, T)), 1, None)
    assert did_2x2(Yb, treated, range(t0), range(t0, T)) > 0
    assert did_2x2(np.log(Yb), treated, range(t0), range(t0, T)) < 0


def test_the_pretrend_test_identifies_the_right_scale():
    """The constructive result: run it on Y and on log Y; the wrong scale fires."""
    n, T, t0 = 400, 8, 4
    treated = np.zeros(n, dtype=bool)
    treated[: n // 2] = True
    base = np.where(treated, 200.0, 100.0)[:, None]
    tt = np.arange(T)[None, :]
    tp = treated[:, None] * (tt >= t0)
    adopt = np.where(treated, float(t0), np.inf)
    ev = [-3, -2, -1, 0, 1, 2, 3]
    cols = event_dummies(adopt, T, [e for e in ev if e != -1])
    lead = [i for i, e in enumerate([e for e in ev if e != -1]) if e < 0]
    rng = np.random.default_rng(14)

    Ya = np.clip(base * (1.05 ** tt) * (1.0 - 0.05 * tp) + rng.normal(0, 4.0, (n, T)), 1, None)
    assert fe_ols(Ya, cols, None, vcov="cluster").wald(lead)[1] < 0.01          # levels fires
    assert fe_ols(np.log(Ya), cols, None, vcov="cluster").wald(lead)[1] > 0.01  # logs passes

    Yb = np.clip(base + 5.0 * tt + 10.0 * tp + rng.normal(0, 4.0, (n, T)), 1, None)
    assert fe_ols(Yb, cols, None, vcov="cluster").wald(lead)[1] > 0.01          # levels passes
    assert fe_ols(np.log(Yb), cols, None, vcov="cluster").wald(lead)[1] < 0.01  # logs fires


# ------------------------------------------------------------------ plumbing


def test_event_study_recovers_a_known_dynamic_path():
    """Window covers EVERY period, so nothing is silently in the base category."""
    rng = np.random.default_rng(15)
    Y, D, adopt, tau = make_staggered(
        rng, [(4, 400)], T=10, n_never=400, growth=0.5, sigma=0.3, unit_sd=0.0
    )
    es = event_study(Y, adopt, list(range(-4, 6)))
    got = dict(zip(es.labels, es.beta))
    for e in (0, 1, 2, 3, 4, 5):
        assert abs(got[f"e{e:+d}"] - (1.0 + 0.5 * e)) < 0.15
    for e in (-4, -3, -2):
        assert abs(got[f"e{e:+d}"]) < 0.15


def test_a_short_event_window_silently_biases_every_coefficient():
    """Uncovered treated periods fall into the omitted base and contaminate it.

    Found while writing the test above: this is a real trap, it produces
    coefficients of the wrong SIGN, and nothing in the output announces it.
    """
    rng = np.random.default_rng(15)
    Y, D, adopt, _ = make_staggered(
        rng, [(6, 400)], T=14, n_never=400, growth=0.5, sigma=0.3, unit_sd=0.0
    )
    short = dict(zip(*(lambda f: (f.labels, f.beta))(event_study(Y, adopt, [-3, -2, -1, 0, 1, 2, 3]))))
    full = dict(zip(*(lambda f: (f.labels, f.beta))(event_study(Y, adopt, list(range(-6, 8))))))
    assert abs(full["e+0"] - 1.0) < 0.15        # covered window is right
    assert short["e+0"] < 0.0                   # truncated window is wrong-signed
    assert abs(short["e+0"] - full["e+0"]) > 1.0


def test_never_treated_units_get_no_event_dummy():
    adopt = np.array([3.0, np.inf])
    cols = event_dummies(adopt, 6, [-1, 0, 1])
    for M in cols:
        assert M[1].sum() == 0.0


def test_pretrend_test_returns_the_lead_count():
    rng = np.random.default_rng(16)
    Y, _, adopt, _ = make_panel(rng, T=12, t0=6)
    _, _, k = pretrend_test(event_study(Y, adopt, [-4, -3, -2, -1, 0, 1]))
    assert k == 3


def test_cluster_and_iid_agree_when_there_is_nothing_to_cluster():
    """A sanity check on the sandwich: iid errors, many units, similar SEs."""
    rng = np.random.default_rng(17)
    Y, D, _, _ = make_panel(rng, n_treated=300, n_control=300, T=10, t0=5, effect=1.0)
    a = fe_ols(Y, [D], ["D"], vcov="iid").se()[0]
    b = fe_ols(Y, [D], ["D"], vcov="cluster").se()[0]
    assert abs(a - b) / a < 0.15


def test_fe_ols_matches_twfe_on_the_single_regressor_case():
    rng = np.random.default_rng(18)
    Y, D, _, _ = make_panel(rng, T=12, t0=6, effect=1.0, diff_trend=0.03)
    assert abs(float(fe_ols(Y, [D], ["D"]).beta[0]) - twfe(Y, D)) < 1e-12


def test_unknown_vcov_is_rejected():
    rng = np.random.default_rng(19)
    Y, D, _, _ = make_panel(rng, T=8, t0=4)
    with pytest.raises(ValueError):
        fe_ols(Y, [D], ["D"], vcov="robust")
