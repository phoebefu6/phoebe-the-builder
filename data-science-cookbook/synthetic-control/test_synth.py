"""Assertions behind every claim in evidence.txt, the README and the notebook.

Run: python -m pytest -q test_synth.py
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from synth import (
    attenuation_prediction,
    fit_synth,
    hull_excess,
    in_donor_band,
    make_panel,
    naive_estimates,
    placebo_pvalue,
    pvalue_floor,
    rmse,
    solve_simplex_ls,
)

EFFECT = 5.0


# ------------------------------------------------------------------ the solver


def test_solver_matches_a_general_purpose_optimiser():
    """The nnls penalty trick is not an approximation anyone should have to trust."""
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(25, 8)), rng.normal(size=25)
    w = solve_simplex_ls(X, y)

    def obj(v):
        return float(((X @ v - y) ** 2).sum())

    res = minimize(
        obj,
        np.full(8, 1 / 8),
        method="SLSQP",
        bounds=[(0, None)] * 8,
        constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    assert obj(w) <= obj(res.x) + 1e-6
    assert np.abs(w).min() >= -1e-9


def test_weights_are_on_the_simplex():
    rng = np.random.default_rng(3)
    Y = make_panel(rng, effect=EFFECT)
    w = fit_synth(Y, 0, 30).weights
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert (w >= -1e-9).all()


def test_ols_leaves_the_simplex():
    rng = np.random.default_rng(3)
    Y = make_panel(rng, effect=EFFECT)
    w = fit_synth(Y, 0, 30, solver="ols").weights
    assert (w < -1e-6).any(), "section 6 depends on OLS producing negative weights"


def test_ols_interpolates_when_donors_outnumber_pre_periods():
    """Section 6's headline: a perfect pre-period fit, with no information in it."""
    rng = np.random.default_rng(11)
    Y = make_panel(rng, n_donors=20, t_pre=12, effect=EFFECT)
    assert fit_synth(Y, 0, 12, solver="ols").pre_rmspe < 1e-8
    assert fit_synth(Y, 0, 12, solver="simplex").pre_rmspe > 0.1


# ------------------------------------------------------------------ section 1


def test_synthetic_beats_every_naive_alternative():
    atts, naive = [], {k: [] for k in ("best_donor", "donor_mean", "did")}
    for s in range(300):
        rng = np.random.default_rng(1000 + s)
        Y = make_panel(rng, effect=EFFECT)
        atts.append(fit_synth(Y, 0, 30).att)
        for k, v in naive_estimates(Y, 0, 30).items():
            naive[k].append(v)
    sc = rmse(np.array(atts), EFFECT)
    assert abs(np.mean(atts) - EFFECT) < 0.1
    for k, v in naive.items():
        assert rmse(np.array(v), EFFECT) > sc * 1.5, f"{k} should be >1.5x worse than synthetic"


def test_fit_is_sparse_without_any_penalty():
    rng = np.random.default_rng(5)
    w = fit_synth(make_panel(rng, effect=EFFECT), 0, 30).weights
    assert (w > 1e-4).sum() < len(w), "the simplex should zero out some donors on its own"


# ------------------------------------------------------------------ section 2


def test_pvalue_floor_is_arithmetic():
    for j in (5, 10, 19, 20, 50):
        assert pvalue_floor(j) == pytest.approx(1.0 / (j + 1))
    assert pvalue_floor(18) > 0.05 and pvalue_floor(19) == pytest.approx(0.05)
    assert pvalue_floor(8) > 0.10 and pvalue_floor(9) == pytest.approx(0.10)


def test_ten_donors_can_never_reject_at_five_percent():
    """Not low power - zero power.  A huge effect changes nothing."""
    for eff in (5.0, 50.0, 500.0):
        for s in range(25):
            rng = np.random.default_rng(2000 + s)
            p, _ = placebo_pvalue(make_panel(rng, n_donors=10, effect=eff), 0, 30)
            assert p >= pvalue_floor(10) > 0.05


def test_pvalue_only_takes_j_plus_one_values():
    rng = np.random.default_rng(9)
    p, stats = placebo_pvalue(make_panel(rng, n_donors=20, effect=EFFECT), 0, 30)
    assert len(stats) == 21
    assert p * 21 == pytest.approx(round(p * 21))


def test_placebo_test_is_calibrated_under_the_null():
    ps = [
        placebo_pvalue(make_panel(np.random.default_rng(21000 + s), effect=0.0), 0, 30)[0]
        for s in range(300)
    ]
    assert 0.02 <= float(np.mean(np.array(ps) <= 0.10)) <= 0.20


# ------------------------------------------------------------------ section 3


def test_pre_fit_improves_with_donors_while_accuracy_does_not():
    out = {}
    for j in (5, 80):
        pre, atts = [], []
        for s in range(300):
            rng = np.random.default_rng(30000 + s)
            f = fit_synth(make_panel(rng, n_donors=j, t_pre=10, effect=EFFECT), 0, 10)
            pre.append(f.pre_rmspe)
            atts.append(f.att)
        out[j] = (float(np.mean(pre)), rmse(np.array(atts), EFFECT))
    assert out[80][0] < out[5][0] / 2, "pre-fit must get visibly better"
    assert out[80][1] > out[5][1] * 0.9, "accuracy must not follow it"


# ------------------------------------------------------------------ section 4


def test_synthetic_unit_never_leaves_the_donor_range():
    """The structural guarantee, at machine precision, including far outside the hull."""
    for h in (0.0, 5.0, 50.0):
        for s in range(40):
            rng = np.random.default_rng(40000 + s)
            Y = make_panel(rng, effect=EFFECT, hull_shift=h)
            assert in_donor_band(Y, 0, fit_synth(Y, 0, 30))


def test_hull_excess_is_zero_without_a_shift_and_grows_with_one():
    rng = np.random.default_rng(13)
    assert hull_excess(make_panel(rng, effect=EFFECT), 0, 30)["post"] >= 0.0
    a = np.mean(
        [hull_excess(make_panel(np.random.default_rng(40000 + s), effect=EFFECT, hull_shift=0.0), 0, 30)["post"] for s in range(60)]
    )
    b = np.mean(
        [hull_excess(make_panel(np.random.default_rng(40000 + s), effect=EFFECT, hull_shift=10.0), 0, 30)["post"] for s in range(60)]
    )
    assert b > a + 5.0


def test_bias_is_a_fraction_of_the_shift_not_the_shift():
    """The intuitive closed form (bias = h) is wrong: the optimiser buys part of it back."""
    for h, lo, hi in ((5.0, 0.05, 0.6), (10.0, 0.2, 0.8)):
        atts = [
            fit_synth(make_panel(np.random.default_rng(40000 + s), effect=EFFECT, hull_shift=h), 0, 30).att
            for s in range(200)
        ]
        frac = (float(np.mean(atts)) - EFFECT) / h
        assert lo < frac < hi, f"h={h} gave bias/h={frac:.3f}"


def test_more_donors_fix_coverage_bias_but_not_noise():
    """Section 4 against section 9 - the same lever, opposite verdicts."""
    def bias(j):
        atts = [
            fit_synth(make_panel(np.random.default_rng(41000 + s), n_donors=j, effect=EFFECT, hull_shift=5.0), 0, 30).att
            for s in range(200)
        ]
        return float(np.mean(atts)) - EFFECT

    def noise_rmse(j):
        atts = [
            fit_synth(make_panel(np.random.default_rng(90000 + s), n_donors=j, t_pre=8, effect=EFFECT), 0, 8).att
            for s in range(200)
        ]
        return rmse(np.array(atts), EFFECT)

    assert bias(80) < bias(5) * 0.6, "donors must remove coverage bias"
    assert noise_rmse(80) > noise_rmse(5) * 0.8, "donors must NOT remove noise"


# ------------------------------------------------------------------ section 5


def test_raw_gap_false_alarms_on_an_unfittable_unit_and_the_ratio_does_not():
    def fire(stat):
        ps = [
            placebo_pvalue(
                make_panel(np.random.default_rng(50000 + s), effect=0.0, hull_shift=5.0), 0, 30, statistic=stat
            )[0]
            for s in range(250)
        ]
        return float(np.mean(np.array(ps) <= 0.10))

    assert fire("gap") > 0.25
    assert fire("ratio") < 0.12


def test_the_ratio_pays_for_it_in_power():
    def power(stat):
        ps = [
            placebo_pvalue(
                make_panel(np.random.default_rng(50000 + s), effect=EFFECT, hull_shift=5.0), 0, 30, statistic=stat
            )[0]
            for s in range(250)
        ]
        return float(np.mean(np.array(ps) <= 0.10))

    assert power("ratio") < power("gap") - 0.2


# ------------------------------------------------------------------ section 7


def _contamination_gap(gamma, n_hit, n_sims=250):
    """measured attenuation minus predicted attenuation, on identical panels."""
    ratios, preds = [], []
    for s in range(n_sims):
        base = make_panel(np.random.default_rng(70000 + s), effect=EFFECT)
        hit = [int(i) for i in np.argsort(fit_synth(base, 0, 30).weights)[::-1][:n_hit]]
        Y = make_panel(np.random.default_rng(70000 + s), effect=EFFECT, contaminated={i: gamma for i in hit})
        f = fit_synth(Y, 0, 30)
        ratios.append(f.att / EFFECT)
        preds.append(attenuation_prediction(f.weights, {i: gamma for i in hit}))
    return float(np.mean(ratios) - np.mean(preds))


def test_contamination_matches_the_closed_form():
    """estimate/tau == 1 - sum_j w_j gamma_j, using the weights actually chosen.

    The raw difference does not go to zero, it goes to the estimator's own clean bias - which
    is what the gamma=0 row measures.  Net of that constant the closed form is exact, and the
    fact that the SAME constant appears at every contamination level is the actual evidence
    that the attenuation is fully explained by the weights.
    """
    clean = _contamination_gap(0.0, 1)
    assert abs(clean) < 0.01
    for gamma, n_hit in ((0.5, 1), (1.0, 1), (1.0, 3), (0.5, 5)):
        assert abs(_contamination_gap(gamma, n_hit) - clean) < 0.002, f"gamma={gamma} hit={n_hit}"


def test_contamination_leaves_the_pre_period_fit_untouched():
    clean, dirty = [], []
    for s in range(150):
        base = make_panel(np.random.default_rng(70000 + s), effect=EFFECT)
        f0 = fit_synth(base, 0, 30)
        hit = [int(i) for i in np.argsort(f0.weights)[::-1][:3]]
        Y = make_panel(np.random.default_rng(70000 + s), effect=EFFECT, contaminated={i: 1.0 for i in hit})
        clean.append(f0.pre_rmspe)
        dirty.append(fit_synth(Y, 0, 30).pre_rmspe)
    assert abs(np.mean(clean) - np.mean(dirty)) < 0.02, "the diagnostic must stay perfect while the answer is wrong"


# ------------------------------------------------------------------ section 8


def test_anticipation_biases_the_estimate_downward():
    """Opposite sign to every other bias in this build."""
    base = np.mean([fit_synth(make_panel(np.random.default_rng(80000 + s), effect=EFFECT), 0, 30).att for s in range(200)])
    ant = np.mean(
        [fit_synth(make_panel(np.random.default_rng(80000 + s), effect=EFFECT, anticipation=6), 0, 30).att for s in range(200)]
    )
    assert ant < base - 1.0


def test_the_prefit_screen_sees_loud_anticipation_and_misses_quiet_anticipation():
    clean = [fit_synth(make_panel(np.random.default_rng(80000 + s), effect=EFFECT), 0, 30).pre_rmspe for s in range(300)]
    thresh = float(np.quantile(clean, 0.90))

    def power(a, share):
        return float(
            np.mean(
                [
                    fit_synth(
                        make_panel(np.random.default_rng(80000 + s), effect=EFFECT, anticipation=a, anticipation_share=share),
                        0,
                        30,
                    ).pre_rmspe
                    > thresh
                    for s in range(300)
                ]
            )
        )

    assert power(6, 1.0) > 0.8
    assert power(4, 0.25) < 0.3


# ------------------------------------------------------------------ section 9


def test_history_buys_accuracy_and_donors_do_not():
    def r(tp, j):
        atts = [
            fit_synth(make_panel(np.random.default_rng(90000 + s), n_donors=j, t_pre=tp, effect=EFFECT), 0, tp).att
            for s in range(300)
        ]
        return rmse(np.array(atts), EFFECT)

    assert r(120, 5) < r(8, 5) * 0.7, "16x the history must clearly help"
    assert r(8, 80) > r(8, 5) * 0.85, "16x the donors must not"
