"""Tests for the interval library. A wrong interval method returns a plausible band, so these
check identities, guarantees and the two failure modes that look like success."""

from __future__ import annotations

import intervals as iv
import numpy as np
import pytest


def test_series_is_reproducible_and_returns_the_sd_it_used():
    a, sda = iv.make_series(np.random.default_rng(1), 200)
    b, sdb = iv.make_series(np.random.default_rng(1), 200)
    assert np.allclose(a, b) and np.allclose(sda, sdb)
    assert np.allclose(sda, 4.0)


def test_volatility_regimes_hold_the_average_variance_fixed():
    """The whole point of the regime world: nothing in the aggregate looks different."""
    y, sd = iv.make_series(np.random.default_rng(2), 24000, vol_period=24, vol_ratio=9.0)
    assert abs(np.mean(sd**2) / 16.0 - 1.0) < 0.05
    assert len(np.unique(np.round(sd, 6))) == 2


def test_regime_phase_varies_across_draws():
    """Without a per-series phase the regime is a deterministic function of t, a fixed training
    length always forecasts into the same regime, and the conditional split has an empty cell."""
    rng = np.random.default_rng(3)
    seen = {tuple(np.unique(np.round(iv.make_series(rng, 180, vol_period=24, vol_ratio=9.0)[1][168:], 4)))
            for _ in range(40)}
    assert len(seen) > 1


def test_fit_recovers_a_noiseless_trend_and_season():
    y, _ = iv.make_series(np.random.default_rng(4), 200, sigma=1e-9)
    fit = iv.fit_trend_season(y[:180])
    assert np.allclose(iv.predict(fit, 12), y[180:192], atol=1e-4)
    assert fit["s2"] < 1e-12


def test_leverage_grows_with_extrapolation_distance():
    y, _ = iv.make_series(np.random.default_rng(5), 200)
    lev = iv.leverage(iv.fit_trend_season(y[:168]), 24)
    assert lev[0] < lev[-1]
    assert np.all(lev > 0)


def test_every_method_returns_ordered_finite_bands():
    y, _ = iv.make_series(np.random.default_rng(6), 200)
    for name, m in iv.METHODS.items():
        lo, hi = m(y[:168], 12, 0.95)
        assert lo.shape == hi.shape == (12,), name
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)), name
        assert np.all(hi > lo), name


def test_a_higher_confidence_level_is_a_wider_band():
    y, _ = iv.make_series(np.random.default_rng(7), 300)
    for name, m in iv.METHODS.items():
        w80 = np.mean(np.subtract(*reversed(m(y[:240], 12, 0.80))))
        w99 = np.mean(np.subtract(*reversed(m(y[:240], 12, 0.99))))
        assert w99 > w80, name


def test_sqrt_h_band_widens_and_the_flat_one_does_not():
    y, _ = iv.make_series(np.random.default_rng(8), 200)
    lo, hi = iv.gaussian_iid(y[:168], 12, 0.95)
    w = hi - lo
    assert abs(w[-1] / w[0] - np.sqrt(12)) < 1e-9
    lo2, hi2 = iv.gaussian_flat(y[:168], 12, 0.95)
    assert np.allclose(hi2 - lo2, (hi2 - lo2)[0])


def test_conformal_uses_the_finite_sample_index_not_the_plain_quantile():
    """The ceil((m+1)(1-alpha))/m index is what buys the guarantee; the plain quantile is the
    off-by-one that silently removes it, so the band must be at least as wide."""
    y, _ = iv.make_series(np.random.default_rng(9), 260)
    lo, hi = iv.split_conformal(y[:240], 12, 0.95, k=20, min_train=120)
    errs = iv._rolling_errors(y[:240], 12, 20, 120)
    plain = np.array([np.quantile(np.abs(errs[:, j][~np.isnan(errs[:, j])]), 0.95)
                      for j in range(12)])
    assert np.all((hi - lo) / 2 >= plain - 1e-12)


def test_rolling_errors_are_out_of_sample_at_every_origin():
    """A fit that peeked at its own test points would move when those points move. Perturb the
    single last observation: it belongs to exactly one origin's window, at exactly one horizon,
    and the error there must shift by exactly the perturbation while every other cell is
    untouched - which is only true if no training set ever contained it."""
    y, _ = iv.make_series(np.random.default_rng(10), 220)
    base = iv._rolling_errors(y, 12, 15, 120)
    y2 = y.copy()
    y2[219] += 50.0
    delta = iv._rolling_errors(y2, 12, 15, 120) - base
    hit = np.argwhere(np.abs(np.nan_to_num(delta)) > 1e-9)
    assert hit.tolist() == [[14, 11]], f"the perturbation reached {hit.tolist()}"
    assert delta[14, 11] == pytest.approx(50.0)


def test_coverage_run_agrees_with_a_hand_rolled_count():
    rng = np.random.default_rng(11)
    r = iv.coverage_run(iv.gaussian_flat, 6, 120, 50, rng, 0.95)
    rng2 = np.random.default_rng(11)
    hits = []
    for _ in range(50):
        y, _s = iv.make_series(rng2, 126)
        lo, hi = iv.gaussian_flat(y[:120], 6, 0.95)
        hits.append((y[120:] >= lo) & (y[120:] <= hi))
    assert np.array_equal(r["hits"], np.array(hits))


def test_a_well_specified_band_keeps_its_claim():
    """The calibration guard: if this drifts, every 'undercovers' finding is measuring a bug."""
    r = iv.coverage_run(iv.gaussian_theory, 12, 240, 800, np.random.default_rng(12), 0.95)
    assert abs(r["hits"].mean() - 0.95) < 0.02


def test_interval_score_charges_for_width_and_for_misses():
    actual = np.array([10.0, 10.0])
    tight_hit = iv.interval_score(np.array([9.0, 9.0]), np.array([11.0, 11.0]), actual, 0.95)
    wide_hit = iv.interval_score(np.array([0.0, 0.0]), np.array([20.0, 20.0]), actual, 0.95)
    miss = iv.interval_score(np.array([0.0, 0.0]), np.array([5.0, 5.0]), actual, 0.95)
    assert np.mean(tight_hit) < np.mean(wide_hit)
    assert np.mean(miss) > np.mean(wide_hit)


def test_interval_score_cannot_be_gamed_by_widening_a_calibrated_band():
    rng = np.random.default_rng(13)
    s_fair, s_wide = [], []
    for _ in range(200):
        y, _s = iv.make_series(rng, 180)
        a = y[168:]
        lo, hi = iv.gaussian_theory(y[:168], 12, 0.95)
        mid = (lo + hi) / 2
        s_fair.append(np.mean(iv.interval_score(lo, hi, a, 0.95)))
        s_wide.append(np.mean(iv.interval_score(mid + 3 * (lo - mid), mid + 3 * (hi - mid),
                                                a, 0.95)))
    assert np.mean(s_wide) > np.mean(s_fair)


def test_coverage_check_power_behaves_like_a_power_function():
    assert iv.coverage_check_power(0.90, 0.95, 5000) > iv.coverage_check_power(0.90, 0.95, 50)
    assert iv.coverage_check_power(0.94, 0.95, 500) < iv.coverage_check_power(0.80, 0.95, 500)
    assert (iv.coverage_check_power(0.90, 0.95, 1000, 1 / 20)
            < iv.coverage_check_power(0.90, 0.95, 1000))
    assert iv.coverage_check_power(0.95, 0.95, 1000) == pytest.approx(0.05, abs=0.01)


def test_notebook_matches_library():
    """The notebook re-derives the engine inline so it runs standalone in Colab. Day 171's bug
    was exactly this: the inline copy silently lost a parameter and the prose then asserted the
    opposite of what the code did. Execute the notebook's engine cell and diff the behaviour
    against intervals.py on identical inputs."""
    import json

    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(code_cells[0]["source"]).split("\nruns = {")[0]
    ns: dict = {}
    exec(compile(src, "<demo.ipynb>", "exec"), ns)

    for kw in ({}, {"rho": 0.7}, {"vol_period": 24, "vol_ratio": 9.0},
               {"vol_period": 24, "vol_ratio": 9.0, "vol_phase": 3}, {"sigma": 9.0}):
        a, sa = ns["make_series"](np.random.default_rng(5), 200, **kw)
        b, sb = iv.make_series(np.random.default_rng(5), 200, **kw)
        assert np.allclose(a, b), f"make_series differs for {kw}"
        assert np.allclose(sa, sb), f"the sd column differs for {kw}"

    y, _ = iv.make_series(np.random.default_rng(6), 260)
    assert set(ns["METHODS"]) == set(iv.METHODS)
    for name in iv.METHODS:
        for conf in (0.80, 0.95):
            lo_a, hi_a = ns["METHODS"][name](y[:240], 12, conf)
            lo_b, hi_b = iv.METHODS[name](y[:240], 12, conf)
            assert np.allclose(lo_a, lo_b) and np.allclose(hi_a, hi_b), f"{name} at {conf}"

    assert np.allclose(ns["_rolling_errors"](y, 12, 10, 180),
                       iv._rolling_errors(y, 12, 10, 180), equal_nan=True)
    r_a = ns["coverage_run"](ns["gaussian_theory"], 12, 168, 40, np.random.default_rng(7), 0.95)
    r_b = iv.coverage_run(iv.gaussian_theory, 12, 168, 40, np.random.default_rng(7), 0.95)
    assert np.array_equal(r_a["hits"], r_b["hits"])
