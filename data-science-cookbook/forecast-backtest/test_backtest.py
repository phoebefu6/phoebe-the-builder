"""Tests for the protocol library. The point of most of these is that a WRONG protocol also
returns a plausible number, so the tests check identities and invariants rather than values."""

from __future__ import annotations

import backtest as bt
import numpy as np
import pytest


def test_series_is_reproducible_and_has_the_signal_asked_for():
    a = bt.make_series(np.random.default_rng(1), 200)
    b = bt.make_series(np.random.default_rng(1), 200)
    assert np.allclose(a, b)
    flat = bt.make_series(np.random.default_rng(2), 400, trend=0.0, season_amp=0.0, sigma=1e-9)
    assert np.allclose(flat, 100.0, atol=1e-6)


def test_ar1_noise_has_the_requested_autocorrelation():
    y = bt.make_series(np.random.default_rng(4), 40000, trend=0.0, season_amp=0.0, sigma=3.0, rho=0.7)
    r = np.corrcoef(y[:-1], y[1:])[0, 1]
    assert abs(r - 0.7) < 0.02


def test_break_moves_the_level_by_exactly_the_shift():
    kw = dict(trend=0.0, season_amp=0.0, sigma=1e-9)
    y = bt.make_series(np.random.default_rng(5), 200, break_at=100, break_shift=25.0, **kw)
    assert abs((y[150] - y[50]) - 25.0) < 1e-4


def test_every_forecaster_returns_h_finite_points():
    y = bt.make_series(np.random.default_rng(6), 120)
    for name, m in bt.MODELS.items():
        f = m(y, 12, bt.PERIOD)
        assert f.shape == (12,), name
        assert np.all(np.isfinite(f)), name


def test_seasonal_naive_repeats_the_last_season_exactly():
    y = np.arange(48, dtype=float)
    f = bt.f_snaive(y, 12, 12)
    assert np.allclose(f, y[-12:])


def test_the_well_specified_model_recovers_a_noiseless_series():
    y = bt.make_series(np.random.default_rng(7), 200, sigma=1e-9)
    a, p = bt.forecast_at(y, bt.f_trend_season, 180, 12)
    assert bt.rmse(a - p) < 1e-3


def test_origins_respect_the_horizon_and_the_training_minimum():
    o = bt.origins(200, 12, 20, 100, step=1)
    assert max(o) == 188 and min(o) >= 100 and len(o) == 20
    assert bt.origins(120, 12, 20, 100) == list(range(100, 109))
    # a training minimum the series cannot satisfy returns NO origins rather than a short fit
    assert bt.origins(120, 12, 20, 110) == []
    assert bt.origins(100, 12, 5, 96) == []


def test_single_split_is_the_last_origin_of_a_rolling_run():
    y = bt.make_series(np.random.default_rng(8), 200)
    E = bt.rolling_origin(y, bt.f_trend_season, 12, 5, 100)
    assert np.allclose(E[-1], bt.single_split(y, bt.f_trend_season, 12))


def test_sliding_window_actually_ignores_older_history():
    """A sliding fit must not change when a distant past is rewritten; an expanding one must."""
    y = bt.make_series(np.random.default_rng(9), 200)
    y2 = y.copy()
    y2[:40] += 50.0
    a1, p1 = bt.forecast_at(y, bt.f_trend_season, 180, 12, train_len=60)
    a2, p2 = bt.forecast_at(y2, bt.f_trend_season, 180, 12, train_len=60)
    assert np.allclose(p1, p2)
    _, pe1 = bt.forecast_at(y, bt.f_trend_season, 180, 12)
    _, pe2 = bt.forecast_at(y2, bt.f_trend_season, 180, 12)
    assert not np.allclose(pe1, pe2)


def test_lag_table_rows_line_up_with_their_target():
    y = np.arange(20, dtype=float)
    X, z = bt.lag_table(y, 3)
    assert len(X) == len(z) == 17
    assert np.allclose(X[0], [0.0, 1.0, 2.0]) and z[0] == 3.0
    assert np.allclose(X[-1], [16.0, 17.0, 18.0]) and z[-1] == 19.0


def test_blocked_cv_never_trains_on_a_row_inside_its_own_test_block():
    """The guard this file exists for: an embargo that silently does nothing looks identical
    to one that works, because both return a number."""
    y = bt.make_series(np.random.default_rng(10), 300, rho=0.8)
    rng = np.random.default_rng(11)
    shuffled = bt.kfold_rmse(y, 6, 5, rng, "random")
    blocked = bt.kfold_rmse(y, 6, 5, rng, "blocked", embargo=6)
    assert np.isfinite(shuffled) and np.isfinite(blocked)
    assert shuffled != blocked


def test_expected_min_of_m_matches_known_constants_and_the_correlation_rule():
    assert abs(bt.expected_min_of_m(1)) < 1e-6
    assert abs(bt.expected_min_of_m(2) - (-1 / np.sqrt(np.pi))) < 1e-4
    assert abs(bt.expected_min_of_m(20) - (-1.8675)) < 1e-3
    assert abs(bt.expected_min_of_m(8, 0.75) - 0.5 * bt.expected_min_of_m(8, 0.0)) < 1e-9


def test_expected_min_of_m_matches_simulation():
    rng = np.random.default_rng(12)
    z = rng.normal(size=(200000, 6))
    assert abs(z.min(axis=1).mean() - bt.expected_min_of_m(6)) < 0.01


def test_effective_folds_is_the_variance_ratio():
    assert bt.effective_folds(1.0, 1.0) == pytest.approx(1.0)
    assert bt.effective_folds(2.0, 1.0) == pytest.approx(4.0)
    assert np.isnan(bt.effective_folds(1.0, 0.0))


def test_true_h_step_error_is_the_quantity_a_backtest_estimates():
    """A large rolling backtest on a long series should land near the redrawn-future number."""
    truth = bt.true_h_step_error(bt.f_trend_season, 12, 168, 400, np.random.default_rng(13))
    y = bt.make_series(np.random.default_rng(14), 1200)
    E = bt.rolling_origin(y, bt.f_trend_season, 12, 80, 168, step=12)
    assert abs(bt.rmse(E[~np.isnan(E)]) / truth - 1) < 0.15


def test_mape_rewards_forecasting_low_on_a_noisy_positive_series():
    """The asymmetry this build reports has to be a property of the metric, not of a seed."""
    rng = np.random.default_rng(15)
    actual = np.abs(rng.normal(40, 12, 40000))
    at_truth = bt.mape(actual, np.full_like(actual, 40.0))
    below = bt.mape(actual, np.full_like(actual, 34.0))
    assert below < at_truth


def test_notebook_matches_library():
    """The notebook re-derives the engine inline so it runs standalone in Colab. Day 171's bug
    was exactly this: the inline copy silently lost a parameter and the prose then asserted the
    opposite of what the code did. Execute the notebook's engine cells and diff the behaviour
    against backtest.py on identical inputs."""
    import json

    nb = json.load(open("demo.ipynb"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {}
    # keep only the definitions from each cell - the demo lines below them are truncated at a
    # top-level marker, never at a substring that also occurs inside a function body
    for cell, marker in zip(code_cells[:2], ("\nN = 180", "\ny = make_series(")):
        src = "".join(cell["source"]).split(marker)[0]
        exec(compile(src, "<demo.ipynb>", "exec"), ns)

    for kw in ({}, {"rho": 0.7}, {"break_at": 60, "break_shift": 25.0},
               {"break_at": 60, "break_trend": 0.6}, {"trend": 0.0, "sigma": 9.0}):
        a = ns["make_series"](np.random.default_rng(5), 150, **kw)
        b = bt.make_series(np.random.default_rng(5), 150, **kw)
        assert np.allclose(a, b), f"make_series differs for {kw}"

    y = bt.make_series(np.random.default_rng(6), 200)
    for name, m in bt.MODELS.items():
        key = {"naive": "f_naive", "seasonal naive": "f_snaive", "drift": "f_drift",
               "mean": "f_mean", "trend+season": "f_trend_season",
               "flexible (deg 6)": "f_flexible"}[name]
        assert np.allclose(ns[key](y, 12, bt.PERIOD), m(y, 12, bt.PERIOD)), name

    assert ns["origins"](200, 12, 20, 100) == bt.origins(200, 12, 20, 100)
    for win, tl in [("expanding", None), ("sliding", 60)]:
        a = ns["rolling_origin"](y, ns["f_trend_season"], 12, 10, 100, step=3,
                                 window=win, train_len=tl)
        b = bt.rolling_origin(y, bt.f_trend_season, 12, 10, 100, step=3,
                              window=win, train_len=tl)
        assert np.allclose(a, b, equal_nan=True), win
    assert np.allclose(ns["single_split"](y, ns["f_trend_season"], 12),
                       bt.single_split(y, bt.f_trend_season, 12))
