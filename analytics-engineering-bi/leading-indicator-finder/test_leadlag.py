"""Every number the README states, re-derived and asserted.

Run: ``python -m pytest test_leadlag.py -q``. The expensive studies use fewer
repetitions than ``evidence.py`` and are asserted as inequalities wide enough
that the reduction cannot flip them.
"""

from __future__ import annotations

import leadlag as L
import numpy as np
import pytest
from scipy import stats

W = L.World()


@pytest.fixture(scope="module")
def world():
    return L.simulate(W)


# ---- the simulator matches its own closed forms -------------------------


def test_gains_are_declared_once():
    assert W.gain["signups"] == pytest.approx(W.c_v * W.c_y)
    assert W.gain["activations"] == pytest.approx(W.c_y)
    zero = [c for c in L.CANDIDATES if c not in ("signups", "activations")]
    assert all(W.gain[c] == 0.0 for c in zero)


@pytest.mark.parametrize("metric", L.CANDIDATES)
def test_do_operator_matches_closed_form(metric):
    """Common random numbers, so this is exact rather than statistical."""
    deltas = []
    for s in range(8):
        base = L.simulate(W, seed=5000 + s)
        forced = L.simulate(W, seed=5000 + s, force=(metric, 1.0))
        deltas.append(forced["revenue"].mean() - base["revenue"].mean())
    assert float(np.mean(deltas)) == pytest.approx(W.gain[metric], abs=1e-9)


def test_sensors_forecast_but_cannot_be_pushed():
    for c in ("web_sessions", "awareness_index"):
        assert c in W.informative
        assert W.gain[c] == 0.0
    assert W.actionable == ["signups", "activations"]


def test_simulation_is_reproducible():
    a, b = L.simulate(W, seed=11), L.simulate(W, seed=11)
    assert np.array_equal(a["revenue"], b["revenue"])
    assert not np.array_equal(L.simulate(W, seed=12)["revenue"], a["revenue"])


# ---- the lag convention ------------------------------------------------


def test_lagged_corr_sign_convention():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = np.zeros(400)
    y[4:] = x[:-4]          # x leads y by 4
    assert L.rank_pearson_lead(x, y)[1] == 4
    assert L.rank_pearson_abs_sym(y, x)[1] == -4


def test_deseasonalize_removes_a_pure_annual_cycle():
    t = np.arange(240)
    z = 3.0 * L.season(t) + 1.5
    assert np.abs(L.deseasonalize(z)).max() < 1e-9


# ---- the funnel trades warning time against everything else ------------


def test_earlier_stage_is_weaker_and_less_pushable(world):
    corr = {c: L.lagged_corr(world[c], world["revenue"], W.true_lead[c])
            for c in ("activations", "signups", "web_sessions")}
    assert corr["activations"] > corr["signups"] > corr["web_sessions"]
    assert W.gain["activations"] > W.gain["signups"] > W.gain["web_sessions"]
    assert corr["activations"] == pytest.approx(0.709, abs=0.01)
    assert corr["web_sessions"] == pytest.approx(0.612, abs=0.01)


# ---- what each ranker says ---------------------------------------------


def test_abs_ccf_peak_crowns_a_metric_that_follows_revenue(world):
    y = world["revenue"]
    peaks = {c: L.rank_pearson_abs_sym(world[c], y) for c in L.CANDIDATES}
    top = max(L.CANDIDATES, key=lambda c: abs(peaks[c][0]))
    assert top == "support_tickets"
    r, lag = peaks[top]
    assert lag < 0 and r == pytest.approx(0.955, abs=0.01)
    best_real = max(L.rank_pearson_lead(world[c], y)[0] for c in W.informative)
    assert abs(r) > best_real + 0.20


def test_persistence_gives_a_lagging_metric_a_positive_lead_correlation(world):
    r, lag = L.rank_pearson_lead(world["support_tickets"], world["revenue"])
    assert lag > 0 and r > 0.30
    assert L.granger_f(world["support_tickets"], world["revenue"])[1] > 0.05


def test_shared_calendar_almost_ties_the_best_real_indicator(world):
    raw = L.rank_pearson_lead(world["marketing_spend"], world["revenue"])[0]
    pw = L.rank_prewhitened(world["marketing_spend"], world["revenue"])[0]
    best_real = max(L.rank_pearson_lead(world[c], world["revenue"])[0]
                    for c in W.informative)
    assert raw > 0.55 and best_real - raw < 0.12
    assert pw < 0.30


def test_granger_separates_perfectly_on_this_world(world):
    y = world["revenue"]
    for c in W.informative:
        assert L.granger_f(world[c], y)[1] < 1e-10
    for c in [c for c in L.CANDIDATES if c not in W.informative]:
        assert L.granger_f(world[c], y)[1] > 0.05


# ---- the horizon -------------------------------------------------------


@pytest.fixture(scope="module")
def oos(world):
    return {h: {c: L.oos_gain(world["revenue"], world[c], h) for c in L.CANDIDATES}
            for h in (1, 3)}


def test_oos_never_uses_a_lag_shorter_than_the_horizon(world):
    for h in (1, 3, 6):
        g = L.oos_gain(world["revenue"], world["web_sessions"], h)
        assert g["lag"] >= h


def test_horizon_reverses_the_shortlist(oos):
    best1 = max(L.CANDIDATES, key=lambda c: oos[1][c]["gain_pct"])
    best3 = max(L.CANDIDATES, key=lambda c: oos[3][c]["gain_pct"])
    assert best1 == "activations" and best3 == "web_sessions"
    assert oos[1]["activations"]["gain_pct"] > 40
    assert oos[3]["activations"]["gain_pct"] < 5
    assert oos[3]["activations"]["dm_p"] > 0.05      # no longer significant
    rho = stats.spearmanr([oos[1][c]["gain_pct"] for c in W.informative],
                          [oos[3][c]["gain_pct"] for c in W.informative]).statistic
    assert rho < 0


def test_every_distractor_is_rejected_at_the_needed_horizon(oos):
    for c in [c for c in L.CANDIDATES if c not in W.informative]:
        assert oos[3][c]["dm_p"] > 0.05
    assert max(oos[3][c]["gain_pct"] for c in L.CANDIDATES
               if c not in W.informative) < 3.0


def test_a_positive_percentage_is_not_a_finding(oos):
    """A placebo posts a positive gain; the test on the differential does not."""
    p1 = oos[3]["placebo_1"]
    assert p1["gain_pct"] > 0 and p1["dm_p"] > 0.05


def test_top_forecaster_at_the_needed_horizon_cannot_be_moved(oos):
    top = max(L.CANDIDATES, key=lambda c: oos[3][c]["gain_pct"])
    assert W.gain[top] == 0.0
    lever = max(W.actionable, key=lambda c: W.gain[c])
    assert W.true_lead[lever] == 1


def test_diebold_mariano_is_silent_on_identical_errors():
    """A degenerate loss differential is no evidence, so the p-value is 1.0."""
    rng = np.random.default_rng(5)
    e = rng.normal(size=200)
    assert L.diebold_mariano(e, e.copy()) == 1.0
    assert L.diebold_mariano(e * 2.0, e) < 0.01
    assert L.diebold_mariano(e, e * 2.0) > 0.99


# ---- the measured null -------------------------------------------------


@pytest.fixture(scope="module")
def null_rates():
    reps = 120
    acc = {}
    for s in range(reps):
        y, X = L.simulate_null(120, 90000 + s, kind="ar1")
        for k, v in L.scan_flags(y, X).items():
            acc[k] = acc.get(k, 0) + int(v)
    return {k: v / reps for k, v in acc.items()}


def test_correlation_scan_finds_an_indicator_in_an_empty_world(null_rates):
    assert null_rates["scan_naive"] > 0.95
    assert null_rates["one_test_naive"] > 0.10       # nominal 0.05


def test_neither_correction_alone_is_enough(null_rates):
    assert null_rates["scan_bonferroni"] > 0.40
    assert null_rates["scan_bartlett"] > 0.70
    assert null_rates["scan_bartlett_bonferroni"] < 0.05


def test_granger_is_calibrated_and_only_needs_multiplicity(null_rates):
    family = 1 - 0.95 ** 10
    assert abs(null_rates["granger_best_of_k"] - family) < 0.15
    assert null_rates["granger_bonferroni"] < 0.12


def test_bartlett_overcorrects_the_random_walk():
    reps = 120
    acc = {}
    for s in range(reps):
        y, X = L.simulate_null(120, 90000 + s, kind="rw")
        for k, v in L.scan_flags(y, X).items():
            acc[k] = acc.get(k, 0) + int(v)
    rw = {k: v / reps for k, v in acc.items()}
    assert rw["fixed_lag_naive"] > 0.70
    assert rw["fixed_lag_bartlett"] < 0.15          # far below the AR(1) case


# ---- the lag estimate --------------------------------------------------


def test_lag_is_the_easy_part_when_the_signal_is_real():
    est = [L.rank_pearson_lead(L.simulate(L.World(T=240), seed=7000 + s)["signups"],
                               L.simulate(L.World(T=240), seed=7000 + s)["revenue"])[1]
           for s in range(60)]
    assert np.mean(np.array(est) == 2) > 0.95


def test_lag_is_unreadable_when_the_signal_is_weak():
    ww = L.World(T=60, sd_web=9.0)
    hits, rs = [], []
    for s in range(80):
        d = L.simulate(ww, seed=8000 + s)
        r, lag = L.rank_pearson_lead(d["web_sessions"], d["revenue"])
        hits.append(lag == 3)
        rs.append(r)
    assert float(np.mean(rs)) < 0.32
    assert float(np.mean(hits)) < 0.30


def test_lag_stability_passes_the_calendar_only_metric(world):
    shares = {}
    n = world["revenue"].size
    for c in L.CANDIDATES:
        lags = [L.rank_pearson_lead(world[c][s:s + 96], world["revenue"][s:s + 96])[1]
                for s in range(0, n - 96 + 1, 6)]
        a = np.array(lags)
        shares[c] = float(np.mean(a == np.bincount(a).argmax()))
    assert shares["marketing_spend"] == 1.0            # a perfect score
    assert max(shares[c] for c in ("placebo_1", "placebo_2", "placebo_3")) < 0.75
    assert min(shares[c] for c in W.informative) > 0.90


# ---- the corrections behave -------------------------------------------


def test_bartlett_is_never_more_permissive_than_the_naive_test():
    for r in (0.2, 0.4, 0.6):
        for rx in (0.0, 0.5, 0.9):
            n, ry = 120, 0.7
            assert (L.bartlett_corr_pvalue(r, n, rx, ry)
                    >= L.naive_corr_pvalue(r, n) - 1e-12)
