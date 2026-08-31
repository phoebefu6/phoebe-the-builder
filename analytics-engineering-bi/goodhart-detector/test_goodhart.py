"""Tests for the world, the detectors, and the claims the README makes.

Anything the README states as a number is either asserted here or derived from
a closed form that is asserted here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import goodhart as G

W = G.World()
SMALL = replace(W, n_agents=250)


# ---------------------------------------------------------------- the world
def test_closed_forms_are_arithmetic():
    assert W.exploit_edge == pytest.approx(W.gamma - W.beta * W.kappa)
    assert W.outcome_cost == pytest.approx(W.a_y * W.kappa)
    assert W.payoff_ratio == pytest.approx(W.gamma / (W.beta * W.kappa))


def test_default_world_has_an_exploit_worth_taking():
    assert W.exploit_edge > 0, "the exploit must beat honest work on the proxy"
    assert W.outcome_cost > 0, "and it must cost the outcome something"


def test_rho_clean_matches_a_simulation_with_nobody_gaming():
    panel = G.simulate(W, gaming=False)
    r = np.corrcoef(panel.proxy.ravel(), panel.outcome.ravel())[0, 1]
    assert r == pytest.approx(W.rho_clean, abs=0.02)


def test_rho_clean_is_a_strong_proxy():
    assert W.rho_clean == pytest.approx(0.80, abs=0.005)


def test_no_gaming_means_no_diverted_effort():
    panel = G.simulate(W, gaming=False, regime="continuous")
    assert panel.diverted.max() == 0.0


def test_gaming_diverts_effort_only_after_the_target_is_set():
    panel = G.simulate(W, regime="continuous")
    assert panel.diverted[: panel.t_target].max() == 0.0
    assert panel.diverted[panel.t_target :].mean() > 0.1


def test_simulation_is_deterministic():
    a = G.simulate(W, regime="threshold")
    b = G.simulate(W, regime="threshold")
    assert np.array_equal(a.proxy, b.proxy)
    assert np.array_equal(a.outcome, b.outcome)


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError):
        G.simulate(W, regime="whatever")


def test_threshold_regime_sets_a_line_and_continuous_does_not():
    assert G.simulate(W, regime="threshold").threshold is not None
    assert G.simulate(W, regime="continuous").threshold is None


# ------------------------------------------------------- the headline result
def test_proxy_rises_while_the_outcome_falls():
    panel = G.simulate(W, regime="continuous")
    d = G.decompose(W, panel)
    assert d["proxy_delta"] > 0, "the KPI improved"
    assert d["outcome_delta"] < 0, "and the thing it stood for got worse"


def test_exchange_rate_matches_its_closed_form():
    d = G.decompose(W, G.simulate(W, regime="continuous"))
    assert d["exchange_rate"] == pytest.approx(d["exchange_rate_closed"], rel=0.10)
    assert d["exchange_rate"] < -1.0


def test_threshold_regime_hides_the_damage_in_the_aggregate():
    """True harm is several times what the observed total shows."""
    d = G.decompose(W, G.simulate(W, regime="threshold"))
    assert d["outcome_delta_true"] < 0
    assert abs(d["outcome_delta_true"]) > 3 * abs(d["outcome_delta"])


def test_correlation_moves_far_less_than_the_outcome():
    panel = G.simulate(W, regime="continuous")
    r_pre = np.corrcoef(panel.pre(panel.proxy), panel.pre(panel.outcome))[0, 1]
    r_post = np.corrcoef(panel.post(panel.proxy), panel.post(panel.outcome))[0, 1]
    d = G.decompose(W, panel)
    lost = abs(d["outcome_delta_true"] / panel.pre(panel.outcome).mean())
    assert abs(r_post - r_pre) < 0.15, "the correlation stays comfortable"
    assert lost > 0.40, "while nearly half the outcome is gone"


# ------------------------------------------------------------- the detectors
def test_every_detector_returns_a_usable_verdict():
    for regime in ("continuous", "threshold"):
        panel = G.simulate(SMALL, regime=regime)
        for name, v in G.run_all(panel).items():
            assert v.name == name
            assert 0.0 <= v.pvalue <= 1.0, f"{name} returned p={v.pvalue}"


def test_the_outcome_free_detectors_are_labelled_as_such():
    panel = G.simulate(SMALL, regime="threshold")
    free = {n for n, v in G.run_all(panel).items() if not v.needs_outcome}
    assert free == {"holdout_divergence", "bunching", "dispersion_shift"}


def test_bunching_is_undefined_without_a_line():
    v = G.bunching(G.simulate(W, regime="continuous"))
    assert np.isnan(v.stat) and v.pvalue == 1.0


def test_bunching_fires_hard_on_a_threshold_target():
    v = G.bunching(G.simulate(W, regime="threshold"))
    assert v.stat > 0.5, "excess mass appears just above the line"
    assert v.pvalue < 1e-6


def test_holdout_divergence_needs_no_outcome_and_still_fires():
    v = G.holdout_divergence(G.simulate(W, regime="continuous"))
    assert not v.needs_outcome
    assert v.stat < 0 and v.pvalue < 0.01


def test_detectors_are_quiet_when_nobody_games():
    panel = G.simulate(W, regime="threshold", gaming=False)
    fired = [n for n, v in G.run_all(panel).items() if v.fires()]
    assert len(fired) <= 1, f"too many false alarms on a clean world: {fired}"


def test_verdict_fires_respects_alpha():
    v = G.Verdict("x", True, 0.0, 0.04)
    assert v.fires(0.05) and not v.fires(0.01)


# ------------------------------------------ the winner's curse (section 3)
def _curse(n_select: int, reps: int = 60):
    win, rnd = [], []
    for rep in range(reps):
        rng = np.random.default_rng(9000 + rep)
        betas, gammas = rng.uniform(0.55, 1.15, 12), rng.uniform(0.45, 1.35, 12)
        y, P = G.simulate_candidates(SMALL, betas, gammas, seed=int(rng.integers(1, 2**31)))
        fy, fP = y.ravel(), P.reshape(12, -1)
        idx = rng.choice(fy.size, n_select, replace=False)
        sel = np.array([np.corrcoef(fP[j][idx], fy[idx])[0, 1] for j in range(12)])
        w_, r_ = int(np.argmax(sel)), int(rng.integers(0, 12))
        win.append(np.corrcoef(fP[w_], fy)[0, 1] - sel[w_])
        rnd.append(np.corrcoef(fP[r_], fy)[0, 1] - sel[r_])
    return float(np.mean(win)), float(np.mean(rnd))


def test_the_chosen_proxy_decays_although_nobody_games():
    win, rnd = _curse(15)
    assert win < -0.03, "selection alone produces a correlation drop"
    assert win < rnd, "and a randomly chosen candidate does not decay like that"


def test_the_curse_shrinks_as_the_selection_sample_grows():
    small, _ = _curse(15)
    large, _ = _curse(900)
    assert abs(small) > 3 * abs(large)


def test_selection_alone_can_mimic_real_gaming():
    """The whole point: a drop this size is not evidence of anything."""
    win, _ = _curse(15)
    panel = G.simulate(replace(W, scruple_median=3.0), regime="continuous")
    r_pre = np.corrcoef(panel.pre(panel.proxy), panel.pre(panel.outcome))[0, 1]
    r_post = np.corrcoef(panel.post(panel.proxy), panel.post(panel.outcome))[0, 1]
    assert abs(win) >= abs(r_post - r_pre), "selection out-drops real gaming"


def test_true_rho_rises_with_the_honest_loading():
    rhos = [G.true_rho(W, b) for b in (0.6, 0.8, 1.0, 1.2)]
    assert rhos == sorted(rhos)


def test_candidate_simulator_shapes():
    betas, gammas = np.linspace(0.6, 1.1, 5), np.full(5, 1.0)
    y, P = G.simulate_candidates(SMALL, betas, gammas, n_periods=9)
    assert y.shape == (9, SMALL.n_agents)
    assert P.shape == (5, 9, SMALL.n_agents)


# ----------------------------------------------- calibration and its absence
def test_residual_trend_is_miscalibrated_and_the_readme_says_so():
    """Documented defect: it ignores the error in its own pre-period fit."""
    fp = sum(
        G.residual_trend(G.simulate(replace(SMALL, seed=4000 + r),
                                    regime="threshold", gaming=False)).pvalue < 0.05
        for r in range(80)
    ) / 80
    assert fp > 0.05, "if this ever calibrates, the README claim must be updated"


def test_corr_drop_is_roughly_calibrated_on_a_proxy_nobody_chose():
    fp = sum(
        G.corr_drop(G.simulate(replace(SMALL, seed=4000 + r),
                               regime="threshold", gaming=False)).pvalue < 0.05
        for r in range(80)
    ) / 80
    assert fp < 0.15, "the inflation in section 3 comes from selection, not the test"


# ------------------------------------------------- the holdout is a policy
def _leak_power(leak: float, reps: int = 50) -> float:
    hits = 0
    for rep in range(reps):
        w = replace(SMALL, seed=6000 + rep)
        panel = G.simulate(w, regime="continuous")
        contaminated = panel.holdout + leak * w.gamma * panel.diverted
        leaked = G.Panel(panel.proxy, panel.outcome, contaminated, panel.diverted,
                         panel.t_target, panel.threshold)
        hits += G.holdout_divergence(leaked).pvalue < 0.05
    return hits / reps


def test_a_clean_holdout_detects_and_a_leaked_one_does_not():
    assert _leak_power(0.0) > 0.9
    assert _leak_power(1.0) < 0.1


def test_partial_leakage_is_survivable():
    assert _leak_power(0.25) > 0.8, "a quarter of the exploit leaking is tolerable"


# ------------------------------------------------------------ panel plumbing
def test_panel_slices_line_up_with_the_target_date():
    panel = G.simulate(SMALL, n_pre=4, n_post=7, regime="continuous")
    assert panel.n_periods == 11 and panel.t_target == 4
    assert panel.pre(panel.proxy).size == 4 * SMALL.n_agents
    assert panel.post(panel.proxy).size == 7 * SMALL.n_agents
    assert panel.post(panel.proxy, upto=6).size == 2 * SMALL.n_agents


def test_detector_names_cover_the_registry():
    assert set(G.DETECTOR_NAMES) == {d.__name__ for d in G.DETECTORS}
    assert len(G.DETECTORS) == 7


def test_fisher_z_clamps_a_perfect_correlation():
    z, se = G._fisher_z(1.0, 100)
    assert np.isfinite(z) and se == pytest.approx(1 / np.sqrt(97))


def test_alarms_stay_quiet_when_the_exploit_is_not_worth_taking():
    """gamma below beta*kappa means honest work is the better way to move the
    proxy, so nobody diverts anything and no detector should complain."""
    for g in (0.20, 0.40, 0.55, 0.59):
        tame = replace(W, gamma=g)
        assert tame.exploit_edge <= 0
        for regime in ("continuous", "threshold"):
            assert G.simulate(tame, regime=regime).diverted.max() == 0.0
    tame = replace(W, gamma=0.20)
    panel = G.simulate(tame, regime="continuous")
    assert G.corr_drop(panel).pvalue > 0.05
    assert G.ratio_shift(panel).pvalue > 0.05
    assert G.holdout_divergence(panel).pvalue > 0.05


def test_bunching_reports_the_excess_mass_ratio():
    panel = G.simulate(W, regime="threshold")
    v = G.bunching(panel, width=0.35)
    assert np.isfinite(v.stat)
    wide = G.bunching(panel, width=1.20)
    assert wide.stat < v.stat, "a wider window dilutes the bunch it is measuring"
