"""Every claim the README makes is pinned here.

Analytic quantities are asserted exactly; simulated ones are asserted with a tolerance
wide enough for Monte Carlo noise and narrow enough to fail if the mechanism changes.
"""

from __future__ import annotations

import guardrails as G
import numpy as np
import pytest
from scipy import stats

N = 8135
DAY = 14
ALPHA = 0.05


# ---------------------------------------------------------------- the world -----------
def test_lever_raises_the_primary_metric_by_12_percent():
    assert G.primary_lift(1.0) == pytest.approx(0.12, abs=1e-12)
    assert G.conversion_rate(1.0) == pytest.approx(0.112, abs=1e-12)


def test_lever_destroys_value_while_raising_the_primary():
    assert G.true_value(0.0) == pytest.approx(62.0, abs=1e-9)
    assert G.value_change(1.0) == pytest.approx(-0.04752, abs=5e-5)
    assert G.value_change(1.0) < 0 < G.primary_lift(1.0)


def test_value_is_monotone_down_in_intensity():
    xs = [G.value_change(a) for a in np.linspace(0, 1, 21)]
    assert all(b <= a + 1e-12 for a, b in zip(xs, xs[1:]))


def test_the_rate_falls_three_times_as_far_as_the_total():
    ratio = G.quality_change(1.0) / G.value_change(1.0)
    assert ratio == pytest.approx(3.15, abs=0.05)
    assert G.retention_quality(0.0) == pytest.approx(0.62, abs=1e-12)


def test_a_clean_change_moves_volume_and_leaves_the_rate_alone():
    assert G.clean_value_change(0.07) == pytest.approx(0.07)


# ---------------------------------------------------------------- maturity ------------
def test_a_90_day_metric_has_no_denominator_in_a_14_day_window():
    assert G.observable_fraction(14, 90) == 0.0
    assert G.observable_fraction(180, 90) == pytest.approx(0.5)


def test_a_7_day_metric_loses_half_its_converters_at_day_14():
    assert G.observable_fraction(14, 7) == pytest.approx(0.5)


def test_maturity_is_monotone_in_the_window():
    xs = [G.observable_fraction(d, 7) for d in range(8, 200)]
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))


# ---------------------------------------------------------------- power ---------------
def test_the_experiment_is_sized_for_the_win_at_80_percent():
    assert G.n_for_power(0.80, 1.0) == N
    assert G.primary_power(1.0, N, ALPHA) == pytest.approx(0.80, abs=0.005)


def test_no_guardrail_reaches_half_the_power_of_the_win():
    powers = [G.analytic_power(g, 1.0, N, DAY, ALPHA) for g in G.GUARDRAILS]
    runnable = [p for p in powers if not np.isnan(p)]
    assert max(runnable) < 0.41
    ratio = max(runnable) / G.primary_power(1.0, N, ALPHA)
    assert ratio == pytest.approx(0.504, abs=0.02), "the best guardrail is barely half the win"
    dash = [G.analytic_power(g, 1.0, N, DAY, ALPHA) for g in G.GUARDRAILS
            if g.name in G.DASHBOARD_SUITE]
    assert max(dash) / G.primary_power(1.0, N, ALPHA) < 0.42


def test_best_guardrail_is_d7_retention_and_best_dashboard_is_tickets():
    runnable = [g for g in G.GUARDRAILS if not np.isnan(G.analytic_power(g, 1.0, N, DAY, ALPHA))]
    best = max(runnable, key=lambda g: G.analytic_power(g, 1.0, N, DAY, ALPHA))
    assert best.name == "d7_retention"
    dash = [g for g in runnable if g.name in G.DASHBOARD_SUITE]
    assert max(dash, key=lambda g: G.analytic_power(g, 1.0, N, DAY, ALPHA)).name == "support_ticket_rate"


def test_detecting_the_harm_needs_multiples_of_the_experiment():
    need = G.n_for_power(0.80, 1.0, ALPHA, G.GUARDRAIL_BY_NAME["d7_retention"], DAY)
    assert need / N == pytest.approx(3.15, abs=0.15)
    dash_need = G.n_for_power(0.80, 1.0, ALPHA, G.GUARDRAIL_BY_NAME["support_ticket_rate"], DAY)
    assert dash_need / N == pytest.approx(4.24, abs=0.15)


def test_the_placebo_guardrail_has_power_exactly_alpha():
    g = G.GUARDRAIL_BY_NAME["page_latency_ms"]
    assert G.analytic_z(g, 1.0, N, DAY) == 0.0
    assert G.analytic_power(g, 1.0, N, DAY, ALPHA) == pytest.approx(ALPHA, abs=1e-9)
    assert G.n_for_power(0.80, 1.0, ALPHA, g, DAY) is None


def test_the_d90_guardrail_cannot_be_run_at_all():
    g = G.GUARDRAIL_BY_NAME["d90_retention"]
    assert np.isnan(G.analytic_power(g, 1.0, N, DAY, ALPHA))
    assert G.n_for_power(0.80, 1.0, ALPHA, g, DAY) is None


# ---------------------------------------------------------------- simulator -----------
@pytest.fixture(scope="module")
def sims():
    rng = np.random.default_rng(11)
    return (G.simulate_experiment(0.0, N, DAY, 12_000, rng),
            G.simulate_experiment(1.0, N, DAY, 12_000, rng))


def test_every_guardrail_holds_its_false_positive_rate_under_the_null(sims):
    z0, _ = sims
    crit = G.crit_value(ALPHA)
    for g in G.GUARDRAILS:
        col = z0[g.name]
        if np.all(np.isnan(col)):
            continue
        assert float(np.mean(col > crit)) == pytest.approx(ALPHA, abs=0.012), g.name


def test_simulated_power_matches_the_analytic_model(sims):
    _, z1 = sims
    crit = G.crit_value(ALPHA)
    for g in G.GUARDRAILS:
        col = z1[g.name]
        if np.all(np.isnan(col)):
            continue
        sim = float(np.mean(col > crit))
        ana = G.analytic_power(g, 1.0, N, DAY, ALPHA)
        assert sim == pytest.approx(ana, abs=0.025), g.name


def test_the_primary_hits_its_design_power_in_simulation(sims):
    _, z1 = sims
    assert float(np.mean(z1["primary"] > G.crit_value(ALPHA))) == pytest.approx(0.80, abs=0.02)


def test_guardrails_are_near_independent_under_the_null(sims):
    """So the false-block arithmetic is exactly 1 - (1 - alpha)^k, not a hand-wave."""
    z0, _ = sims
    import itertools
    pairs = list(itertools.combinations(G.COMPUTABLE_SUITE, 2))
    worst = max(abs(np.corrcoef(z0[a], z0[b])[0, 1]) for a, b in pairs)
    assert worst < 0.05
    k = len(G.COMPUTABLE_SUITE)
    assert float(np.mean(G.any_fires(z0, G.COMPUTABLE_SUITE, ALPHA))) == pytest.approx(
        1 - (1 - ALPHA) ** k, abs=0.015)


# ---------------------------------------------------------------- decision rules ------
def test_a_harmful_change_clears_the_dashboard_a_quarter_of_the_time(sims):
    _, z1 = sims
    assert 1 - float(np.mean(G.any_fires(z1, G.DASHBOARD_SUITE, ALPHA))) == pytest.approx(0.23, abs=0.04)


def test_the_suite_blocks_a_third_of_harmless_changes(sims):
    z0, _ = sims
    assert float(np.mean(G.any_fires(z0, G.COMPUTABLE_SUITE, ALPHA))) == pytest.approx(0.336, abs=0.02)


def test_adding_a_placebo_guardrail_only_adds_false_blocks(sims):
    z0, _ = sims
    without = [s for s in G.DASHBOARD_SUITE if s != "page_latency_ms"]
    with_it = G.DASHBOARD_SUITE
    assert float(np.mean(G.any_fires(z0, with_it, ALPHA))) > float(np.mean(G.any_fires(z0, without, ALPHA)))


def test_bonferroni_holds_the_family_error_and_costs_detection(sims):
    z0, z1 = sims
    k = len(G.COMPUTABLE_SUITE)
    assert float(np.mean(G.any_fires(z0, G.COMPUTABLE_SUITE, ALPHA / k))) == pytest.approx(ALPHA, abs=0.015)
    unc = float(np.mean(G.any_fires(z1, G.COMPUTABLE_SUITE, ALPHA)))
    cor = float(np.mean(G.any_fires(z1, G.COMPUTABLE_SUITE, ALPHA / k)))
    assert unc - cor > 0.35


def test_pooling_beats_splitting_at_a_matched_false_alarm_rate(sims):
    z0, z1 = sims
    w = G.sensitivity_weights(G.COMPUTABLE_SUITE, 1.0, N, DAY)
    crit = float(np.quantile(G.composite_z(z0, G.COMPUTABLE_SUITE, w), 1 - ALPHA))
    pooled = float(np.mean(G.composite_z(z1, G.COMPUTABLE_SUITE, w) > crit))
    split = float(np.mean(G.any_fires(z1, G.COMPUTABLE_SUITE, ALPHA / len(G.COMPUTABLE_SUITE))))
    assert pooled > split + 0.30
    assert pooled == pytest.approx(0.86, abs=0.05)


def test_the_composite_null_is_calibrated_by_simulation_not_assumed(sims):
    z0, _ = sims
    w = G.sensitivity_weights(G.COMPUTABLE_SUITE, 1.0, N, DAY)
    null = G.composite_z(z0, G.COMPUTABLE_SUITE, w)
    # Calibrated, not assumed. Here it lands close to unit variance -- which is a RESULT
    # of the guardrails being near-independent, not a premise of the method.
    assert float(np.std(null)) == pytest.approx(1.0, abs=0.05)
    assert float(np.quantile(null, 1 - ALPHA)) == pytest.approx(G.crit_value(ALPHA), abs=0.12)


def test_an_unweighted_composite_still_beats_the_suite(sims):
    z0, z1 = sims
    crit = float(np.quantile(G.composite_z(z0, G.COMPUTABLE_SUITE), 1 - ALPHA))
    pooled = float(np.mean(G.composite_z(z1, G.COMPUTABLE_SUITE) > crit))
    assert pooled > float(np.mean(G.any_fires(z1, G.COMPUTABLE_SUITE, ALPHA / len(G.COMPUTABLE_SUITE))))


# ---------------------------------------------------------------- non-inferiority -----
def test_proving_non_inferiority_needs_tens_of_times_the_sample():
    g = G.GUARDRAIL_BY_NAME["d7_retention"]
    crit = G.crit_value(ALPHA)
    need = next(N * m for m in range(1, 400)
                if 1 - stats.norm.cdf(crit - G.analytic_z(g, 0.20, N * m, DAY)) >= 0.80)
    assert need / N == pytest.approx(66, abs=3)


# ---------------------------------------------------------------- observational -------
@pytest.fixture(scope="module")
def cohort():
    return G.passive_cohort(200_000, np.random.default_rng(5))


def test_correlation_and_causal_sensitivity_rank_differently(cohort):
    values, retained = cohort
    corr = {g.name: abs(np.corrcoef(values[g.name], retained)[0, 1]) for g in G.GUARDRAILS}
    sens = {g.name: G.analytic_z(g, 1.0, N, DAY) for g in G.GUARDRAILS}
    top_corr = max(corr, key=lambda k: corr[k])
    top_sens = max(sens, key=lambda k: sens[k])
    assert top_corr == "d90_retention"
    assert top_sens == "d7_retention"
    assert top_corr != top_sens
    rho, _ = stats.spearmanr([corr[g.name] for g in G.GUARDRAILS],
                             [sens[g.name] for g in G.GUARDRAILS])
    assert abs(rho) < 0.55, "the two rankings must not be interchangeable"


def test_the_best_correlate_cannot_be_measured_on_decision_day(cohort):
    values, retained = cohort
    corr = {g.name: abs(np.corrcoef(values[g.name], retained)[0, 1]) for g in G.GUARDRAILS}
    top = max(corr, key=lambda k: corr[k])
    assert G.observable_fraction(DAY, G.GUARDRAIL_BY_NAME[top].maturity_days) == 0.0


def test_the_placebo_is_uncorrelated_with_the_outcome_too(cohort):
    values, retained = cohort
    assert abs(np.corrcoef(values["page_latency_ms"], retained)[0, 1]) < 0.01


# ---------------------------------------------------------------- end to end ----------
def test_evidence_runs_and_its_headline_numbers_hold():
    import evidence

    s1 = evidence.section_1()
    assert s1["masking_ratio"] == pytest.approx(3.15, abs=0.05)
    s2 = evidence.section_2()
    assert s2["best"]["name"] == "d7_retention"
    assert s2["best"]["power"] < 0.41
    assert s2["n_per_arm"] == N


def test_a_year_of_passing_experiments_ends_with_less_than_it_started():
    import evidence

    s6 = evidence.section_6()
    dash = s6["policies"]["dashboard suite"]
    assert dash["reported_lift"] > 0.25, "the slide shows a big year"
    assert dash["value_change"] < -0.05, "and the retention rate is down"
    assert dash["mean_shipped_a"] > dash["mean_proposed_a"] + 0.05, "shipping selects for harm"
    comp = s6["policies"]["composite index"]
    assert comp["clean_block_rate"] < dash["clean_block_rate"] / 2
    assert comp["retained_change"] > dash["retained_change"]
