"""Every headline number in the README, asserted.

The propriety results are also checked against their closed forms, so the
grid search cannot quietly agree with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import declog as D

# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_shape_of_the_experiment():
    assert len(D.RULES) == 6
    assert len(D.FORECASTERS) == 6
    assert len(D.RECORDS) == 20
    outcomes, reports = D.simulate()
    assert len(outcomes) == 4000
    assert set(reports) == {f.name for f in D.FORECASTERS}


def test_simulation_is_deterministic():
    a_out, a_rep = D.simulate()
    D.simulate.cache_clear()
    b_out, b_rep = D.simulate()
    assert np.array_equal(a_out, b_out)
    for name in a_rep:
        assert np.allclose(a_rep[name], b_rep[name])


# --------------------------------------------------------------------------
# Propriety - the core claim
# --------------------------------------------------------------------------


def test_exactly_three_rules_are_proper():
    proper = sorted(r.name for r in D.RULES if D.propriety(r.name)[0])
    improper = sorted(r.name for r in D.RULES if not D.propriety(r.name)[0])
    assert proper == ["brier", "log", "spherical"]
    assert improper == ["absolute", "confidence_points", "threshold_01"]


def test_proper_rules_have_zero_gap_at_every_belief():
    """Zero to floating point - the grid optimum lands exactly on p."""
    for name in ("brier", "log", "spherical"):
        ok, gap, bad = D.propriety(name)
        assert ok and bad == ()
        assert gap < 1e-12, (name, gap)


def test_improper_rules_misreport_every_single_belief():
    for name in ("absolute", "threshold_01", "confidence_points"):
        ok, gap, bad = D.propriety(name)
        assert not ok
        assert gap == pytest.approx(0.5, abs=1e-9)
        assert len(bad) == 99


def test_brier_optimum_matches_its_closed_form():
    """E[L] = p(q-1)^2 + (1-p)q^2, minimised at q = p. Checked analytically."""
    for p in (0.05, 0.3, 0.55, 0.7, 0.99):
        assert D.optimal_report(D.RULES_BY_NAME["brier"], p) == pytest.approx(p, abs=1e-3)


def test_log_optimum_matches_its_closed_form():
    """E[L] = -p ln q - (1-p) ln(1-q), minimised at q = p."""
    for p in (0.1, 0.4, 0.62, 0.9):
        assert D.optimal_report(D.RULES_BY_NAME["log"], p) == pytest.approx(p, abs=1e-3)


def test_absolute_and_points_pay_for_certainty():
    """Believe anything above a coin flip, report 1.00, score better."""
    for name in ("absolute", "confidence_points"):
        for p in (0.55, 0.6, 0.7, 0.8, 0.9):
            assert D.optimal_report(D.RULES_BY_NAME[name], p) == 1.0


def test_the_two_improper_rules_fail_differently_below_a_coin_flip():
    """`absolute` collapses to 0. The points game stakes the most it can.

    `confidence_points` wagers the number written down, so below 0.5 the
    optimum is the largest stake still on the favoured side - 0.499, not 0.
    Different mechanism, same result: the report is never the belief.
    """
    for p in (0.1, 0.3, 0.45):
        assert D.optimal_report(D.RULES_BY_NAME["absolute"], p) == 0.0
        assert D.optimal_report(D.RULES_BY_NAME["confidence_points"], p) == pytest.approx(
            0.499, abs=1e-9
        )


def test_honest_reporting_is_strictly_punished_under_absolute():
    """The concrete indictment: honesty scores worse than a confident lie."""
    absolute = D.RULES_BY_NAME["absolute"]
    p = 0.55
    honest = D.expected_loss(absolute, p)[np.argmin(np.abs(D.GRID - p))]
    lie = D.expected_loss(absolute, p)[-1]
    assert lie < honest


def test_threshold_rule_is_blind_across_half_its_range():
    lo, hi, width = D.optimal_report_set(D.RULES_BY_NAME["threshold_01"], 0.7)
    assert (lo, hi) == (0.5, 1.0)
    assert width == pytest.approx(0.5)
    # Every proper rule has a single point optimum.
    for name in ("brier", "log", "spherical"):
        assert D.optimal_report_set(D.RULES_BY_NAME[name], 0.7)[2] == 0.0


# --------------------------------------------------------------------------
# Ranking instability
# --------------------------------------------------------------------------


def test_proper_rules_all_crown_the_calibrated_forecaster():
    for name in ("brier", "log", "spherical"):
        assert D.ranking(name)[0] == "calibrated"


def test_average_error_crowns_the_overconfident_forecaster():
    """The spreadsheet rule promotes the person who is most often wrongly sure."""
    assert D.ranking("absolute")[0] == "overconfident"
    assert D.ranking("absolute")[1] == "calibrated"


def test_the_points_game_crowns_the_underconfident_forecaster():
    """Two homebrew rules, two opposite wrong answers."""
    assert D.ranking("confidence_points")[0] == "underconfident"


def test_five_rules_put_the_base_rate_forecaster_last_and_log_does_not():
    """Log loss is unbounded, and that reverses a ranking.

    `noisy_expert` has real information - unbiased, just inconsistent - and
    makes 141 confident misses in 4000. Under log loss those dominate the
    average so completely that it finishes BELOW the forecaster that knows
    nothing and never commits. An unbounded rule can rank information below
    the absence of information.
    """
    last = {r.name: D.ranking(r.name)[-1] for r in D.RULES}
    assert last["log"] == "noisy_expert"
    assert [n for n, v in last.items() if v == "base_rate"] == [
        "brier", "spherical", "absolute", "threshold_01", "confidence_points"
    ]
    t = D.score_table()
    assert t["noisy_expert"]["brier"] < t["base_rate"]["brier"]   # better by Brier
    assert t["noisy_expert"]["log"] > t["base_rate"]["log"]       # worse by log


def test_worst_rule_pair_flips_six_of_fifteen_pairings():
    rd = D.ranking_disagreement()
    off = {k: v for k, v in rd.items() if k[0] != k[1]}
    assert max(off.values()) == 6
    assert rd[("absolute", "log")] == 6
    # A rule never disagrees with itself.
    assert all(rd[(r.name, r.name)] == 0 for r in D.RULES)


# --------------------------------------------------------------------------
# Murphy decomposition
# --------------------------------------------------------------------------


def test_base_rate_forecaster_is_perfectly_calibrated_and_worthless():
    d = D.decompositions()["base_rate"]
    assert d["reliability"] == pytest.approx(0.0, abs=1e-6)
    assert d["resolution"] == pytest.approx(0.0, abs=1e-6)
    assert d["brier"] == max(D.decompositions()[f.name]["brier"] for f in D.FORECASTERS)


def test_more_reliable_can_still_score_worse():
    """If this is empty, 'improve your calibration' would be sound advice."""
    flips = D.reliability_beats_resolution()
    assert flips
    assert ("noisy_expert", "optimist") in flips
    assert ("base_rate", "calibrated") in flips


def test_decomposition_residual_is_small_and_reported():
    """Binning makes the identity approximate; it must not be hidden."""
    for f in D.FORECASTERS:
        m = D.decompositions()[f.name]
        assert abs(m["check"] - m["brier"]) < 0.005, f.name


def test_uncertainty_is_shared_by_every_forecaster():
    """It is a property of the events, not of the forecaster."""
    uncs = {round(D.decompositions()[f.name]["uncertainty"], 12) for f in D.FORECASTERS}
    assert len(uncs) == 1


# --------------------------------------------------------------------------
# Resulting
# --------------------------------------------------------------------------


def test_a_good_decision_loses_often():
    r = D.resulting()
    assert r["expected_value"] > 0
    assert r["share_judged_bad"] == pytest.approx(0.379, abs=0.01)


def test_outcome_based_review_is_wrong_four_times_in_ten():
    p = D.resulting_portfolio()
    assert p["n"] == 200
    assert p["misjudged"] == 83
    assert p["misjudged_rate"] == pytest.approx(0.415, abs=1e-9)
    assert p["good_called_bad"] == 44
    assert p["bad_called_good"] == 39
    assert p["good_called_bad"] + p["bad_called_good"] == p["misjudged"]


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------


def test_ranking_forecasters_needs_more_decisions_than_anyone_logs():
    m = D.power_matrix()
    assert len(m) == 15
    assert int(np.median(list(m.values()))) == 461
    lo, ln, hi, hn = D.cheapest_and_dearest_comparison()
    assert lo == ("underconfident", "base_rate") and int(ln) == 65
    assert hi == ("optimist", "noisy_expert") and int(hn) == 53520


def test_a_five_year_log_resolves_only_a_minority_of_comparisons():
    """One decision a week for five years is 260 records."""
    m = D.power_matrix()
    resolvable = sum(1 for n in m.values() if n <= 260)
    assert resolvable < len(m) / 2
    assert resolvable == 4


# --------------------------------------------------------------------------
# The linter
# --------------------------------------------------------------------------


def test_half_the_corpus_is_unscoreable():
    rep = D.resolvability_report()
    assert rep["n"] == 20
    assert rep["resolvable"] == 10
    assert rep["per_field"] == {
        "has_probability": 12, "has_resolution_date": 12,
        "has_metric": 11, "has_threshold": 10,
    }


def test_every_unscoreable_record_has_a_scoreable_twin():
    """The corpus pairs each vague record with the same decision written properly."""
    rep = D.resolvability_report()
    assert len(rep["unscoreable"]) == 10
    by_id = {r.id: r for r in D.RECORDS}
    for bad_id in rep["unscoreable"]:
        twin_id = f"D-{int(bad_id.split('-')[1]) + 1:03d}"
        assert D.resolvable(by_id[twin_id])
        assert by_id[twin_id].decision == by_id[bad_id].decision


def test_linter_rejects_a_vague_resolution_date():
    r = D.Record("X", "d", "c", 0.6, "Q4", "m", "> 1")
    assert D.lint(r)["has_resolution_date"] is False
    assert not D.resolvable(r)


def test_linter_rejects_a_certainty():
    """0 and 1 are not forecasts; nothing can update them and log loss is infinite."""
    for p in (0.0, 1.0):
        r = D.Record("X", "d", "c", p, "2027-01-01", "m", "> 1")
        assert D.lint(r)["has_probability"] is False


# --------------------------------------------------------------------------
# Rules behave as scoring rules
# --------------------------------------------------------------------------


def test_every_loss_is_minimised_by_a_correct_confident_call():
    """Sanity: predicting 1 for an event that happens is the best possible case."""
    y1, y0 = np.array([1.0]), np.array([0.0])
    for r in D.RULES:
        best = r.loss(np.array([0.999]), y1)[0]
        worst = r.loss(np.array([0.001]), y1)[0]
        assert best < worst, r.name
        assert r.loss(np.array([0.001]), y0)[0] < r.loss(np.array([0.999]), y0)[0], r.name


def test_log_loss_is_the_only_unbounded_rule():
    unbounded = [r.name for r in D.RULES if not r.bounded]
    assert unbounded == ["log"]
    huge = D.log_loss(np.array([1e-9]), np.array([1.0]))[0]
    assert huge > 20
