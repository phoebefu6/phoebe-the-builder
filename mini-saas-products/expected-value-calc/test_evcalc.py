"""Every headline number in the README, asserted.

Closed forms are checked against the simulation wherever one exists, so a
Monte Carlo drift cannot quietly move a number the page quotes.
"""

from __future__ import annotations

import math

import evcalc as E
import numpy as np
import pytest

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def test_shape():
    assert len(E.INPUTS) == 4
    assert E.OPTIONS == ("build", "buy", "defer")
    sims = E.simulate()
    assert len(sims["build"]) == E.N_SIMS


def test_simulation_is_deterministic():
    a = E.simulate()["build"].copy()
    E.simulate.cache_clear()
    assert np.allclose(a, E.simulate()["build"])


def test_inputs_are_ordered_ranges():
    for i in E.INPUTS:
        assert i.low < i.mid < i.high, i.name


def test_defer_is_exactly_zero():
    assert np.all(E.simulate()["defer"] == 0.0)


# --------------------------------------------------------------------------
# Two averaging errors
# --------------------------------------------------------------------------


def test_the_typed_middle_is_not_the_mean():
    shifts = E.mode_vs_mean()
    assert shifts["build_months"]["actual_mean"] == pytest.approx(7.01, abs=0.05)
    assert shifts["build_months"]["shift"] > 0.9
    assert shifts["seats"]["actual_mean"] == pytest.approx(38.4, abs=0.3)
    # A symmetric range shifts almost not at all - the effect is skew, not noise.
    assert abs(shifts["hourly_cost"]["shift"]) < 0.1


def test_jensen_is_measured_at_the_means_not_the_mode():
    """Otherwise the two errors are confounded and the gap means nothing."""
    fa = E.flaw_of_averages()
    at_means = E.estimate_at_input_means()
    for opt in E.OPTIONS:
        assert fa[opt]["at_input_means"] == pytest.approx(at_means[opt])
        assert fa[opt]["jensen_gap"] == pytest.approx(at_means[opt] - fa[opt]["true_ev"])


def test_the_famous_error_is_the_smaller_one_here():
    """`build` is near-linear over the range; typing the mode is the whole problem."""
    fa = E.flaw_of_averages()
    whole = fa["build"]["true_ev"] - fa["build"]["elicited_mid_estimate"]
    assert whole == pytest.approx(60_769, rel=0.02)
    assert abs(fa["build"]["jensen_gap"]) < 0.01 * abs(whole)
    # `buy` does have curvature: the seat cap is a min().
    assert fa["buy"]["jensen_overstates_by"] == pytest.approx(0.021, abs=0.005)


def test_the_typed_estimate_almost_never_happens():
    p = E.probability_of_the_point_estimate()
    assert p["build"] < 0.02
    assert p["buy"] < 0.05


# --------------------------------------------------------------------------
# Ranking the mean
# --------------------------------------------------------------------------


def test_expected_values():
    ev = E.true_expected_value()
    assert ev["build"] == pytest.approx(128_909, rel=0.02)
    assert ev["buy"] == pytest.approx(99_933, rel=0.02)
    assert ev["defer"] == 0.0


def test_the_higher_ev_option_loses_more_often_than_it_wins():
    """The headline conflict. Both statements are true of the same numbers."""
    c = E.ranking_conflict()
    assert c["best_by_ev"] == "build"
    assert c["pairs"][("build", "buy")] == pytest.approx(0.493, abs=0.01)
    assert c["pairs"][("build", "buy")] < 0.5
    assert len(c["conflicts"]) == 1
    a, b, gap, p = c["conflicts"][0]
    assert (a, b) == ("build", "buy") and gap > 0 and p < 0.5


def test_pairwise_probabilities_are_complementary():
    for a in E.OPTIONS:
        for b in E.OPTIONS:
            if a == b:
                continue
            assert E.beats(a, b) + E.beats(b, a) == pytest.approx(1.0, abs=1e-6)


def test_build_carries_the_longer_tail_and_the_bigger_downside():
    b, y = E.downside("build"), E.downside("buy")
    assert b["p90"] > y["p90"]        # longer right tail lifts the mean
    assert b["p10"] < y["p10"]        # and a worse floor
    assert b["p_loss"] == pytest.approx(0.348, abs=0.01)
    assert y["p_loss"] == pytest.approx(0.174, abs=0.01)
    assert b["median"] < y["median"]  # which is why it wins less than half


# --------------------------------------------------------------------------
# Sensitivity
# --------------------------------------------------------------------------


def test_adoption_dominates_and_the_hourly_rate_barely_matters():
    rows = E.tornado()
    assert rows[0][0] == "seats"
    assert rows[-1][0] == "hourly_cost"
    assert rows[0][3] / rows[-1][3] > 10


def test_the_tornado_overstates_the_spread():
    """It lines up worst cases that rarely co-occur; it is not a distribution."""
    ish = E.interaction_share()
    assert ish["ratio"] < 1.0
    assert ish["ratio"] == pytest.approx(0.289, abs=0.02)


def test_every_switching_point_is_inside_the_plausible_range():
    for name, sp in E.switching_points().items():
        assert sp is not None, name
        i = E.INPUTS_BY_NAME[name]
        assert i.low < sp < i.high, name


def test_the_decision_flips_within_a_rounding_error_of_the_estimate():
    """The recommendation is balanced exactly where the estimate sits."""
    sp = E.switching_points()
    assert sp["seats"] == pytest.approx(33.27, abs=0.3)
    assert abs(sp["seats"] - E.INPUTS_BY_NAME["seats"].mid) < 2
    assert sp["build_months"] == pytest.approx(5.63, abs=0.2)


def test_the_switching_point_actually_switches():
    """Either side of it, the recommendation is different."""
    sp = E.switching_points()["seats"]
    base = E.midpoints()
    for delta, expected in ((-3.0, "buy"), (+3.0, "build")):
        args = {k: np.array([v]) for k, v in base.items()}
        args["seats"] = np.array([sp + delta])
        gap = (E.value_of_option("build", **args)[0]
               - E.value_of_option("buy", **args)[0])
        assert (gap > 0) == (expected == "build"), (delta, gap)


# --------------------------------------------------------------------------
# Repeating the bet
# --------------------------------------------------------------------------


def test_positive_ensemble_growth_and_negative_time_average():
    assert E.ensemble_growth() == pytest.approx(1.05)
    assert E.time_average_growth() == pytest.approx(math.sqrt(E.UP * E.DOWN), rel=1e-12)
    assert E.time_average_growth() == pytest.approx(0.9487, abs=1e-4)
    assert E.ensemble_growth() > 1.0 > E.time_average_growth()


def test_maximising_ev_per_round_wipes_out_the_median_run():
    """Not literally zero - about two millionths of the stake. Say the number."""
    full = E.trajectories(fraction=1.0)
    assert full["mean"] > 100
    assert 0 < full["median"] < 1e-4
    assert full["p_ruin_99pct"] == pytest.approx(0.89, abs=0.02)


def test_kelly_matches_its_closed_form_and_grows_the_median():
    b, a = E.UP - 1.0, 1.0 - E.DOWN
    assert E.kelly_fraction() == pytest.approx((E.P_UP * b - (1 - E.P_UP) * a) / (a * b))
    assert E.kelly_fraction() == pytest.approx(0.25)
    kelly = E.trajectories(fraction=E.kelly_fraction())
    assert kelly["median"] > 1.0
    assert kelly["p_ruin_99pct"] < 0.01


def test_kelly_beats_full_staking_on_median_and_loses_on_mean():
    """Exactly the trade the two averages describe."""
    full = E.trajectories(fraction=1.0)
    kelly = E.trajectories(fraction=E.kelly_fraction())
    assert kelly["median"] > full["median"]
    assert kelly["mean"] < full["mean"]


# --------------------------------------------------------------------------
# Value of information
# --------------------------------------------------------------------------


def test_evpi_is_positive_and_material():
    v = E.evpi()
    assert v["with_perfect_information"] > v["best_without_information"]
    assert v["evpi"] == pytest.approx(67_214, rel=0.03)
    assert v["evpi"] / v["best_without_information"] > 0.4


def test_information_about_one_input_is_never_worth_more_than_all_of_it():
    v = E.evpi()["evpi"]
    for name in (i.name for i in E.INPUTS):
        assert 0 <= E.evppi(name) <= v + 1e-6, name


def test_information_is_not_additive():
    info = {k: val for k, val in E.information_value().items() if not k.startswith("_")}
    assert sum(info.values()) > E.evpi()["evpi"]


def test_studying_the_right_input_is_worth_sixty_times_the_wrong_one():
    info = {k: v for k, v in E.information_value().items() if not k.startswith("_")}
    assert max(info, key=info.get) == "seats"
    assert min(info, key=info.get) == "hourly_cost"
    assert info["seats"] / max(info["hourly_cost"], 1.0) > 50


def test_the_input_that_swings_most_is_the_one_worth_learning():
    """Tornado ordering and information value agree at the top, not everywhere."""
    assert E.tornado()[0][0] == "seats"
    info = {k: v for k, v in E.information_value().items() if not k.startswith("_")}
    assert max(info, key=info.get) == "seats"
