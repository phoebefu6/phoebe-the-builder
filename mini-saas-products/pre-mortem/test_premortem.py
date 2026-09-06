"""Every headline number in the README, asserted.

The arithmetic claims are also checked against closed forms where one
exists, so a refactor cannot quietly move a number the page quotes.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import premortem as P
import pytest

# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def test_plan_shape():
    assert len(P.PLAN) == 12
    assert len(P.MODES) == 14
    assert len(P.SCALES) == 2


def test_twelve_confident_steps_are_a_coin_flip():
    assert P.plan_success() == pytest.approx(0.4485, abs=5e-4)
    assert P.weakest_step_success() == 0.88
    # No single step looks alarming; the product does.
    assert all(s.p_success >= 0.88 for s in P.PLAN)
    assert P.plan_success() < 0.5


def test_plan_success_matches_the_closed_form():
    expected = math.prod(s.p_success for s in P.PLAN)
    assert P.plan_success() == pytest.approx(expected, rel=1e-12)


def test_steps_to_coin_flip():
    assert P.steps_to_coin_flip() == 11
    avg = float(np.mean([s.p_success for s in P.PLAN]))
    assert avg ** 11 < 0.5 <= avg ** 10


def test_independence_is_the_optimistic_assumption():
    c = P.correlated_plan_success()
    assert c["correlated"] < c["independent"]
    assert c["gap"] == pytest.approx(0.0466, abs=3e-3)


# --------------------------------------------------------------------------
# Cox: the matrix cannot rank
# --------------------------------------------------------------------------


def test_default_matrix_inverts_a_quarter_of_the_pairs_it_orders():
    q = P.ranking_quality(P.SCALES[0])
    assert q["pairs"] == 91           # C(14,2)
    assert q["ordered_by_matrix"] == 71
    assert q["tied_by_matrix"] == 20
    assert q["inversions"] == 18
    assert q["inversion_rate"] == pytest.approx(0.2535, abs=1e-3)
    assert q["undecided_rate"] == pytest.approx(0.2198, abs=1e-3)


def test_the_second_scale_is_no_better():
    q = P.ranking_quality(P.SCALES[1])
    assert q["inversions"] == 20
    assert q["ordered_by_matrix"] == 66
    assert q["inversion_rate"] > P.ranking_quality(P.SCALES[0])["inversion_rate"]


def test_pair_count_is_the_binomial():
    assert P.ranking_quality(P.SCALES[0])["pairs"] == len(
        list(itertools.combinations(P.MODES, 2)))


def test_an_inversion_really_is_backwards():
    """Every reported inversion: higher score, strictly lower expected loss."""
    scale = P.SCALES[0]
    for hi, lo, ratio in P.inversions(scale):
        a = next(m for m in P.MODES if m.id == hi)
        b = next(m for m in P.MODES if m.id == lo)
        assert scale.score(a) > scale.score(b)
        assert a.expected_loss < b.expected_loss
        assert ratio == pytest.approx(b.expected_loss / a.expected_loss)


def test_the_biggest_expected_loss_is_ranked_eighth_by_the_matrix():
    scale = P.SCALES[0]
    worst = P.by_expected_loss()[0]
    assert worst.id == "F06"
    assert worst.expected_loss == 320_000
    assert [m.id for m in P.by_matrix(scale)].index("F06") == 7      # 8th
    assert [m.id for m in P.by_prevention_value()].index("F06") == 0  # 1st
    # Low probability, high loss - the band throws the loss away.
    assert scale.cell(worst) == (2, 4)
    assert scale.score(worst) == 8


# --------------------------------------------------------------------------
# Compression and ordinal arithmetic
# --------------------------------------------------------------------------


def test_risks_share_cells_and_become_indistinguishable():
    c = P.range_compression(P.SCALES[0])
    assert c["occupied_cells"] == 7
    assert c["shared_cells"] == 4
    assert c["worst_cell"] == (3, 3)
    assert c["worst_pair"] == ("F03", "F11")
    assert c["worst_ratio"] == pytest.approx(2.571, abs=1e-3)


def test_twentyfive_cells_collapse_to_fourteen_scores():
    o = P.ordinal_product_is_meaningless(P.SCALES[0])
    assert o["cells"] == 25
    assert o["distinct_scores"] == 14
    assert o["colliding_scores"] == 10
    assert o["example"] == (12, [(3, 4), (4, 3)])


def test_band_multiplication_has_no_unit():
    """Two cells scoring 12 whose expected losses differ by a factor of two."""
    scale = P.SCALES[0]
    a = P.FailureMode("A", "x" * 30, 0.30, 2_000_000, 1.0, 0.5)
    b = P.FailureMode("B", "x" * 30, 0.60, 500_000, 1.0, 0.5)
    assert scale.score(a) == scale.score(b) == 12
    assert a.expected_loss == pytest.approx(2 * b.expected_loss)


# --------------------------------------------------------------------------
# Two scales, two answers
# --------------------------------------------------------------------------


def test_two_conventional_scales_pick_different_top_risks():
    d = P.scale_disagreement()
    assert d["n_flips"] == 13
    assert d["top_by_a"] == "F08"
    assert d["top_by_b"] == "F03"
    assert d["same_top"] is False


def test_scale_bands_are_monotone():
    for sc in P.SCALES:
        assert list(sc.p_edges) == sorted(sc.p_edges)
        assert list(sc.loss_edges) == sorted(sc.loss_edges)
        # One-indexed: a band of 0 would zero the product and make the
        # lowest-likelihood row score 0 at every impact.
        assert sc.p_band(0.0) == 1
        assert sc.loss_band(0.0) == 1
        assert sc.p_band(1.0) == len(sc.p_edges) + 1
        assert min(sc.score(m) for m in P.MODES) > 0


# --------------------------------------------------------------------------
# The orderings
# --------------------------------------------------------------------------


def test_the_matrix_order_is_not_the_prevention_order():
    d = P.ordering_disagreement()
    assert d["matrix_equals_prevention"] is False
    assert d["matrix_top3"] == ["F08", "F01", "F03"]
    assert d["expected_loss_top3"] == ["F06", "F08", "F01"]
    assert d["prevention_top3"] == ["F06", "F01", "F03"]


def test_every_prevention_is_worth_buying_in_this_register():
    """No negative-value preventions here, so the disagreement is pure ordering."""
    assert P.ordering_disagreement()["negative_value_modes"] == []
    assert all(m.prevention_value > 0 for m in P.MODES)


def test_prevention_value_and_ratio_are_different_orders():
    assert [m.id for m in P.by_prevention_value()] != [m.id for m in P.by_prevention_ratio()]


def test_expected_loss_arithmetic():
    assert P.total_expected_loss() == pytest.approx(2_018_000)
    for m in P.MODES:
        assert m.expected_loss == pytest.approx(m.probability * m.loss)
        assert m.prevention_value == pytest.approx(
            m.probability * m.prevention_effect * m.loss - m.prevention_cost)


# --------------------------------------------------------------------------
# It is a knapsack
# --------------------------------------------------------------------------


def test_optimal_beats_or_matches_both_heuristics_everywhere():
    for r in P.allocation_comparison():
        assert r["optimal"] >= r["matrix"] - 1e-6, r
        assert r["optimal"] >= r["ratio"] - 1e-6, r


def test_the_matrix_order_leaves_a_third_of_the_benefit_unbought():
    rows = {r["budget"]: r for r in P.allocation_comparison()}
    r = rows[100_000]
    assert r["matrix"] == pytest.approx(443_200)
    assert r["optimal"] == pytest.approx(685_000)
    assert r["matrix_shortfall"] / r["optimal"] > 0.34


def test_greedy_by_ratio_loses_to_the_matrix_at_the_tightest_budget():
    """The honest part: neither heuristic is reliable."""
    rows = {r["budget"]: r for r in P.allocation_comparison()}
    assert rows[50_000]["ratio_beats_matrix"] is False
    assert rows[50_000]["ratio"] < rows[50_000]["matrix"]
    # ... and it wins at every larger budget tested.
    for b in (100_000, 150_000, 200_000):
        assert rows[b]["ratio_beats_matrix"] is True


def test_optimal_allocation_respects_the_budget():
    for b in (25_000, 50_000, 100_000):
        res = P.optimal_allocation(b)
        cost = sum(m.prevention_cost for m in P.MODES if m.id in res["bought"])
        assert cost <= b


def test_optimal_allocation_is_not_a_prefix_of_the_matrix_order():
    """No ordering finds it: the best set is not the top-k of the matrix."""
    best = set(P.optimal_allocation(100_000)["bought"])
    matrix_prefix = {m.id for m in P.by_matrix(P.SCALES[0])[: len(best)]}
    assert best != matrix_prefix
    # It is reachable by working down the ratio order under this budget,
    # which is exactly why the ratio heuristic wins at 100k and loses at 50k.
    assert len(best) >= 4


# --------------------------------------------------------------------------
# The linter
# --------------------------------------------------------------------------


def test_half_the_raw_notes_are_not_actionable():
    r = P.notes_report()
    assert r["n"] == 8
    assert r["actionable"] == 4
    assert r["per_field"] == {
        "has_mechanism": 4, "has_probability": 6, "has_loss": 5,
        "has_prevention_cost": 4, "has_prevention_effect": 4,
    }
    assert r["vague"] == ["Data quality issues", "Key person risk",
                          "Budget overrun", "Scope creep"]


def test_a_category_is_not_a_mechanism():
    assert P.lint({"cause": "Data quality issues"})["has_mechanism"] is False
    long = "Metric definitions silently differ between the two engines"
    assert P.lint({"cause": long})["has_mechanism"] is True


def test_linter_rejects_a_certainty_and_a_zero():
    for p in (0.0, 1.0):
        assert P.lint({"cause": "x" * 30, "probability": p})["has_probability"] is False


def test_a_note_with_probability_and_loss_is_still_not_actionable():
    """Two of four numbers is the usual state of a risk register."""
    note = {"cause": "Query costs land 3x over budget and finance halts it",
            "probability": 0.3, "loss": 600_000}
    assert P.lint(note)["has_probability"] and P.lint(note)["has_loss"]
    assert not P.actionable(note)
