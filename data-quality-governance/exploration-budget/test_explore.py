"""Tests for the exploration-budget engine. Fast properties first, then the study's headlines from results.json."""

from __future__ import annotations

import json
import os

import explore as E
import numpy as np
import pytest


def test_estimators_on_a_hand_case() -> None:
    y = np.array([1, 0, 1, 1, 0, 1])
    a = np.array([1, 1, 1, 0, 0, 1], bool)
    prod = np.array([1, 1, 0, 0, 0, 0], bool)
    sl = np.array([[0, 0, 1, 0, 0, 0]], bool)  # sample one of the two challenger-only approvals
    # overlap: 1 repaid of 2; challenger-only sampled at q=0.5: one repaid, weight 2 -> (1 + 2) / 4
    assert E.targeted_est(y, a, prod, sl, 0.5)[0] == pytest.approx(3 / 4)
    assert E.uniform_est(y, a, np.array([[1, 1, 0, 0, 0, 1]], bool))[0] == pytest.approx(2 / 3)
    assert np.isnan(E.uniform_est(y, a, np.zeros((1, 6), bool))[0])


def test_both_designs_are_unbiased_for_repay() -> None:
    """A design that only looks good because it is biased toward the right answer would pass the ranking test."""
    r = E.one_seed(11, n=20_000, reps=300)
    for k in ("uniform_0.05", "targeted_0.05"):
        assert abs(r[k]["bias_oracle"]) < 0.01 and abs(r[k]["bias_label"]) < 0.01


def test_targeted_pool_repays_better_than_the_refused_rows() -> None:
    r = E.one_seed(12, n=20_000, reps=50)
    assert r["challenger_only_repay"] > r["rejected_repay"] + 0.05


def test_cheapest_returns_none_when_nothing_qualifies() -> None:
    fake = {
        f"uniform_{s:g}": {"p_right": {"lo": 0.5}, "net_cost": {"mean": 1}, "extra_loans": {"mean": 1}}
        for s in E.UNIFORM
    }
    assert E.cheapest(fake, "uniform") is None


R = json.load(open("results.json"))["main"] if os.path.exists("results.json") else None
needs = pytest.mark.skipif(R is None, reason="evidence.py has not been run")


@needs
def test_targeted_reaches_the_verdict_for_under_a_third_of_the_cost() -> None:
    u, t = E.cheapest(R, "uniform"), E.cheapest(R, "targeted")
    assert u and t and t["net_cost"] * 3 < u["net_cost"]


@needs
def test_at_matched_spend_targeted_is_more_powerful() -> None:
    u, t = R["uniform_0.005"], R["targeted_0.02"]
    assert abs(u["net_cost"]["mean"] - t["net_cost"]["mean"]) / u["net_cost"]["mean"] < 0.1
    assert t["p_right"]["lo"] > u["p_right"]["hi"]


@needs
def test_ranking_needs_far_less_than_day_200s_level_estimate() -> None:
    assert E.cheapest(R, "uniform")["share"] < 0.05


@needs
def test_targeted_fairness_bias_does_not_shrink_with_spend() -> None:
    """Negative result: more targeted money cannot reach good payers nobody would approve."""
    t5, t50, u5 = (R[k]["tpr_b_bias"]["mean"] for k in ("targeted_0.05", "targeted_0.5", "uniform_0.05"))
    assert t5 > 0.05 and t50 > 0.05 and abs(t50 - t5) < 0.01 and abs(u5) < 0.01
    assert R["unreachable_good_b"]["mean"] > 0.1
