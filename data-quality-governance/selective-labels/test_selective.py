"""Tests for the selective-labels engine. Fast properties first, then the study's headlines from results.json."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import selective as S


def test_officers_approve_the_capacity_and_raise_the_bar_for_b() -> None:
    d = S.draw(40_000, np.random.default_rng(1))
    assert abs(d["approved"].mean() - S.APPROVE) < 1e-3
    a, b = d["approved"][d["g"] == 0].mean(), d["approved"][d["g"] == 1].mean()
    assert b < a - 0.1


def test_labels_are_a_selected_sample() -> None:
    d = S.draw(40_000, np.random.default_rng(2))
    assert d["y"][d["approved"]].mean() > d["y"].mean() + 0.2
    # approved at a low bureau score means a high unrecorded read
    low = d["x"] < -0.5
    assert d["u"][low & d["approved"]].mean() > 1.0


def test_metrics_on_a_hand_case() -> None:
    y = np.array([1, 1, 0, 1, 0, 1])
    a = np.array([1, 0, 1, 1, 0, 1], bool)
    g = np.array([0, 0, 0, 1, 1, 1])
    m = S.metrics(y, a, g)
    assert m["repay"] == pytest.approx(3 / 4)
    assert m["tpr_A"] == pytest.approx(1 / 2) and m["tpr_B"] == pytest.approx(1.0)
    w = S.metrics(y, a, g, np.array([1, 1, 3, 1, 1, 1.0]))
    assert w["repay"] == pytest.approx(3 / 6)


def test_ipw_can_pass_when_selection_is_on_recorded_data() -> None:
    """A detector that is always wrong looks like the finding; prove IPW recovers the truth in the control world."""
    r = S.one_seed(7, S.CONTROL, n=30_000)
    o = r["eval_full_model"]
    assert abs(o["ipw"]["repay"] - o["truth"]["repay"]) < 0.01
    assert abs(o["labeled"]["repay"] - o["truth"]["repay"]) > 0.01  # the naive view is still off


def test_label_training_is_harmless_when_selection_is_on_recorded_data() -> None:
    r = S.one_seed(8, S.CONTROL, n=30_000)
    assert abs(r["eval_lab_model"]["truth"]["repay"] - r["eval_full_model"]["truth"]["repay"]) < 0.015


def test_parse_world_rejects_bad_input() -> None:
    assert S.parse_world("") == S.WORLD
    for bad in ("nope=1", "sel_u=x", "sel_u", "share_b=1.5"):
        with pytest.raises(ValueError):
            S.parse_world(bad)
    assert S.parse_world("sel_u=0, bar_b=1")["bar_b"] == 1.0


R = json.load(open("results.json")) if os.path.exists("results.json") else None
needs = pytest.mark.skipif(R is None, reason="evidence.py has not been run")


@needs
def test_labeled_rows_rank_the_worse_model_first_at_every_seed() -> None:
    o, lm = R["main"]["eval_full_model"], R["main"]["eval_lab_model"]
    assert o["truth"]["repay"]["lo"] > lm["truth"]["repay"]["hi"]
    assert lm["labeled"]["repay"]["lo"] > o["labeled"]["repay"]["hi"]


@needs
def test_ipw_fails_on_unrecorded_selection_and_explore_does_not() -> None:
    o = R["main"]["eval_full_model"]
    tr = o["truth"]["repay"]["mean"]
    assert o["ipw"]["repay"]["mean"] - tr > 0.05
    assert abs(o["explore_0.1"]["repay"]["mean"] - tr) < 0.01
    assert o["rejected_bad"]["repay"]["mean"] < tr - 0.1


@needs
def test_higher_bar_turns_into_a_tilt_toward_b() -> None:
    m = R["main"]
    assert m["b_labeled_repay"]["mean"] > m["a_labeled_repay"]["mean"]
    assert m["share_b_approved"]["labeled"]["lo"] > m["share_b_approved"]["oracle"]["hi"]


@needs
def test_ipw_tpr_fails_even_in_control_for_lack_of_overlap() -> None:
    """Negative result: repay among approvals is recovered, but TPR's denominator includes never-approved payers."""
    c = R["control"]["eval_full_model"]
    assert abs(c["ipw"]["repay"]["mean"] - c["truth"]["repay"]["mean"]) < 0.01
    assert c["ipw"]["tpr_B"]["mean"] - c["truth"]["tpr_B"]["mean"] > 0.08


@needs
def test_explore_interval_shrinks_with_slice() -> None:
    ci = [R["main"]["eval_full_model"][f"explore_{e:g}"]["ci_half"]["mean"] for e in S.EXPLORE]
    assert ci == sorted(ci, reverse=True) and ci[2] < 0.03
