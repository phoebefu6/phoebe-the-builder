"""Tests for the exact escalation engine. Each property is one a plausible bug would break."""

from __future__ import annotations

import escalation as E
import numpy as np
import pytest


def test_tables_are_distributions_and_accuracies_are_probabilities() -> None:
    for seg in (E.SEGMENTS, {s: dict(v, inflate=E.INFLATE_V2) for s, v in E.SEGMENTS.items()}):
        for s, t in E.segment_tables(seg).items():
            assert abs(t["pmf"].sum() - 1) < 1e-12, s
            assert (t["p_true"] >= 0).all() and (t["p_true"] <= 1).all(), s


def test_inflation_changes_reported_confidence_not_skill() -> None:
    """The upgrade is defined as same skill, louder numbers: overall accuracy must be identical."""
    v2 = {s: dict(v, inflate=E.INFLATE_V2) for s, v in E.SEGMENTS.items()}
    a, b = E.segment_tables(), E.segment_tables(v2)
    for s in E.SEGS:
        acc1 = float((a[s]["pmf"] * a[s]["p_true"]).sum())
        acc2 = float((b[s]["pmf"] * b[s]["p_true"]).sum())
        conf1 = float((a[s]["pmf"] * E.CENTERS).sum())
        conf2 = float((b[s]["pmf"] * E.CENTERS).sum())
        assert abs(acc1 - acc2) < 1e-9, s
        assert conf2 > conf1 + 0.02, s


def test_calibration_shape() -> None:
    """faq is calibrated (accuracy ~ reported confidence); refund is the most overconfident."""
    t = E.segment_tables()
    gap = {s: float((t[s]["pmf"] * (E.CENTERS - t[s]["p_true"])).sum()) for s in E.SEGS}
    assert abs(gap["faq"]) < 0.01
    assert gap["refund"] > gap["account"] > gap["faq"]


def test_never_escalate_costs_exactly_the_model_errors() -> None:
    r = E.evaluate({s: 0.0 for s in E.SEGS})
    t = E.segment_tables()
    want = sum(E.BASE["tickets"] * E.SEGMENTS[s]["share"] * float((t[s]["pmf"] * (1 - t[s]["p_true"])).sum())
               * E.SEGMENTS[s]["cost"] for s in E.SEGS)
    assert r["escalated"] == 0 and r["review_cost"] == 0
    assert abs(r["cost"] - want) < 1e-6


def test_escalate_everything_under_infinite_capacity_is_all_human() -> None:
    prm = dict(E.BASE, capacity=1e9)
    r = E.evaluate({s: 1.0 for s in E.SEGS}, prm)
    assert abs(r["escalated"] - prm["tickets"]) < 1e-6
    assert abs(r["error_rate"] - prm["human_err"]) < 1e-9


def test_human_error_curve() -> None:
    p = E.BASE
    assert E.human_error(0.0, p) == p["human_err"]
    assert E.human_error(p["calm_until"], p) == p["human_err"]
    assert abs(E.human_error(1.0, p) - p["human_err_full"]) < 1e-12
    assert E.human_error(2.0, p) == E.human_error(1.0, p)


def test_overflow_reviews_exactly_capacity() -> None:
    r = E.evaluate({s: 0.95 for s in E.SEGS})
    assert r["escalated"] > E.BASE["capacity"]
    assert abs(r["escalated"] * r["reviewed_share"] - E.BASE["capacity"]) < 1e-6
    assert r["utilisation"] == 1.0


def test_chow_threshold_is_where_a_review_breaks_even() -> None:
    """At the cutoff, (1-p)*cost == review + human_err*cost; just below it a review pays."""
    prm = E.BASE
    for s in E.SEGS:
        c, g = E.SEGMENTS[s]["cost"], E.SEGMENTS[s]["gamma"]
        t = E.chow_threshold(c, prm, g)
        if t == 0.0:
            assert prm["review_cost"] + prm["human_err"] * c >= c
            continue
        assert abs((1 - t ** g) * c - (prm["review_cost"] + prm["human_err"] * c)) < 1e-9


def test_faq_review_never_pays() -> None:
    v = E.value_curve()["faq"]
    assert all(x < 0 for x in v if x == x)


def test_joint_optimum_beats_every_other_policy() -> None:
    rows = {r["policy"]: r["cost"] for r in E.policy_table()}
    best = rows.pop("priced + capacity (joint)")
    assert all(best < c for c in rows.values())


def test_headline_numbers() -> None:
    rows = {r["policy"]: r for r in E.policy_table()}
    never, acc = rows["never escalate"], rows["accuracy rule: conf < 0.90"]
    one, cal, joint = rows["best single threshold"], rows["priced, calibrated"], rows["priced + capacity (joint)"]
    assert acc["cost"] > never["cost"], "the accuracy rule should cost more than doing nothing"
    assert cal["cost"] > one["cost"], "calibrated pricing without capacity should lose to one cutoff"
    assert joint["cost"] / one["cost"] - 1 == pytest.approx(-0.194, abs=0.002)
    assert joint["esc_by_seg"]["faq"] == 0 and joint["esc_by_seg"]["account"] < 1
    assert joint["utilisation"] < 0.85, "the optimum stops short of filling the team"


def test_load_curve_has_an_interior_minimum() -> None:
    c = E.load_curve()
    i = int(np.argmin([r["cost"] for r in c]))
    assert 0 < i < len(c) - 1
    assert c[i]["utilisation_demand"] < 1.0
    assert c[-1]["cost"] > c[i]["cost"] * 1.3


def test_upgrade_cuts_escalations_and_raises_cost_and_refit_recovers() -> None:
    u = E.upgrade()
    assert u["after"]["escalation_rate"] < u["before"]["escalation_rate"] / 3
    assert u["after"]["cost"] / u["before"]["cost"] - 1 == pytest.approx(0.267, abs=0.003)
    assert abs(u["refit"]["cost"] - u["before"]["cost"]) / u["before"]["cost"] < 0.005


def test_simulation_matches_exact_inside_wilson() -> None:
    cal = E.calibrate()
    assert len(cal) == 21
    assert all(r["inside"] for r in cal)


def test_simulation_check_can_fail() -> None:
    """A verifier must fail on a wrong answer: shift the exact rate and it should leave the interval."""
    thr = E.policies()["priced + capacity (joint)"]
    mc = E.simulate(thr)
    k, n = mc["refund"]["errors"], mc["refund"]["tickets"]
    lo, hi = E.wilson(k, n)
    exact = E.evaluate(thr)["errors"]["refund"] / (E.BASE["tickets"] * E.SEGMENTS["refund"]["share"])
    assert lo <= exact <= hi
    assert not (lo <= exact + 0.02 <= hi)


@pytest.mark.parametrize("bad", ["nope=1", "capacity", "capacity=x", "capacity=0", "human_err=0.5"])
def test_parse_params_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        E.parse_params(bad)


def test_parse_params_overrides() -> None:
    p = E.parse_params("capacity=1500, review_cost=10")
    assert p["capacity"] == 1500 and p["review_cost"] == 10 and p["tickets"] == E.BASE["tickets"]
