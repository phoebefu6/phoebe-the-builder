"""Tests for the audit engine. Fast properties first, then the study's headlines from results.json."""

from __future__ import annotations

import json
import os

import audit as A
import pytest


def test_quota_matches_every_marginal_but_not_the_cells() -> None:
    p, q = A.pop_shares(), A.QUOTA
    assert abs(sum(q.values()) - 1) < 1e-9
    assert abs((q["B-urban"] + q["B-rural"]) - (p["B-urban"] + p["B-rural"])) < 1e-9
    assert abs((q["A-rural"] + q["B-rural"]) - (p["A-rural"] + p["B-rural"])) < 1e-9
    assert q["B-rural"] / p["B-rural"] < 0.1


def test_sample_draws_exact_cell_counts() -> None:
    d = A.sample(A.QUOTA, 20_000, 1)
    counts = {c: int((d["cell"] == i).sum()) for i, c in enumerate(A.CELLS)}
    assert counts == {"A-urban": 6_100, "A-rural": 7_900, "B-urban": 5_900, "B-rural": 100}


def test_thin_file_only_shifts_rural_b_history() -> None:
    d = A.sample(A.pop_shares(), 40_000, 2)
    m = lambda i: d["history"][d["cell"] == i].mean()  # noqa: E731
    assert abs(m(0)) < 0.05 and abs(m(1)) < 0.05 and abs(m(2)) < 0.05
    assert abs(m(3) + A.WORLD["thin_file"]) < 0.08


def test_audit_flags_cells_and_passes_marginals() -> None:
    au = A.audit(A.sample(A.QUOTA, 20_000, 3), A.pop_shares())
    assert not au["group B"]["flag"] and not au["region rural"]["flag"]
    assert au["B-rural"]["flag"] and au["B-rural"]["ratio"] < 0.1


def test_detector_can_pass_a_proportional_sample() -> None:
    """A detector that always fires looks identical to one that works on the biased input; prove it can say ok."""
    for seed in (4, 5, 6):
        au = A.audit(A.sample(A.pop_shares(), 20_000, seed), A.pop_shares())
        assert not any(v["flag"] for v in au.values())


def test_small_segment_is_not_flagged_on_noise() -> None:
    """Ratio outside the band but a Wilson interval that still covers the population share: no flag."""
    skew = {"A-urban": 0.46, "A-rural": 0.30, "B-urban": 0.20, "B-rural": 0.04}
    au = A.audit(A.sample(skew, 60, 7), A.pop_shares())
    assert au["B-rural"]["ratio"] < 0.8
    lo, hi = A.wilson(au["B-rural"]["rows"], 60)
    assert lo <= 0.12 <= hi and not au["B-rural"]["flag"]
    # same ratio on a big sample IS flagged: the interval, not the ratio, made the difference
    big = A.sample(skew, 20_000, 7)
    assert A.audit(big, A.pop_shares())["B-rural"]["flag"]


def test_reweighting_an_empty_cell_does_not_crash_or_invent_rows() -> None:
    d = A.sample(A.EMPTY, 5_000, 8)
    w = A.cell_weights(d, A.pop_shares())
    assert len(w) == len(d["y"]) and (w > 0).all() and not (d["cell"] == 3).any()


def test_parse_shares_rejects_bad_input() -> None:
    assert A.parse_shares("") == A.QUOTA
    for bad in ("nope=1", "B-rural=x", "B-rural", "B-rural=0.5"):
        with pytest.raises(ValueError):
            A.parse_shares(bad)
    q = A.parse_shares("B-rural=0.03, A-rural=0.37, B-urban=0.27, A-urban=0.33")
    assert q["B-rural"] == 0.03


R = json.load(open("results.json")) if os.path.exists("results.json") else None
needs = pytest.mark.skipif(R is None, reason="evidence.py has not been run")


@needs
def test_headline_cost_lands_on_rural_b_only() -> None:
    q, p = R["quota"], R["proportional"]
    assert q["B-rural"]["good_rejected"]["lo"] > 2 * p["B-rural"]["good_rejected"]["hi"]
    for c in ("A-urban", "A-rural", "B-urban"):
        assert abs(q[c]["good_rejected"]["mean"] - p[c]["good_rejected"]["mean"]) < 0.02


@needs
def test_overall_accuracy_hides_it() -> None:
    own, dep = R["quota_on_own_test"]["overall"]["acc"]["mean"], R["quota"]["overall"]["acc"]["mean"]
    assert 0 < own - dep < 0.03


@needs
def test_empty_cell_cannot_be_reweighted_and_collecting_converges() -> None:
    e, ew = (
        R["empty"]["B-rural"]["good_rejected"]["mean"],
        R["empty_reweighted"]["B-rural"]["good_rejected"]["mean"],
    )
    assert abs(e - ew) < 0.03 and e > 0.55
    assert (
        R["reweighted"]["B-rural"]["good_rejected"]["mean"] < R["quota"]["B-rural"]["good_rejected"]["mean"]
    )
    curve = [v["mean"] for v in R["curve"]]
    assert curve[0] > curve[-1] and curve[-2] - R["proportional"]["B-rural"]["good_rejected"]["mean"] < 0.03
