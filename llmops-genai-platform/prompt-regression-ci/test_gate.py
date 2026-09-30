"""Engine tests. Each checker is also shown able to FAIL (a verifier must be tested against passing)."""

from __future__ import annotations

import gate as G
import numpy as np
import pytest
from scipy import stats


def test_calibration_is_clean() -> None:
    cal = G.calibrate(8000)
    assert cal["poisson_binomial_vs_binom_gap"] < 1e-12
    assert sum(not r["inside_99"] for r in cal["red_rates"]) <= 1


def test_simulation_check_can_fail() -> None:
    """A simulation of the WRONG scenario must land outside the exact interval."""
    pi_a = G.held_out_set()
    pi_b, _ = G.scenario("hard break", pi_a)
    t = G.threshold(G.flag_prob(pi_a, pi_a, "majority", 1, False))
    hits = int((G.simulate(pi_a, pi_b, "majority", 1, False, 4000) >= t).sum())
    lo, hi = G.wilson(hits, 4000)
    assert not lo <= G.red_prob(G.flag_prob(pi_a, pi_a, "majority", 1, False), t) <= hi


def test_flag_prob_closed_form_for_naive() -> None:
    pi = np.array([0.3, 0.9])
    assert G.flag_prob(pi, pi, "majority", 1, False) == pytest.approx(pi * (1 - pi))


def test_fisher_table_matches_scipy_and_is_one_sided() -> None:
    t = G.reject_table("fisher", 5)
    assert t[5, 0] and not t[0, 5] and not t[3, 3]
    assert t[5, 1] == (stats.fisher_exact([[5, 0], [1, 4]], alternative="greater")[1] < 0.05)


def test_fisher_cannot_fire_at_k3() -> None:
    """3/3 vs 0/3 has one-sided p = 0.05 exactly, not below: the app must not use fisher at k = 3."""
    assert not G.reject_table("fisher", 3).any()


def test_threshold_is_the_smallest_valid_one() -> None:
    p0 = G.flag_prob(G.held_out_set(), G.held_out_set(), "majority", 1, False)
    t = G.threshold(p0)
    assert G.red_prob(p0, t) <= G.ALPHA_GATE < G.red_prob(p0, t - 1)


def test_noop_zero_tolerance_is_always_red_and_quarantine_blinds_flaky() -> None:
    ev = {r["rule"]: r for r in G.evaluate()}
    assert ev["naive k=1"]["noop_red_at_T1"] > 0.99
    q, m = ev["quarantine + majority k=3"], ev["majority k=3"]
    assert q["break on flaky cases"]["red_at_T"] < m["break on flaky cases"]["red_at_T"]
    assert q["hard break"]["red_at_T"] > m["hard break"]["red_at_T"]


def test_audit_lists_and_guards() -> None:
    a = [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]]
    b = [[0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]]
    res = G.audit(a, b)
    assert res["majority"] == [0] and res["naive"] == [0, 1] and res["flaky"] == [1]
    for bad in (([[1, 2]], [[1, 1]]), ([[1, 1]], [[1]]), ([], [])):
        with pytest.raises(ValueError):
            G.audit(*bad)
