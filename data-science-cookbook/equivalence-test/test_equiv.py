"""Engine tests. Each checker is also shown to FAIL on a wrong input (a verifier must be tested
against passing and failing)."""

from __future__ import annotations

import equiv as E
import numpy as np
import pytest


def test_calibration_is_clean() -> None:
    cal = E.calibrate()
    assert cal["scipy_decision_mismatches"] == 0
    assert cal["tost_vs_ci90_mismatches"] == 0
    assert cal["max_gap_vs_noncentral_t"] < 1e-9
    assert cal["max_sum_gap"] < 1e-9


def test_tost_size_never_exceeds_alpha_and_hits_it_at_large_n() -> None:
    for n in (10, 30, 100, 500):
        assert E.p_equivalent(n, E.MARGIN) <= E.ALPHA + 1e-9
    assert abs(E.p_equivalent(500, E.MARGIN) - E.ALPHA) < 1e-4


def test_power_rises_in_n_so_the_bisection_is_valid() -> None:
    p = [E.p_equivalent(n, 0.0) for n in range(5, 120, 5)]
    assert all(b >= a for a, b in zip(p, p[1:]))


def test_n_for_power_is_the_smallest_such_n() -> None:
    n = E.n_for_power(0.8, 0.0)
    assert E.p_equivalent(n, 0.0) >= 0.8 > E.p_equivalent(n - 1, 0.0)


def test_wilson_contains_exact_zero_at_zero_hits() -> None:
    """The first run flagged an exact-0 outcome as OUTSIDE: the bound was 2.7e-20, not 0."""
    assert E.wilson(0, 20_000)[0] == 0.0
    assert E.wilson(20_000, 20_000)[1] == 1.0


def test_monte_carlo_check_can_fail() -> None:
    """Compare raw replicates drawn at the WRONG delta against the exact numbers: must say outside."""
    rng = np.random.default_rng(1)
    x, y = rng.normal(0, 1, (20_000, 50)), rng.normal(0.25, 1, (20_000, 50))
    dec = E.tost_batch(x, y, E.MARGIN)
    hits = int(np.sum(dec["equivalent"]))
    lo, hi = E.wilson(hits, 20_000)
    assert not (lo <= E.p_equivalent(50, 0.0) <= hi)


def test_analyse_outcomes_and_guards() -> None:
    same = E.analyse(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400) + 0.01, 0.5)
    assert same["outcome"] == "equiv_only"
    with pytest.raises(ValueError):
        E.analyse([1.0], [1.0, 2.0], 0.5)
    with pytest.raises(ValueError):
        E.analyse([1.0, 2.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        E.analyse([3.0, 3.0], [3.0, 3.0], 1.0)


def test_exemplar_is_the_misleading_report() -> None:
    ex = E.exemplar()
    assert ex["p_diff"] > 0.25 and ex["outcome"] == "inconclusive"
    assert ex["true_delta"] == ex["margin"]
