"""Engine tests. Each checker is also shown able to FAIL (a verifier must be tested against passing)."""

from __future__ import annotations

import numpy as np
import paired as P
import pytest


def test_calibration_is_clean() -> None:
    cal = P.calibrate()
    assert cal["scipy_decision_mismatches"] == 0
    assert cal["max_gap_vs_student_nct_at_rho0"] < 1e-5
    assert cal["gap_160_vs_320_nodes"] < 1e-5


def test_calibration_can_fail_on_a_wrong_rho() -> None:
    """The rho = 0 anchor must NOT match when the quadrature runs at another rho."""
    from scipy import stats

    n, d = 20, 0.5
    c = stats.t.ppf(0.975, 2 * n - 2)
    exact = stats.nct.sf(c, 2 * n - 2, d * np.sqrt(n / 2)) + stats.nct.sf(c, 2 * n - 2, -d * np.sqrt(n / 2))
    assert abs(P.power_independent(n, 0.3, d) - exact) > 1e-3


def test_paired_size_is_alpha_and_independent_size_moves_with_rho() -> None:
    assert abs(P.power_paired(15, 0.6, 0.0) - P.ALPHA) < 1e-12
    assert abs(P.power_independent(15, 0.0, 0.0) - P.ALPHA) < 1e-6
    assert P.power_independent(15, -0.5, 0.0) > 0.09  # anti-conservative
    assert P.power_independent(15, 0.6, 0.0) < 0.01  # conservative


def test_no_nan_in_the_far_tail() -> None:
    """scipy's nct.cdf(-c) returns nan at nc ~ 10; the symmetric form must not."""
    assert P.power_paired(40, 0.95) == pytest.approx(1.0, abs=1e-9)
    assert all(np.isfinite([r["power_paired"] for r in P.grid()]))


def test_n_for_power_is_the_smallest_such_n() -> None:
    for rho, kind in ((0.5, "paired"), (0.8, "independent")):
        n = P.n_for_power(0.8, rho, kind)
        f = P.power_paired if kind == "paired" else P.power_independent
        assert f(n, rho) >= 0.8 > f(n - 1, rho)


def test_break_even_rho_is_small_positive_and_shrinks_with_n() -> None:
    be = [P.break_even_rho(n) for n in (3, 10, 80)]
    assert all(0 < b < 0.15 for b in be)
    assert be[0] > be[1] > be[2]


def test_monte_carlo_agrees_or_is_rechecked() -> None:
    for r in P.monte_carlo(reps=4000):
        for k in ("paired", "independent"):
            assert r[k]["inside_99"] or r[k]["recheck"]["inside_99"]


def test_monte_carlo_check_can_fail() -> None:
    """Replicates drawn at the wrong rho must land outside the exact value's interval."""
    rng = np.random.default_rng(3)
    x = P._pairs(rng, 20, 0.2, 0.5, 20000)
    k = int((P._t_paired(x[..., 0], x[..., 1]) > 0).sum())
    lo, hi = P.wilson(k, 20000)
    assert not lo <= P.power_paired(20, 0.8) <= hi


def test_analyse_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="pairs must match"):
        P.analyse([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="at least 3"):
        P.analyse([1, 2], [2, 3])
    with pytest.raises(ValueError, match="constant"):
        P.analyse([1, 2, 3], [2, 3, 4])


def test_exemplar_is_a_typical_disagreement() -> None:
    ex = P.exemplar()
    assert ex["p_paired"] < P.ALPHA <= ex["p_independent"]
    assert ex["p_only_paired"] > 0.5  # the disagreement shown is the common case at this design
