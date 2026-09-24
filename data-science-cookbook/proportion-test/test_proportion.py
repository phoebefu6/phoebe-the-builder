"""Tests for the engine. Every README claim has a test that fails if the claim stops being true.
All numbers are exact, so the assertions are equalities or strict inequalities, not bands."""

from __future__ import annotations

import json
import os

import numpy as np
import proportion as P
import pytest
from scipy import stats

# ---------------------------------------------------------------- calibration against scipy


def test_z_squared_is_chi_square() -> None:
    """The identity the whole 'z or chi-square?' question dissolves into."""
    for n1, n2 in [(7, 9), (20, 20), (15, 60)]:
        assert np.abs(P.z_stat(n1, n2) ** 2 - P.chi2_uncorrected(n1, n2)).max() < 1e-10
        assert np.allclose(P.p_z(n1, n2), stats.chi2.sf(P.chi2_uncorrected(n1, n2), 1))


def test_fisher_matches_scipy_on_every_table() -> None:
    n1, n2 = 9, 13
    pv = P.p_fisher(n1, n2)
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            assert pv[i, j] == pytest.approx(stats.fisher_exact([[i, n1 - i], [j, n2 - j]])[1],
                                             abs=1e-12)


def test_yates_matches_scipy() -> None:
    n1, n2 = 11, 8
    pv = P.p_yates(n1, n2)
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            if 0 < i + j < n1 + n2:
                ref = stats.chi2_contingency([[i, n1 - i], [j, n2 - j]], correction=True)[1]
                assert pv[i, j] == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("i,j", [(0, 4), (2, 8), (5, 6), (10, 3)])
def test_barnard_matches_scipy_within_grid_error(i: int, j: int) -> None:
    n1, n2 = 15, 12
    ref = stats.barnard_exact(np.array([[i, j], [n1 - i, n2 - j]])).pvalue
    got = P.p_barnard(n1, n2)[i, j]
    assert abs(got - ref) < 5e-4
    assert got <= ref + 1e-9, "a nuisance grid can only UNDER-estimate the supremum"


def test_degenerate_tables_give_p_one() -> None:
    """Edge case: nobody converted anywhere - no test may claim a difference."""
    pv = P.all_pvalues(10, 10)
    for t in P.TESTS:
        assert pv[t][0, 0] == pytest.approx(1.0)
        assert pv[t][10, 10] == pytest.approx(1.0)


# ------------------------------------------------------------------------ exact size/power


def test_weights_sum_to_one() -> None:
    assert P.weights(30, 40, 0.2, 0.3).sum() == pytest.approx(1.0)


def test_barnard_never_exceeds_alpha() -> None:
    for n1, n2 in [(10, 10), (20, 80)]:
        curve = P.size_curve(P.p_barnard(n1, n2), n1, n2, P.SIZE_PS)
        assert curve.max() <= P.ALPHA + 1e-9


def test_fisher_is_conservative_and_z_can_run_hot() -> None:
    """Both verdicts must be reachable on real designs, not contrived ones."""
    pv = P.all_pvalues(20, 80)
    assert P.size_curve(pv["fisher"], 20, 80, P.SIZE_PS).max() < P.BAND_LO
    assert P.size_curve(pv["z"], 20, 80, P.SIZE_PS).max() > P.BAND_HI


def test_exact_rate_matches_simulation_once() -> None:
    """'Exact' is checked against brute force once, so it is not just a claim."""
    rng = np.random.default_rng(1)
    n1, n2 = 30, 30
    rej = P.p_z(n1, n2) < P.ALPHA
    a, b = rng.binomial(n1, 0.1, 200_000), rng.binomial(n2, 0.3, 200_000)
    assert abs(rej[a, b].mean() - P.rate(rej, n1, n2, 0.1, 0.3)) < 0.004


def test_verdict_branches() -> None:
    assert P.verdict(0.06) == "INFLATED"
    assert P.verdict(0.03) == "CONSERVATIVE"
    assert P.verdict(0.05) == "ok"


def test_exemplar_is_the_rule_chosen_table() -> None:
    ex = P.exemplar()
    assert (ex["x1"], ex["x2"]) == (0, 4)
    assert ex["z"] < 0.05 and ex["barnard"] < 0.05
    assert ex["fisher"] > 0.05 and ex["yates"] > 0.05


def test_lattice_regions_nest_on_20_20() -> None:
    c = P.lattice_regions()["counts"]
    assert c["other"] == 0
    assert sum(c.values()) == 21 * 21


# ------------------------------------------------------------------ stored study (README)

needs = pytest.mark.skipif(not os.path.exists("results.json"), reason="run evidence.py first")


@needs
def test_headline_numbers() -> None:
    R = json.load(open("results.json"))
    worst = max(R["size"], key=lambda r: r["z"])
    assert (worst["n1"], worst["n2"]) == (20, 80)
    assert worst["z"] == pytest.approx(0.0877, abs=1e-4)
    assert all(r["fisher_verdict"] == "CONSERVATIVE" for r in R["size"])
    assert all(r["barnard"] <= 0.05 + 1e-9 for r in R["size"])


@needs
def test_rule_of_five_misses_half_the_broken_cells() -> None:
    ec = json.load(open("results.json"))["expected_count_rule"]
    assert ec["safe_but_broken"] == ec["unsafe_broken"] == 12
    assert ec["unsafe_but_fine"] / (ec["unsafe_but_fine"] + ec["unsafe_broken"]) > 0.9


@needs
def test_barnard_beats_fisher_everywhere_measured() -> None:
    R = json.load(open("results.json"))
    gains = [r["barnard"] - r["fisher"] for r in R["power"]]
    assert min(gains) > 0
    assert max(gains) == pytest.approx(0.1085, abs=1e-4)


@needs
def test_results_are_reproducible() -> None:
    """No seeds anywhere, so a rerun must be byte-identical."""
    import subprocess
    import sys

    before = open("results.json").read()
    subprocess.run([sys.executable, "evidence.py"], check=True, capture_output=True)
    assert open("results.json").read() == before
