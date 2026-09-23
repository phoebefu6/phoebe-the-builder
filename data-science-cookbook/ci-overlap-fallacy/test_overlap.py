"""Tests for the engine.

A claim that appears in the README or the notebook has a test that would fail if the claim
stopped being true. The geometry is closed form, so those tests assert equalities; the Monte
Carlo claims assert against intervals, never a bare point estimate.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import overlap as o
import pytest
from scipy import stats

# ---------------------------------------------------------------------------
# 1. geometry - exact
# ---------------------------------------------------------------------------


def test_se_ratio_bounds() -> None:
    assert o.se_ratio(1.0, 1.0) == pytest.approx(math.sqrt(2), abs=1e-12)
    assert o.se_ratio(1.0, 1e6) == pytest.approx(1.0, abs=1e-5)
    for s2 in np.geomspace(0.01, 100, 200):
        r = o.se_ratio(1.0, float(s2))
        assert 1.0 <= r <= math.sqrt(2) + 1e-12


def test_se_ratio_rejects_degenerate_input() -> None:
    """The one edge case: two zero standard errors have no ratio, and saying 1.0 would be a lie."""
    with pytest.raises(ValueError):
        o.se_ratio(0.0, 0.0)
    with pytest.raises(ValueError):
        o.se_ratio(-1.0, 1.0)


def test_matching_level_is_83_4_percent_at_equal_se() -> None:
    assert o.MATCHING_LEVEL_EQUAL_SE == pytest.approx(0.8342, abs=5e-4)
    assert o.z_for_level(o.MATCHING_LEVEL_EQUAL_SE) == pytest.approx(o.Z_ALPHA / math.sqrt(2))


def test_matching_level_rises_toward_95_as_se_diverge() -> None:
    levels = [o.matching_level(1.0, r) for r in o.RATIO_GRID]
    assert levels == sorted(levels)
    assert levels[-1] < 0.95


def test_overlap_at_significance_is_two_minus_root_two() -> None:
    assert o.overlap_fraction_at_significance(1.0, 1.0) == pytest.approx(2 - math.sqrt(2))


def test_permitted_overlap_has_no_single_number() -> None:
    """'A little overlap is fine' - the permitted share falls from 58.6% toward 0."""
    f = [o.overlap_fraction_at_significance(1.0, r) for r in o.RATIO_GRID]
    assert f == sorted(f, reverse=True)
    assert f[0] > 0.58 and f[-1] < 0.07


def test_rule_size_is_under_one_percent() -> None:
    assert o.RULE_ALPHA_EQUAL_SE == pytest.approx(0.0056, abs=1e-4)


def test_boundary_geometry_is_consistent() -> None:
    """At d exactly on the test's boundary, the painted overlap equals the closed form."""
    z = o.Z_ALPHA
    d = z * math.sqrt(2)
    share = o.overlap_share(np.array([-z]), np.array([z]), np.array([d - z]), np.array([d + z]))
    assert float(share[0]) == pytest.approx(o.overlap_fraction_at_significance(1, 1))


def test_paired_slack_caps_at_root_two_only_when_independent() -> None:
    assert o.paired_slack(1, 1, 0.0) == pytest.approx(math.sqrt(2) * 1.0, rel=1e-9)
    assert o.paired_slack(1, 1, 0.95) > 6.0
    assert o.paired_slack(1, 1, 1.0) == float("inf")


# ---------------------------------------------------------------------------
# 2. containment - the harness calibration
# ---------------------------------------------------------------------------


def test_containment_holds_when_bars_and_test_share_an_se() -> None:
    r = o.run_means(100, 5, 1.0, 1.0, 1.2, reps=20_000, seed=1, known_sigma=True, test="z")
    assert r["gap_and_not_sig"] == 0.0


def test_containment_breaks_when_they_do_not() -> None:
    """Both branches must be reachable, or the calibration row is a constant wearing a test."""
    r = o.run_means(100, 5, 1.0, 1.0, 1.2, reps=20_000, seed=1, known_sigma=True, test="welch")
    assert r["gap_and_not_sig"] > 0.05


# ---------------------------------------------------------------------------
# 3. proportions - exact
# ---------------------------------------------------------------------------


def test_wald_coverage_worst_cell() -> None:
    assert o.exact_coverage("wald", 0.02, 20) == pytest.approx(0.3318, abs=1e-4)


def test_exact_coverage_matches_simulation() -> None:
    """The enumeration is checked against brute force once, so 'exact' is not a claim."""
    rng = np.random.default_rng(3)
    x = rng.binomial(40, 0.1, size=200_000)
    lo, hi = o.prop_ci("wilson", x, 40)
    sim = float(((lo <= 0.1) & (0.1 <= hi)).mean())
    assert abs(sim - o.exact_coverage("wilson", 0.1, 40)) < 0.003


def test_clopper_pearson_never_undercovers() -> None:
    for n in o.COVERAGE_NS:
        for p in o.COVERAGE_PS:
            assert o.exact_coverage("clopper_pearson", p, n) >= 0.95 - 1e-9


def test_wald_is_zero_width_on_zero_events() -> None:
    lo, hi = o.prop_ci("wald", np.array([0]), 40)
    assert float(lo[0]) == float(hi[0]) == 0.0
    lo, hi = o.prop_ci("wilson", np.array([0]), 40)
    assert float(hi[0]) > 0.0


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError):
        o.prop_ci("bayes", np.array([1]), 10)


def test_two_prop_z_matches_scipy_chi2() -> None:
    z = float(o.two_prop_z(np.array(30.0), 100, np.array(45.0), 100))
    chi2 = stats.chi2_contingency([[30, 70], [45, 55]], correction=False)[0]
    assert z * z == pytest.approx(chi2, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. the stored study - README numbers
# ---------------------------------------------------------------------------

needs_results = pytest.mark.skipif(not os.path.exists("results.json"),
                                   reason="evidence.py has not been run")


@needs_results
def test_headline_dead_zone_share() -> None:
    R = json.load(open("results.json"))
    top = max(R["means"], key=lambda r: r["overlap_given_sig"])
    assert top["overlap_given_sig"] == pytest.approx(0.5236, abs=1e-4)
    assert top["ogs_lo"] > 0.5


@needs_results
def test_paired_reversal_is_stored() -> None:
    R = json.load(open("results.json"))
    top = R["paired"][-1]
    assert top["rho"] == 0.95 and top["overlap_given_sig"] > 0.999


@needs_results
def test_calibration_rows_have_zero_violations() -> None:
    R = json.load(open("results.json"))
    assert all(r["gap_and_not_sig"] == 0.0 for r in R["calibration"])


@needs_results
def test_matching_level_cuts_disagreement_tenfold() -> None:
    R = json.load(open("results.json"))
    for r in R["levels"]:
        assert r["disagree_matched"] * 5 < r["disagree_95"]
