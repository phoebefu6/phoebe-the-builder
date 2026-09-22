"""Tests for the engine.

The rule this build follows, inherited from the four siblings: a claim that appears in the README
or the notebook has a test that would fail if the claim stopped being true. Several of these
exist because the corresponding defect actually shipped in an earlier build.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import nonparam as N
import numpy as np
import pytest
from scipy import stats

# ---------------------------------------------------------------------------
# Calibration - the closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", N.SHAPES)
def test_density_integrates_to_one(shape: str) -> None:
    assert abs(N.density_mass(shape) - 1.0) < 1e-6


@pytest.mark.parametrize("shape,exact", sorted(N.ARE_CLOSED_FORM.items()))
def test_are_matches_closed_form(shape: str, exact: float) -> None:
    """normal -> 3/pi, uniform -> 1, centred exponential -> 3. If these drift, the efficiency
    column in section 2 is not measuring efficiency."""
    assert abs(N.asymptotic_are(shape) - exact) < 1e-6


def test_are_ordering_is_the_story() -> None:
    """Mann-Whitney is less efficient than the t-test on NORMAL data and more efficient on every
    other shape here. That ordering is the reason the folk advice survives as power advice."""
    assert N.asymptotic_are("normal") < 1.0
    for shape in N.SHAPES:
        if shape != "normal":
            assert N.asymptotic_are(shape) >= 1.0


@pytest.mark.parametrize("shape", N.SHAPES)
def test_populations_are_standardised(shape: str) -> None:
    """Analytic standardisation, checked empirically. If a population is not mean 0 SD 1, the
    location-shift rows compare effect sizes instead of tests."""
    rng = np.random.default_rng(3)
    x = N._standard(shape, (2_000_000,), rng)
    assert abs(float(x.mean())) < 0.01
    assert abs(float(x.std(ddof=1)) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# The disagreement design is arithmetic, not simulation
# ---------------------------------------------------------------------------


def test_disagreement_design_disagrees_in_closed_form() -> None:
    """The mean says treated is higher; the probability of superiority says treated is lower.
    Both from closed forms. This is the build."""
    assert N.mix_mean() > 0.0
    assert N.mix_prob_superiority() < 0.5


def test_closed_form_superiority_matches_simulation() -> None:
    """Calibration of the closed form against a large draw - without this the headline rests on
    an algebra step nobody checked."""
    rng = np.random.default_rng(11)
    draws = 4_000_000
    a = rng.standard_normal(draws) * N.MIX_SD
    which = rng.random(draws) < N.MIX_WEIGHTS[0]
    b = np.where(which, N.MIX_MEANS[0], N.MIX_MEANS[1]) + rng.standard_normal(draws) * N.MIX_SD
    measured = float(np.mean(b > a))
    assert abs(measured - N.mix_prob_superiority()) < 0.002
    assert abs(float(b.mean()) - N.mix_mean()) < 0.01
    assert abs(float(b.std(ddof=1)) - N.mix_sd()) < 0.02


def test_both_tests_fire_in_opposite_directions() -> None:
    """The headline as a test: at a sample size where both are powered, the contradiction is
    not rare."""
    r = N.run_disagreement(100, 4000, 7)
    assert r["opposite"] > 0.3
    assert r["t_up"] > r["t_down"]
    assert r["mw_down"] > r["mw_up"]


def test_contradiction_needs_both_tests_powered() -> None:
    """At n = 10 neither test has the power to contradict anything, so the rate is low. The
    contradiction is a large-sample phenomenon, which is the uncomfortable part - more data
    makes it MORE likely, not less."""
    small = N.run_disagreement(10, 4000, 8)
    large = N.run_disagreement(200, 4000, 9)
    assert small["opposite"] < large["opposite"]


# ---------------------------------------------------------------------------
# The two test wrappers
# ---------------------------------------------------------------------------


def test_uncorrected_matches_scipy_without_ties() -> None:
    """The hand-rolled no-tie-correction version must be SciPy's asymptotic test when there are
    no ties, or section 5 measures an implementation difference instead of a tie effect."""
    rng = np.random.default_rng(12)
    a = rng.standard_normal((200, 20))
    b = rng.standard_normal((200, 20)) + 0.4
    pc, _ = N.mannwhitney(a, b)
    pu = N.mannwhitney_uncorrected(a, b)
    assert float(np.max(np.abs(pc - pu))) < 1e-12


def test_tie_correction_only_shrinks_the_variance() -> None:
    """Ties can never make Var(U) larger, so the ratio is bounded by 1 and falls as the scale
    gets coarser. If this inverted, the conservative direction in section 5 would be backwards."""
    rng = np.random.default_rng(13)
    prev = 1.01
    for levels in (7, 5, 3, 2):
        rows = N.to_levels(rng.standard_normal((200, 40)), levels)
        ratio = float(N.tie_variance_ratio(rows, levels).mean())
        assert ratio <= 1.0
        assert ratio < prev
        prev = ratio


def test_direction_is_the_sign_of_phat_minus_half() -> None:
    """Mann-Whitney's reported direction must be the sample probability of superiority against
    a half - the same quantity Day 178 measured as an effect size."""
    rng = np.random.default_rng(14)
    a = rng.standard_normal((50, 30))
    b = rng.standard_normal((50, 30)) + 1.0
    _p, direction = N.mannwhitney(a, b)
    assert np.all(direction > 0)
    _p, direction = N.mannwhitney(b, a)
    assert np.all(direction < 0)


def test_welch_is_welch() -> None:
    rng = np.random.default_rng(15)
    a = rng.standard_normal((20, 25))
    b = rng.standard_normal((20, 25)) + 0.5
    p, _ = N.welch(a, b)
    ref = stats.ttest_ind(b[0], a[0], equal_var=False)
    assert abs(p[0] - float(ref.pvalue)) < 1e-12


# ---------------------------------------------------------------------------
# Verdicts, gates and searches
# ---------------------------------------------------------------------------


def test_verdict_uses_the_whole_interval() -> None:
    """Day 175: a verdict read off a point estimate flagged the harness's own exactly-calibrated
    cell. At a small rep count nothing is called broken; at a large one a real gap is."""
    assert N.verdict(0.0700, 200) == "ok"
    assert N.verdict(0.0700, 100_000) == "INFLATED"
    assert N.verdict(0.0300, 100_000) == "CONSERVATIVE"
    assert N.verdict(0.0500, 100_000) == "ok"


def test_comparable_requires_both_tests_to_behave() -> None:
    assert N.comparable(0.050, 0.050, 100_000)
    assert not N.comparable(0.050, 0.080, 100_000)
    assert not N.comparable(0.020, 0.050, 100_000)


def test_monotone_search_refuses_a_bracket_it_cannot_resolve() -> None:
    """Day 178: a predicate that is False everywhere (SciPy's nct returning nan) made a search
    return `hi` as if it were an answer."""
    assert N._monotone_search(lambda n: n >= 37, 1, 100) == 37
    assert N._monotone_search(lambda n: False, 1, 100) is None
    assert N._monotone_search(lambda n: True, 1, 100) == 1


def test_efficiency_refuses_to_extrapolate() -> None:
    """A target power outside the measured curve gets None, never a guess off the end."""
    ns = [20, 40, 80]
    tp = [0.20, 0.45, 0.80]
    assert N.efficiency_from_curve(ns, tp, 40, 0.99) is None
    assert N.efficiency_from_curve(ns, tp, 40, 0.05) is None
    got = N.efficiency_from_curve(ns, tp, 40, 0.45)
    assert got is not None and abs(got - 1.0) < 1e-9


def test_efficiency_direction() -> None:
    """Above 1 must mean Mann-Whitney was more efficient, i.e. the t-test needed MORE data."""
    ns = [20, 40, 80]
    tp = [0.20, 0.45, 0.80]
    got = N.efficiency_from_curve(ns, tp, 40, 0.80)
    assert got is not None and got > 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_design_lists_keep_their_order() -> None:
    """Seeds are INDICES into these lists. Reordering one silently re-labels every stored cell,
    which is exactly the failure Day 178 shipped with hash-derived seeds."""
    assert N.LOCATION_DESIGNS[0] == ("normal", 20)
    assert N.LOCATION_DESIGNS[-1] == ("exponential", 300)
    assert len(N.LOCATION_DESIGNS) == len(N.SHAPES) * len(N.LOCATION_NS)
    assert N.UNEQUAL_DESIGNS[0] == ("normal", 1.0, 30, 30)
    assert len(N.UNEQUAL_DESIGNS) == len(N.SYMMETRIC) * len(N.UNEQUAL_RATIOS) * len(N.UNEQUAL_NS)
    assert N.LOCATION_REF_N in N.LOCATION_NS


def test_a_cell_is_reproducible() -> None:
    assert N.run_disagreement(30, 1000, 4) == N.run_disagreement(30, 1000, 4)
    assert N.run_location("t5", 0.3, 30, 1000, 4) == N.run_location("t5", 0.3, 30, 1000, 4)


@pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
def test_results_json_is_process_stable() -> None:
    """Day 178: `hash()` on strings is salted per process, so the stored study silently differed
    from the notebook's. Re-running in a FRESH interpreter has to give the same bytes."""
    before = open("results.json", "rb").read()
    subprocess.run([sys.executable, "evidence.py"], check=True, capture_output=True,
                   cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    after = open("results.json", "rb").read()
    assert before == after


@pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
def test_stored_results_carry_the_findings() -> None:
    r = json.load(open("results.json"))
    assert r["calibration_ok"] is True
    assert r["config"]["mix_mean"] > 0
    assert r["config"]["mix_prob_superiority"] < 0.5
    assert max(row["opposite"] for row in r["disagreement"]) > 0.3
    # Mann-Whitney must be shown missing its rate somewhere in the unequal-spread grid, or
    # section 3 has no content - and in BOTH directions, since one-directional failure would be
    # a different (and less interesting) finding.
    verdicts = {row["mw_verdict"] for row in r["unequal"]}
    assert "INFLATED" in verdicts and "CONSERVATIVE" in verdicts
    # Welch does NOT hold everywhere - it is conservative on the contaminated cells - so the
    # claim the build actually makes is about the SIZE of the two failures, not their absence.
    # This asserts that comparison rather than the tidier thing I first wrote.
    worst_welch = max(abs(row["welch"] - 0.05) / 0.05 for row in r["unequal"])
    worst_mw = max(abs(row["mw"] - 0.05) / 0.05 for row in r["unequal"])
    assert worst_welch < 0.25
    assert worst_mw > 2.0
    # Both verdict branches reachable in the tie table.
    assert any(row["null_uncorrected_verdict"] == "CONSERVATIVE" for row in r["ties"])
    # The contradiction rate rises with n - the finding is that more data does not help.
    assert r["disagreement_monotone"] is True


@pytest.mark.skipif(not os.path.exists("results.json"), reason="evidence.py has not been run")
def test_measured_efficiency_tracks_predicted_are() -> None:
    """The measured n-equivalence and the asymptotic prediction are different objects - one is a
    finite-n Monte Carlo reading, the other a limit - so they are checked for AGREEMENT IN
    ORDER, not in value. Day 178 established cross-number agreement as a real detector."""
    r = json.load(open("results.json"))
    rows = [e for e in r["efficiency"] if e["measured"] is not None]
    assert len(rows) >= 4
    rho = stats.spearmanr([e["are"] for e in rows], [e["measured"] for e in rows]).statistic
    assert rho > 0.8
    normal = next(e for e in rows if e["shape"] == "normal")
    assert normal["measured"] < 1.05


def test_wilson_is_symmetric_and_bounded() -> None:
    lo, hi = N.wilson(0.5, 1000)
    assert lo < 0.5 < hi
    assert abs((0.5 - lo) - (hi - 0.5)) < 1e-12
    assert N.wilson(0.0, 100)[0] == 0.0
    assert N.wilson(1.0, 100)[1] == 1.0
    assert N.wilson(0.5, 0) == (0.0, 1.0)


def test_mc_interval_narrows_with_n() -> None:
    rng = np.random.default_rng(21)
    small = N.mc_interval(rng.standard_normal(100))
    large = N.mc_interval(rng.standard_normal(10_000))
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_unknown_population_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError):
        N._standard("gaussian", (10,), np.random.default_rng(0))
    with pytest.raises(ValueError):
        N.density("gaussian")


def test_to_levels_produces_the_requested_scale() -> None:
    rng = np.random.default_rng(22)
    x = N.to_levels(rng.standard_normal((100, 200)), 5)
    assert set(np.unique(x).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert len(np.unique(x)) == 5
    # Equal-probability cut points, so a normal population lands roughly evenly across the scale.
    shares = [float((x == lv).mean()) for lv in range(5)]
    assert max(shares) - min(shares) < 0.02


def test_chunking_does_not_change_a_result() -> None:
    """CHUNK exists for memory, not for numbers. A result that depends on the chunk size is a
    result that depends on the machine."""
    saved = N.CHUNK
    try:
        N.CHUNK = 10_000
        whole = N.run_location("normal", 0.3, 20, 3000, 5)
        N.CHUNK = 250
        split = N.run_location("normal", 0.3, 20, 3000, 5)
    finally:
        N.CHUNK = saved
    assert whole == split


def test_math_import_is_used() -> None:
    assert abs(N.ARE_CLOSED_FORM["normal"] - 3.0 / math.pi) < 1e-15
