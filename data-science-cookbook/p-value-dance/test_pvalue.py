"""Assert the findings, not just that the code runs.

Two rules carried from the sibling builds and worth keeping standard:

- Every predicate a finding is stated as must be CAPABLE of returning either answer. A test that
  only ever sees True is testing a constant. `assumption-pretest-cost` shipped a "win" that was
  measurement error because nothing forced its predicate to be falsifiable.
- Every comparison of two measured rates goes through a 99% Wilson interval, never the point
  estimates. `normality-test-trap` flagged its own known-good calibration cell as broken by
  comparing point estimates at too few replicates.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from pvalue import (
    ALPHA,
    DESIGNS,
    NS,
    analyse,
    analytic_power,
    dance_crosses_threshold,
    dance_has_stopped,
    dance_is_wide,
    ppv_study,
    rate_differs,
    replication_study,
    required_n,
    simulate,
    wilson,
)

TEST_REPS = 12000


@pytest.fixture(scope="module")
def grid():
    return [analyse(d, n, TEST_REPS, seed=i) for i, (d, n) in enumerate(DESIGNS)]


# ---------------------------------------------------------------------------
# The instrument, before any reading taken with it
# ---------------------------------------------------------------------------


def test_wilson_contains_the_rate_and_stays_in_bounds():
    for rate, n in [(0.05, 1000), (0.0, 500), (1.0, 500), (0.5, 10)]:
        lo, hi = wilson(rate, n)
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= rate <= hi
    # More replicates must narrow it, or it is not measuring anything.
    assert (wilson(0.05, 100000)[1] - wilson(0.05, 100000)[0]) < (
        wilson(0.05, 1000)[1] - wilson(0.05, 1000)[0]
    )


def test_rate_differs_is_capable_of_both_answers():
    assert rate_differs(0.05, 40000, 0.10, 40000) is True
    assert rate_differs(0.050, 40000, 0.051, 40000) is False


def test_analytic_power_is_alpha_under_the_null_and_monotone_elsewhere():
    for n in NS:
        assert analytic_power(0.0, n) == pytest.approx(ALPHA, abs=1e-12)
    for n in NS:
        assert analytic_power(0.2, n) < analytic_power(0.5, n) < analytic_power(0.8, n)
    for d in (0.2, 0.5):
        powers = [analytic_power(d, n) for n in NS]
        assert powers == sorted(powers)


def test_required_n_brackets_the_target():
    for d in (0.2, 0.5, 0.8):
        n = required_n(d, 0.80)
        assert analytic_power(d, n) >= 0.80
        assert analytic_power(d, n - 1) < 0.80
    # The headline context number in the README.
    assert required_n(0.2, 0.80) == 394


# ---------------------------------------------------------------------------
# Calibration - the harness must reproduce a known truth first
# ---------------------------------------------------------------------------


def test_every_design_matches_the_noncentral_t_formula(grid):
    bad = [c.label for c in grid if not c.calibrated]
    assert not bad, f"measured power missed the analytic value on: {bad}"


def test_p_is_uniform_under_the_null_and_not_under_an_alternative(grid):
    for c in grid:
        if c.d_true == 0:
            assert c.uniform_ok is True, f"{c.label} failed the uniformity check (KS p={c.ks_p})"
            # Uniform(0,1) has its 5th percentile at 0.05 and its median at 0.5.
            assert c.p_pcts["p5"] == pytest.approx(0.05, abs=0.006)
            assert c.p_pcts["p50"] == pytest.approx(0.50, abs=0.02)
        else:
            assert c.uniform_ok is None
            assert c.ks_p < 0.01, f"{c.label} looks uniform, but it has a real effect"


def test_simulate_is_reproducible_from_its_seed():
    a, da = simulate(0.5, 20, 2000, seed=11)
    b, db = simulate(0.5, 20, 2000, seed=11)
    assert np.array_equal(a, b) and np.array_equal(da, db)
    c, _ = simulate(0.5, 20, 2000, seed=12)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Finding 1 - the dance is widest where it matters least
# ---------------------------------------------------------------------------


def test_the_dance_predicates_can_return_either_answer(grid):
    wide = {dance_is_wide(c) for c in grid}
    crossing = {dance_crosses_threshold(c) for c in grid}
    stopped = {dance_has_stopped(c) for c in grid}
    assert wide == {True, False}, "dance_is_wide never returned both - it is testing a constant"
    assert crossing == {True, False}
    assert stopped == {True, False}, (
        "no design had the dance stop; the 'this is a low-power phenomenon' claim is unsupported"
    )


def test_spread_is_widest_at_high_power_not_low(grid):
    """The reversal of the folk story, asserted rather than asserted-in-prose."""
    widest = max(grid, key=lambda c: c.log10_spread)
    narrowest = min(grid, key=lambda c: c.log10_spread)
    assert widest.power > 0.95, f"widest spread was at power {widest.power}, not a high-power cell"
    assert narrowest.d_true == 0, "the narrowest spread should be the null, where p is uniform"
    # And it is a rank claim across the whole grid, not two cherry-picked rows.
    powers = [c.power for c in grid]
    spreads = [c.log10_spread for c in grid]
    rho = np.corrcoef(np.argsort(np.argsort(powers)), np.argsort(np.argsort(spreads)))[0, 1]
    assert rho > 0.9, f"spread should rise with power across the grid, rank corr was {rho:.3f}"


def test_the_straddle_predicate_is_exactly_power_below_95(grid):
    """The sharpest negative result: for d>0 the decision-relevant dance IS the power question.

    p95 < alpha if and only if more than 95% of replicates fell below alpha, i.e. measured
    power > 0.95. This is an identity, so it is tested as one.
    """
    for c in grid:
        if c.d_true == 0:
            assert dance_crosses_threshold(c) is False  # degenerate, guarded
            continue
        assert dance_crosses_threshold(c) == (c.power < 0.95), c.label


# ---------------------------------------------------------------------------
# Finding 2 - the winner's curse, which is NOT power restated
# ---------------------------------------------------------------------------


def test_significant_results_overstate_the_effect_and_worse_at_low_power(grid):
    alt = [c for c in grid if c.d_true > 0]
    assert all(c.type_m >= 1.0 for c in alt), "type M below 1 means published effects understate"
    low = [c for c in alt if c.power < 0.15]
    high = [c for c in alt if c.power > 0.95]
    assert low and high
    assert min(c.type_m for c in low) > max(c.type_m for c in high)
    # At high power the curse effectively vanishes - the other half of the claim. The threshold
    # is read off the grid rather than guessed: at power > 0.95 the worst cell is d=0.8/n=50 at
    # 1.021, and by power > 0.99 every cell is inside 1.005. Both halves are asserted, because
    # "it shrinks" and "it disappears" are different claims and only the second is the finding.
    assert max(c.type_m for c in high) < 1.05
    very_high = [c for c in alt if c.power > 0.99]
    assert very_high
    assert max(c.type_m for c in very_high) < 1.01


def test_wrong_sign_results_are_significant_at_low_power(grid):
    worst = max((c for c in grid if c.d_true > 0), key=lambda c: c.type_s)
    assert worst.power < 0.10, "the sign error should live in the lowest-power corner"
    # Stated with its interval, never the point estimate.
    assert worst.type_s_ci[0] > 0.08, f"type S interval {worst.type_s_ci} does not clear 8%"
    # And it must genuinely go away at high power, or it is not a low-power finding.
    for c in grid:
        if c.d_true > 0 and c.power > 0.95:
            assert c.type_s == 0.0, f"{c.label} still has sign errors at power {c.power}"


# ---------------------------------------------------------------------------
# Finding 3 - conditioning on p = 0.05 buys nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d,n", [(0.2, 50), (0.5, 20), (0.5, 50), (0.8, 20)])
def test_replication_rate_is_just_the_unconditional_power(d, n):
    r = replication_study(d, n, 60000, seed=hash((d, n)) % 10000)
    assert r.hits > 500, f"only {r.hits} studies landed in the window - too few to read"
    assert not r.differs_from_power, (
        f"{r.label}: replication rate {r.rep_sig_rate:.4f} CI {r.rep_sig_ci} "
        f"excluded power {r.power_analytic:.4f}"
    )


def test_the_differs_predicate_can_fire():
    """The falsifiability guard: feed it a rate that genuinely is not the power."""
    r = replication_study(0.5, 50, 60000, seed=3)
    lo, hi = r.rep_sig_ci
    assert not (lo <= 0.05 <= hi), "the interval must be able to exclude a wrong value"


# ---------------------------------------------------------------------------
# Finding 4 - what a significant result is worth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prior,d,n", [(0.5, 0.5, 50), (0.2, 0.5, 50), (0.1, 0.5, 50), (0.1, 0.2, 20)]
)
def test_simulated_false_discovery_share_matches_the_closed_form(prior, d, n):
    v = ppv_study(prior, d, n, 80000, seed=int(prior * 100) + n)
    assert v.n_sig > 1000
    assert v.matches_formula, (
        f"simulated {v.false_share:.4f} CI {v.false_share_ci} vs formula {v.formula:.4f}"
    )


def test_false_discovery_share_rises_as_the_prior_falls():
    shares = [ppv_study(p, 0.5, 50, 60000, seed=40 + i).false_share
              for i, p in enumerate((0.5, 0.2, 0.1))]
    assert shares == sorted(shares)
    assert shares[-1] > 0.30


def test_low_power_plus_low_prior_makes_most_significant_findings_null():
    v = ppv_study(0.1, 0.2, 20, 120000, seed=77)
    assert v.false_share_ci[0] > 0.75, (
        f"the headline 'most significant findings are null' needs the interval to clear 75%, "
        f"got {v.false_share_ci}"
    )


# ---------------------------------------------------------------------------
# The published artifacts must agree with the library
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists("results.json"), reason="run evidence.py first")
def test_results_json_is_internally_consistent():
    r = json.load(open("results.json"))
    s = r["summary"]
    assert s["n_calibrated"] == s["n_cells"], "evidence.txt shipped an uncalibrated cell"
    assert s["n_replication_differs"] == 0
    assert s["widest"]["power"] > 0.95 and s["narrowest"]["power"] < 0.06
    assert s["worst_type_m"]["value"] > 3.0
    assert s["worst_type_s"]["value"] > 0.08
    assert s["worst_ppv"]["false_share"] > 0.75
    for cell in r["cells"]:
        assert cell["calibrated"]
        if cell["d_true"] > 0:
            assert cell["crosses"] == (cell["power"] < 0.95)
    for row in r["ppv"]:
        assert row["matches_formula"]
