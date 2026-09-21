"""Assert the findings, not just that the code runs.

Rules carried from the four sibling builds in this domain, each of which was learned by shipping
its violation first:

- Every predicate a finding is stated as must be CAPABLE of returning either answer.
- Every comparison of two measured rates goes through a 99% interval, never point estimates.
- A power comparison is only read where both procedures control Type I.
- A search that assumes monotonicity must check it, because a silently-False predicate returns a
  plausible wrong number rather than an error. That one is new here, and it cost this build two
  wrong tables before it was caught.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
from effectsize import (
    METRICS,
    SHAPES,
    SKEWED,
    _monotone_search,
    _standard,
    analyse,
    analytic_power,
    analytic_prob_superiority,
    contaminate,
    correction_fixes_bias,
    correction_matters,
    dichotomisation_cost,
    dichotomisation_is_costly,
    draw_pair,
    metrics,
    metrics_disagree,
    n_for_significance,
    nct_power,
    power_disagreement,
    smallest_significant_d,
    split_beats_t,
    true_prob_superiority,
    wilson,
)
from scipy import stats

REF = 2_000_000
REPS = 8000


# ---------------------------------------------------------------------------
# The instrument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_every_population_is_standardised_to_mean_0_sd_1(shape):
    """The load-bearing detail: if this drifts, 'the same d' is not the same d and the headline
    finding becomes an artefact of the generator."""
    x = _standard(shape, (4_000_000,), np.random.default_rng(7))
    assert abs(float(x.mean())) < 0.004, f"{shape} mean {x.mean()}"
    assert abs(float(x.std(ddof=1)) - 1.0) < 0.004, f"{shape} sd {x.std(ddof=1)}"


@pytest.mark.parametrize("shape", SHAPES)
def test_the_shapes_are_actually_different_from_each_other(shape):
    """A 'shape' that is secretly normal would make the headline vanish for the wrong reason."""
    x = _standard(shape, (400_000,), np.random.default_rng(8))
    skew, kurt = float(stats.skew(x)), float(stats.kurtosis(x))
    if shape == "normal":
        assert abs(skew) < 0.02 and abs(kurt) < 0.05
    elif shape in SKEWED:
        assert skew > 1.0, f"{shape} is supposed to be skewed, got {skew}"
    else:
        assert abs(skew) < 0.05, f"{shape} is supposed to be symmetric, got {skew}"
        assert abs(kurt) > 0.4, f"{shape} is supposed to differ from normal in the tails, {kurt}"


def test_wilson_contains_the_rate_and_narrows_with_n():
    for rate, n in [(0.05, 1000), (0.0, 500), (1.0, 500), (0.5, 10)]:
        lo, hi = wilson(rate, n)
        assert 0.0 <= lo <= rate <= hi <= 1.0
    wide = wilson(0.5, 1000)
    tight = wilson(0.5, 100000)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


# ---------------------------------------------------------------------------
# Calibration - a known truth, reproduced, before any unknown one is reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.2, 0.5, 0.8])
def test_reference_sampler_reproduces_the_closed_form_on_normal_data(d):
    p, err = true_prob_superiority("normal", d, REF, seed=3)
    exact = analytic_prob_superiority(d)
    assert abs(p - exact) <= max(3 * err, 5e-4), f"measured {p}, closed form {exact}"


def test_cohens_d_estimator_recovers_the_true_d_on_normal_data():
    c = analyse("normal", 0.5, 200, REPS, seed=2, truth=(analytic_prob_superiority(0.5), 0.0))
    lo, hi = c.cis["cohens_d"]
    assert lo <= 0.5 <= hi, f"d-hat 99% interval {c.cis['cohens_d']} missed the truth 0.5"


def test_metrics_agree_with_their_own_definitions():
    """Each metric recomputed the slow, obvious way on one small sample."""
    rng = np.random.default_rng(5)
    a, b = draw_pair("normal", 0.5, 40, 40, 1, rng)
    m = {k: float(v[0]) for k, v in metrics(a, b).items()}
    a0, b0 = a[0], b[0]
    pooled = math.sqrt((a0.var(ddof=1) + b0.var(ddof=1)) / 2)
    assert m["cohens_d"] == pytest.approx((b0.mean() - a0.mean()) / pooled, rel=1e-12)
    assert m["glass_delta"] == pytest.approx((b0.mean() - a0.mean()) / a0.std(ddof=1), rel=1e-12)
    # P(X > Y) by brute force over all pairs, ties counted as half.
    gt = float((b0[:, None] > a0[None, :]).mean())
    eq = float((b0[:, None] == a0[None, :]).mean())
    assert m["prob_superiority"] == pytest.approx(gt + eq / 2, abs=1e-12)
    assert m["cliffs_delta"] == pytest.approx(2 * m["prob_superiority"] - 1, abs=1e-12)
    assert set(m) == set(METRICS)


# ---------------------------------------------------------------------------
# The nan defect, and the guard that now catches it
# ---------------------------------------------------------------------------


def test_scipy_nct_really_does_return_nan_somewhere():
    """Regression note, not a wish: if a future SciPy fixes this, this test fails and the
    fallback's justification should be revisited rather than silently carried forward."""
    ns = list(range(5, 400)) + [500, 1000, 2000, 5000, 10000, 20000, 100000]
    nans = [n for n in ns if not math.isfinite(nct_power(0.5, n))]
    assert nans, "scipy.stats.nct no longer produces nan here - re-examine analytic_power"
    assert min(nans) >= 500, f"nan appeared at small n ({min(nans)}), where the fallback is poor"


def test_the_fallback_is_accurate_where_it_is_actually_used():
    ns = list(range(5, 400)) + [500, 1000, 2000, 5000, 10000, 20000, 100000]
    g = power_disagreement(0.5, ns)
    assert g["n_nan"] > 0
    # The nan band starts far above df=400, so only the large-df agreement matters in practice.
    assert g["worst_gap_large_df"] < 1e-3, g
    assert g["first_nan_n"] > 400


def test_analytic_power_is_finite_and_monotone_everywhere_it_is_used():
    for d in (0.1, 0.5):
        vals = [analytic_power(d, n) for n in
                [5, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000, 200000]]
        assert all(math.isfinite(v) for v in vals)
        assert vals == sorted(vals), f"power is not monotone in n at d={d}: {vals}"


def test_monotone_search_raises_instead_of_returning_a_plausible_wrong_number():
    """The actual failure mode: a predicate that is False everywhere used to return `hi`."""
    with pytest.raises(ValueError):
        _monotone_search(lambda n: False, 4, 1000)
    assert _monotone_search(lambda n: n >= 37, 4, 1000) == 37
    assert _monotone_search(lambda n: True, 4, 1000) == 4


def test_the_sample_size_table_matches_an_independent_build():
    """Day 177 computed these from its own code path. Two builds, one answer, or one is wrong."""
    assert n_for_significance(0.2, 0.80) == 394
    assert n_for_significance(0.5, 0.80) == 64
    assert n_for_significance(0.8, 0.80) == 26


def test_smallest_significant_d_falls_with_n_and_lands_near_a_coin_flip():
    ds = [smallest_significant_d(n) for n in (100, 1000, 10000, 100000, 1000000)]
    assert ds == sorted(ds, reverse=True), ds
    assert analytic_prob_superiority(ds[3]) < 0.51


# ---------------------------------------------------------------------------
# Finding 1 - the same d, different answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.2, 0.5, 0.8])
def test_probability_of_superiority_varies_by_shape_at_a_fixed_d(d):
    cells = [analyse(s, d, 20, 200, seed=1,
                     truth=true_prob_superiority(s, d, REF, seed=11)) for s in SHAPES]
    assert metrics_disagree(cells, tol=0.01) is True
    vals = [c.truth_ps for c in cells]
    assert max(vals) - min(vals) > 0.03, f"spread at d={d} was only {max(vals) - min(vals)}"


def test_the_disagreement_predicate_can_return_false():
    """Falsifiability: fed shapes that genuinely agree, it must say so."""
    cells = [analyse(s, 0.5, 20, 200, seed=1,
                     truth=true_prob_superiority(s, 0.5, REF, seed=12))
             for s in ("normal", "uniform")]
    assert metrics_disagree(cells, tol=0.02) is False


def test_disagreement_predicate_refuses_to_compare_across_different_d():
    with pytest.raises(ValueError):
        metrics_disagree([analyse("normal", 0.2, 20, 200, seed=1, truth=(0.55, 0.0)),
                          analyse("normal", 0.5, 20, 200, seed=2, truth=(0.63, 0.0))])


# ---------------------------------------------------------------------------
# Finding 2 - the estimators
# ---------------------------------------------------------------------------


def test_cohens_d_is_biased_upward_at_small_n_and_the_bias_shrinks():
    biases = [analyse("normal", 0.5, n, REPS, seed=4,
                      truth=(analytic_prob_superiority(0.5), 0.0)).d_bias
              for n in (10, 20, 50, 200)]
    assert biases[0] > 0.005, biases
    assert abs(biases[-1]) < abs(biases[0])


def test_hedges_correction_works_on_normal_data_and_fails_on_skewed_data():
    """Both halves asserted - 'it changes the number' and 'it removes the bias' are different
    claims, and only the second is what the correction is for."""
    norm = analyse("normal", 0.5, 10, REPS, seed=6,
                   truth=true_prob_superiority("normal", 0.5, REF, seed=13))
    skew = analyse("lognormal", 0.5, 10, REPS, seed=7,
                   truth=true_prob_superiority("lognormal", 0.5, REF, seed=14))
    assert correction_matters(norm) and correction_matters(skew)
    assert correction_fixes_bias(norm) is True
    assert correction_fixes_bias(skew) is False
    assert abs(skew.g_bias) > 0.05, skew.g_bias


def test_the_rank_summary_needs_no_small_sample_correction():
    for shape in SHAPES:
        c = analyse(shape, 0.5, 10, REPS, seed=9,
                    truth=true_prob_superiority(shape, 0.5, REF, seed=15))
        assert abs(c.ps_bias) < 0.01, f"{shape}: P(X>Y) bias {c.ps_bias} at n=10"


# ---------------------------------------------------------------------------
# Finding 3 - robustness
# ---------------------------------------------------------------------------


def test_glass_delta_is_the_metric_that_breaks_under_contamination():
    rows = {r.metric: r for r in contaminate("normal", 0.5, 100, REPS, seed=10)}
    worst = max(rows.values(), key=lambda r: abs(r.pct_change))
    assert worst.metric == "glass_delta", f"expected glass_delta, got {worst.metric}"
    assert worst.pct_change > 10.0, worst.pct_change
    assert abs(rows["cohens_d"].pct_change) < 3.0, rows["cohens_d"].pct_change
    # And the direction matters: it reports a BIGGER effect, which is the dangerous direction.
    assert rows["glass_delta"].direction == "up"


# ---------------------------------------------------------------------------
# Finding 4 - the median split
# ---------------------------------------------------------------------------


def test_both_dichotomisation_verdicts_are_reachable():
    """If only one verdict ever fires, the study has an opinion rather than a measurement."""
    norm = dichotomisation_cost("normal", 0.4, 200, REPS, seed=20)
    skew = dichotomisation_cost("lognormal", 0.4, 200, REPS, seed=21)
    assert norm.comparable and skew.comparable
    assert dichotomisation_is_costly(norm) is True
    assert split_beats_t(norm) is False
    assert split_beats_t(skew) is True
    assert dichotomisation_is_costly(skew) is False


def test_an_incomparable_row_yields_no_verdict_in_either_direction():
    r = dichotomisation_cost("uniform", 0.4, 50, REPS, seed=22)
    assert r.comparable is False, "expected the split test to be out of band at n=50"
    assert dichotomisation_is_costly(r) is False and split_beats_t(r) is False


# ---------------------------------------------------------------------------
# The published artifacts must agree with the library
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.exists("results.json"), reason="run evidence.py first")
def test_results_json_is_internally_consistent():
    r = json.load(open("results.json"))
    s = r["summary"]
    assert s["n_calibrated"] == s["n_calibration_cells"]
    assert s["ps_spread_at_half"] > 0.05
    assert s["worst_robustness"]["metric"] == "glass_delta"
    assert s["correction_fixes_shapes"] == ["normal"]
    assert s["smallest_at_100k"]["ps"] < 0.51
    assert s["n_t_wins"] >= 1 and s["n_split_wins"] >= 1, "one-sided study"
    assert s["n_dich_comparable"] < s["n_dich_rows"], "the Type I gate disqualified nothing"
    assert r["power_disagreement"]["n_nan"] > 0
    # Seeds are indices into these lists, so their order is part of the result.
    assert [tuple(x) for x in r["cell_designs"]] == [
        (shape, n) for shape in r["shapes"] for n in r["ns"]
    ]
    for row in r["dichotomisation"]:
        if not row["comparable"]:
            assert not row["t_wins"] and not row["split_wins"]
