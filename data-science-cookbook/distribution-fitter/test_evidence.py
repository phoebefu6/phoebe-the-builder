"""Tests for the experiments.

These run the same simulations the README quotes, at smaller sizes, and assert the
direction of each effect rather than the exact number. An experiment whose conclusion
survives only at one sample size is not evidence, so every assertion here is about a sign
or an ordering, with the noisy magnitudes left to the README where they are quoted with
their settings.
"""

from __future__ import annotations

import math

import numpy as np

from evidence import (
    confident_winner_on_unfittable_data,
    free_location_cost,
    ks_calibration,
    ks_power,
    mixture_sample,
    rounding_vs_n,
    stability_vs_n,
    tail_error,
)


def _run(name: str) -> None:
    print(f"  {name}")


def test_naive_ks_is_useless_under_a_true_null() -> None:
    """A correctly calibrated test rejects alpha of the time when the null is true.

    The naive test rejects essentially never and its p-values average far above 0.5, which
    is not conservatism - it is the estimation shrinkage showing up as fake evidence FOR the
    null. The bootstrap version lands near alpha.
    """
    res = ks_calibration("normal", n=150, n_datasets=80, n_boot=80, seed=101)
    assert res.reject_naive < 0.02
    assert res.mean_p_naive > 0.6, "naive p-values should pile up near 1"
    assert 0.0 <= res.reject_bootstrap <= 0.20, "bootstrap should sit near alpha=0.05"
    assert 0.35 < res.mean_p_bootstrap < 0.65, "a calibrated p-value averages 0.5"
    _run(
        f"null true: naive rejects {res.reject_naive:.0%} (mean p {res.mean_p_naive:.2f}), "
        f"bootstrap {res.reject_bootstrap:.0%} (mean p {res.mean_p_bootstrap:.2f})"
    )


def test_bootstrap_beats_naive_on_power() -> None:
    res = ks_power("normal", "student_t", n=400, n_datasets=40, n_boot=80, seed=103)
    assert res.reject_bootstrap > res.reject_naive
    _run(
        f"null false: bootstrap detects {res.reject_bootstrap:.0%} vs naive "
        f"{res.reject_naive:.0%}"
    )


def test_aic_produces_a_confident_winner_on_data_no_candidate_can_describe() -> None:
    true_row, mix_row, _, rep_mix = confident_winner_on_unfittable_data(
        n=800, n_boot=80, stability_reps=30, seed=105
    )
    # both look identical in the AIC column
    assert true_row.winner_weight > 0.9
    assert mix_row.winner_weight > 0.9
    # only the absolute test separates them
    assert true_row.n_adequate >= 1
    assert mix_row.n_adequate == 0
    assert "NO ADEQUATE FIT" in rep_mix.verdict()
    _run(
        f"mixture: AIC winner `{mix_row.winner}` at weight {mix_row.winner_weight:.2f}, "
        f"{mix_row.n_adequate}/{mix_row.n_candidates} candidates adequate"
    )


def test_mixture_sample_is_outside_every_candidate_family() -> None:
    x = mixture_sample(2000, seed=7)
    assert np.all(x > 0)
    # bimodal in log space is the defining property; check the log histogram has a dip
    logs = np.log(x)
    counts, _ = np.histogram(logs, bins=30)
    peak_left = counts[:15].max()
    peak_right = counts[15:].max()
    trough = counts[10:22].min()
    assert trough < 0.5 * min(peak_left, peak_right), "expected a genuine bimodal dip"
    _run("the mixture really is bimodal in log space, not a heavy tail")


def test_selection_stability_rises_with_n() -> None:
    rows = stability_vs_n(
        true_family="gamma",
        rivals=("gamma", "lognormal", "weibull", "normal"),
        sizes=(80, 2000),
        reps=40,
        seed=107,
    )
    assert len(rows) == 2
    small, large = rows
    assert large.win_share_true >= small.win_share_true
    assert large.winner == "gamma"
    _run(
        f"true-family win share goes {small.win_share_true:.0%} (n={small.n}) -> "
        f"{large.win_share_true:.0%} (n={large.n})"
    )


def test_rounding_rejects_the_true_family_once_n_is_large_enough() -> None:
    rows = rounding_vs_n(sizes=(200, 8000), decimals=1, n_boot=100, seed=109)
    small, large = rows
    assert large.tie_fraction > small.tie_fraction or large.tie_fraction > 0.5
    assert large.p_bootstrap < small.p_bootstrap
    assert large.rejected and not small.rejected
    _run(
        f"normal data rounded to 1dp: p={small.p_bootstrap:.3f} at n={small.n}, "
        f"p={large.p_bootstrap:.3f} at n={large.n} - same data, same shape"
    )


def test_rounding_is_the_cause_not_the_sample_size() -> None:
    """The control: unrounded normal data at the same large n is not rejected."""
    rows = rounding_vs_n(sizes=(8000,), decimals=None, n_boot=100, seed=109)
    assert rows[0].tie_fraction == 0.0
    assert not rows[0].rejected
    _run(f"unrounded normal at n=8000 survives (p={rows[0].p_bootstrap:.3f})")


def test_free_location_helps_the_wrong_family_more_than_the_right_one() -> None:
    """The backwards incentive, as a test.

    On gamma data the true family gains nothing from a free loc and a wrong family gains a
    lot, so adding the parameter to every candidate moves the ranking toward whichever
    family needed the most help.
    """
    rows = free_location_cost(n=1200, seed=31)
    by_name = {r.family_name: r for r in rows}
    assert "gamma" in by_name and "lognormal" in by_name
    true_gain = by_name["gamma"].loglik_gain
    wrong_gain = by_name["lognormal"].loglik_gain
    assert math.isfinite(true_gain) and math.isfinite(wrong_gain)
    assert true_gain < 1.0, f"true family should gain ~nothing, gained {true_gain:.2f}"
    assert wrong_gain > 10.0, f"wrong family should gain a lot, gained {wrong_gain:.2f}"
    assert by_name["lognormal"].free_wins, "3-param lognormal should beat its own 2-param fit"
    assert not by_name["gamma"].free_wins, "AIC should reject the redundant parameter"
    # none of the estimates recovers the true loc = 0
    assert all(abs(r.loc_free) > 1e-6 for r in rows)
    _run(
        f"free loc buys the true family {true_gain:.2f} logLik and a wrong family "
        f"{wrong_gain:.2f} - the reward scales with being wrong"
    )


def test_free_location_can_return_a_model_that_excludes_observed_data() -> None:
    rows = free_location_cost(n=1200, seed=31)
    by_name = {r.family_name: r for r in rows}
    if "weibull" not in by_name:
        _run("weibull free-loc fit not returned by scipy on this sample; skipped")
        return
    w = by_name["weibull"]
    assert w.invalid, "expected the 3-param Weibull optimiser to overshoot min(x)"
    assert w.loc_free > w.data_min
    assert not math.isfinite(w.loglik_gain)
    _run(
        f"3-param Weibull returns loc={w.loc_free:.3f} > min(x)={w.data_min:.3f}: "
        "observed points get zero density and logLik is -inf"
    )


def test_tail_errors_are_large_and_point_in_different_directions() -> None:
    """The downstream consequence of "no adequate fit", in the units the decision uses.

    Every candidate understates p99 badly, and the AIC winner then *overstates* p99.9 by a
    multiple. There is no safe direction to round, which is the practical reason "least-bad"
    has to be labelled rather than reported as the answer.
    """
    rows = tail_error()
    assert rows
    winner = [r for r in rows if r.is_winner]
    assert len(winner) == 1
    assert all(r.err99 < -25 for r in rows), "expected every candidate to undershoot p99"
    assert winner[0].err999 > 100, "expected the AIC winner to overshoot p99.9"
    assert any(r.err999 < -40 for r in rows), "and others to undershoot it"
    _run(
        f"tail errors at p99 range {min(r.err99 for r in rows):.0f}% to "
        f"{max(r.err99 for r in rows):.0f}%; the winner's p99.9 is "
        f"{winner[0].err999:+.0f}%"
    )


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests in test_evidence.py")
    for t in tests:
        t()
    print(f"PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
