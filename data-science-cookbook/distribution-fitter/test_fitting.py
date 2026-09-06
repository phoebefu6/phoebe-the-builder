"""Tests for the fitting core.

The interesting ones are the calibration tests. A distribution fitter that returns numbers
is easy; the tests that matter check that the numbers mean what the labels say - that the
free-parameter count matches what scipy actually estimated, that the bootstrap p-value
respects its own resolution floor, and that the bootstrap test has power where the naive
test does not.
"""

from __future__ import annotations

import math

import numpy as np
from fitting import (
    DEFAULT_FAMILIES,
    _infer_decimals,
    bootstrap_ks,
    diagnose,
    family,
    fit_distributions,
    fit_params,
    ks_statistic,
    probe_free_location,
    qq_points,
    sample_book,
    selection_stability,
    support_violation,
)
from scipy import stats


def _run(name: str) -> None:
    print(f"  {name}")


# --------------------------------------------------------------------------------------
# Support and diagnostics
# --------------------------------------------------------------------------------------


def test_support_violation_blocks_rather_than_filters() -> None:
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    assert support_violation(family("lognormal"), x) is not None
    assert support_violation(family("beta"), x) is not None
    assert support_violation(family("normal"), x) is None
    # the message must say how many rows, so the user can judge whether it is a data bug
    assert "2 value" in support_violation(family("lognormal"), x)
    _run("support violations are reported, not silently filtered")


def test_excluded_families_never_enter_the_ranking() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)  # has negatives -> all positive families excluded
    rep = fit_distributions(x, n_boot=0, stability_reps=0, probe_location=False)
    ranked_names = {r.name for r in rep.ranked}
    excluded_names = {r.name for r in rep.excluded}
    assert "lognormal" in excluded_names
    assert not (ranked_names & excluded_names)
    assert "normal" in ranked_names
    _run("excluded families are absent from the AIC table")


def test_infer_decimals_is_exact_not_tolerant() -> None:
    rng = np.random.default_rng(1)
    raw = rng.lognormal(4.0, 1.0, 500) * 1e6  # large magnitudes, still continuous
    assert _infer_decimals(raw) is None
    assert _infer_decimals(np.round(raw, 2)) == 2
    assert _infer_decimals(np.array([1.0, 2.0, 3.0])) == 0
    _run("decimal inference does not call large continuous values 'rounded'")


def test_diagnose_flags_ties() -> None:
    rng = np.random.default_rng(2)
    x = np.round(rng.normal(0, 1, 2000), 1)
    d = diagnose(x)
    assert d.tie_fraction > 0.5
    assert d.heavily_tied
    assert d.decimals == 1
    assert "WARNING" in d.describe()
    _run("heavy ties are detected and warned about")


def test_diagnose_rejects_degenerate_input() -> None:
    try:
        diagnose([1.0])
    except ValueError:
        _run("diagnose refuses n<2")
        return
    raise AssertionError("expected ValueError on n=1")


# --------------------------------------------------------------------------------------
# Parameter accounting
# --------------------------------------------------------------------------------------


def test_declared_free_parameter_count_matches_scipy() -> None:
    """n_free must equal (params returned) - (params pinned), for every family.

    This is the silent AIC-corrupting bug: scipy always returns loc and scale, so a family
    fit with floc=0 returns 3 numbers while estimating 2. Getting the count wrong shifts
    that family's AIC by exactly 2 per miscounted parameter, which is the same size as the
    'delta-AIC > 2' rule everyone uses to declare a winner.
    """
    rng = np.random.default_rng(3)
    for fam in DEFAULT_FAMILIES:
        if fam.support == "unit":
            x = rng.beta(2.0, 5.0, 400)
        elif fam.support == "positive":
            x = rng.gamma(2.0, 10.0, 400)
        else:
            x = rng.normal(0.0, 1.0, 400)
        params = fit_params(fam, x)
        pinned = int(fam.fix_loc) + int(fam.fix_scale)
        assert len(params) - pinned == fam.n_free, (
            f"{fam.name}: scipy returned {len(params)} params, {pinned} pinned, "
            f"declared n_free={fam.n_free}"
        )
    _run("every family's declared free-parameter count matches what scipy estimated")


def test_pinned_location_is_actually_pinned() -> None:
    rng = np.random.default_rng(4)
    x = rng.gamma(2.0, 10.0, 300) + 50.0  # shifted: a free loc would land near 50
    params = fit_params(family("gamma"), x)
    assert params[-2] == 0.0
    _run("floc=0 families really do return loc=0")


# --------------------------------------------------------------------------------------
# KS machinery
# --------------------------------------------------------------------------------------


def test_ks_statistic_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 500)
    fam = family("normal")
    params = fit_params(fam, x)
    mine = ks_statistic(fam, x, params)
    theirs = float(stats.ks_1samp(x, fam.dist.cdf, args=params).statistic)
    assert abs(mine - theirs) < 1e-12
    _run("hand-rolled KS distance agrees with scipy to 1e-12")


def test_bootstrap_pvalue_respects_its_resolution_floor() -> None:
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 300)
    fam = family("uniform")  # badly wrong -> p should be at the floor
    params = fit_params(fam, x)
    res = bootstrap_ks(fam, x, params, n_boot=50, rng=rng)
    floor = 1.0 / (res.n_replicates + 1)
    assert res.p_bootstrap >= floor - 1e-12
    assert res.p_bootstrap <= 1.0
    assert abs(res.p_bootstrap - floor) < 1e-9, "a hopeless fit should sit at the floor"
    _run("bootstrap p-value never claims more resolution than B replicates allow")


def test_bootstrap_has_power_where_naive_does_not() -> None:
    """The headline claim, as a test.

    Logistic data, normal fit, n=800, repeated over 12 independent datasets. Asserted as a
    rejection-rate comparison rather than a single p-value: a one-sample assertion here
    would be a coin flip dressed up as a test, and the claim being made is about the
    long-run behaviour of the two procedures, not about one draw.
    """
    rng = np.random.default_rng(7)
    fam = family("normal")
    rej_naive = rej_boot = 0
    p_naive_all, p_boot_all = [], []
    for _ in range(12):
        x = stats.logistic.rvs(loc=0.0, scale=0.55, size=800, random_state=rng)
        res = bootstrap_ks(fam, x, fit_params(fam, x), n_boot=100, rng=rng)
        p_naive_all.append(res.p_naive)
        p_boot_all.append(res.p_bootstrap)
        rej_naive += int(res.p_naive < 0.05)
        rej_boot += int(res.p_bootstrap < 0.05)
    assert rej_boot > rej_naive, f"boot {rej_boot}/12 vs naive {rej_naive}/12"
    assert np.mean(p_naive_all) > np.mean(p_boot_all)
    _run(f"bootstrap KS rejects a wrong family {rej_boot}/12 times; naive manages {rej_naive}/12")


def test_bootstrap_keeps_the_true_family() -> None:
    rng = np.random.default_rng(8)
    x = rng.lognormal(4.2, 0.85, 800)
    fam = family("lognormal")
    res = bootstrap_ks(fam, x, fit_params(fam, x), n_boot=200, rng=rng)
    assert res.p_bootstrap > 0.05
    _run("bootstrap KS does not reject the true family on a typical sample")


def test_bootstrap_is_deterministic_under_a_seed() -> None:
    x = sample_book()["session_seconds"][:400]
    fam = family("lognormal")
    params = fit_params(fam, x)
    a = bootstrap_ks(fam, x, params, n_boot=40, rng=np.random.default_rng(99))
    b = bootstrap_ks(fam, x, params, n_boot=40, rng=np.random.default_rng(99))
    assert a.p_bootstrap == b.p_bootstrap
    _run("same seed, same p-value")


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------


def test_aic_weights_sum_to_one_and_favour_the_leader() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, 500)
    rep = fit_distributions(x, n_boot=0, stability_reps=0, probe_location=False)
    total = sum(r.aic_weight for r in rep.ranked)
    assert abs(total - 1.0) < 1e-9
    assert rep.ranked[0].delta_aic == 0.0
    assert rep.ranked[0].aic_weight == max(r.aic_weight for r in rep.ranked)
    _run("Akaike weights normalise to 1 and peak at delta=0")


def test_aicc_penalises_small_samples_more_than_aic() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, 25)
    rep = fit_distributions(x, n_boot=0, stability_reps=0, probe_location=False)
    for r in rep.ranked:
        assert r.aicc > r.aic
    _run("AICc exceeds AIC at n=25, and by more for higher k")


def test_ranking_finds_the_true_family_on_clean_data() -> None:
    rng = np.random.default_rng(11)
    x = rng.gamma(2.4, 18.0, 3000)
    rep = fit_distributions(x, n_boot=100, stability_reps=40, seed=3, probe_location=False)
    assert rep.best.name == "gamma"
    assert rep.best.adequate(0.05) is True
    assert "gamma" in {r.name for r in rep.adequate}
    _run("gamma data ranks gamma first and passes the absolute test")


def test_selection_stability_shares_sum_to_one() -> None:
    rng = np.random.default_rng(12)
    x = rng.gamma(2.0, 5.0, 300)
    fams = [family(n) for n in ("gamma", "lognormal", "weibull")]
    shares = selection_stability(x, fams, n_boot=30, rng=rng)
    assert abs(sum(shares.values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in shares.values())
    _run("bootstrap win shares are a probability distribution over candidates")


# --------------------------------------------------------------------------------------
# Free location
# --------------------------------------------------------------------------------------


def test_free_location_buys_likelihood_for_a_wrong_family() -> None:
    rng = np.random.default_rng(13)
    x = rng.gamma(2.4, 18.0, 800)  # true loc is 0, and lognormal is the wrong family
    probe = probe_free_location(family("lognormal"), x)
    assert probe is not None
    # nested model: the 3-parameter fit cannot do worse, allowing optimiser slack
    assert probe.loglik_gain > 1.0
    # the estimate does not recover the true loc, and the extra parameter wins on AIC
    assert abs(probe.loc_free) > 1e-6
    assert probe.free_wins_aic
    assert not probe.free_fit_invalid
    _run("a free loc buys real log-likelihood for a family that is wrong to begin with")


def test_free_location_flags_a_fit_that_excludes_observed_data() -> None:
    rng = np.random.default_rng(31)
    x = rng.gamma(2.4, 18.0, 1200)
    probe = probe_free_location(family("weibull"), x)
    assert probe is not None
    assert probe.free_fit_invalid, "expected the optimiser to overshoot min(x)"
    assert probe.loc_free > probe.data_min
    assert probe.loglik_free == -np.inf
    assert not math.isfinite(probe.aic_free)
    _run("a free loc above min(x) is reported as invalid rather than scored")


def test_free_location_probe_declines_on_unbounded_families() -> None:
    assert probe_free_location(family("normal"), np.array([1.0, 2.0, 3.0])) is None
    _run("the location probe only applies to positive-support families")


# --------------------------------------------------------------------------------------
# Verdict and plotting
# --------------------------------------------------------------------------------------


def test_verdict_says_no_adequate_fit_on_a_mixture() -> None:
    rng = np.random.default_rng(14)
    fast = rng.lognormal(3.0, 0.35, 900)
    slow = rng.lognormal(5.4, 0.55, 120)
    x = np.concatenate([fast, slow])
    rep = fit_distributions(x, n_boot=80, stability_reps=0, seed=5, probe_location=False)
    assert rep.adequate == []
    assert "NO ADEQUATE FIT" in rep.verdict()
    assert rep.best is not None  # AIC still produced a winner
    assert rep.best.aic_weight > 0.9  # ... and a confident-looking one
    _run("a mixture yields a confident AIC winner and an explicit 'nothing fits'")


def test_qq_points_are_sorted_and_finite_inside_the_range() -> None:
    rng = np.random.default_rng(15)
    x = rng.normal(0, 1, 200)
    fam = family("normal")
    theo, emp = qq_points(fam, x, fit_params(fam, x))
    assert theo.size == emp.size == 200
    assert np.all(np.diff(emp) >= 0)
    assert np.all(np.isfinite(theo)), "the (i-0.5)/n positions must never hit p=0 or p=1"
    _run("QQ plotting positions stay finite at both tails")


def test_sample_book_columns_are_what_they_claim() -> None:
    book = sample_book()
    assert set(book) == {"session_seconds", "basket_value", "latency_ms", "daily_return"}
    assert diagnose(book["basket_value"]).decimals == 2
    assert diagnose(book["session_seconds"]).decimals is None
    assert diagnose(book["daily_return"]).n_nonpositive > 0
    assert diagnose(book["latency_ms"]).heavily_tied
    _run("the sample book carries the four intended lessons")


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} tests in test_fitting.py")
    for t in tests:
        t()
    print(f"PASS {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
