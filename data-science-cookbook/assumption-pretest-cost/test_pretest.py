"""The measured error rates are the whole product, so everything underneath them is checked.

A false-positive study is worthless if the null it claims to simulate is not actually true, and a
measured error rate is worthless if the statistic producing it is wrong. Both are tested here,
along with the two findings the build rests on.
"""

from __future__ import annotations

import numpy as np
import pretest as P
import pytest
from scipy import stats

# --------------------------------------------------------------------------- the statistics


def test_student_matches_scipy() -> None:
    rng = np.random.default_rng(3)
    x, y = P.draw("normal", 14, 9, 1.0, 2.5, 60, 0.0, rng)
    got = P.student_p(x, y)
    for i in range(x.shape[0]):
        want = stats.ttest_ind(x[i], y[i], equal_var=True).pvalue
        assert got[i] == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_welch_matches_scipy() -> None:
    rng = np.random.default_rng(4)
    x, y = P.draw("normal", 14, 9, 1.0, 2.5, 60, 0.0, rng)
    got = P.welch_p(x, y)
    for i in range(x.shape[0]):
        want = stats.ttest_ind(x[i], y[i], equal_var=False).pvalue
        assert got[i] == pytest.approx(want, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("name", P.PRETESTS)
def test_pretest_matches_scipy(name: str) -> None:
    """Including the hand-rolled F-test, which is the only one not lifted straight from scipy."""
    rng = np.random.default_rng(5)
    x, y = P.draw("normal", 13, 9, 1.0, 2.0, 40, 0.0, rng)
    got = P.pretest_p(name, x, y)
    for i in range(x.shape[0]):
        if name == "bartlett":
            want = stats.bartlett(x[i], y[i]).pvalue
        elif name == "f-test":
            f = x[i].var(ddof=1) / y[i].var(ddof=1)
            lo = stats.f.cdf(f, 12, 8)
            want = 2 * min(lo, 1 - lo)
        else:
            want = stats.levene(x[i], y[i], center={"levene": "mean", "brown-forsythe": "median"}[name]).pvalue
        assert got[i] == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_levene_and_brown_forsythe_are_not_the_same_test() -> None:
    """They are routinely used interchangeably; scipy's default is the median-centred one."""
    rng = np.random.default_rng(6)
    x, y = P.draw("lognormal", 20, 20, 1.0, 1.0, 200, 0.0, rng)
    mean_c, med_c = P.pretest_p("levene", x, y), P.pretest_p("brown-forsythe", x, y)
    assert not np.allclose(mean_c, med_c), "the two centrings must differ on skewed data"
    assert np.allclose(med_c, [stats.levene(x[i], y[i]).pvalue for i in range(x.shape[0])]), \
        "brown-forsythe must be what a bare scipy.stats.levene call does"


def test_degenerate_samples_do_not_raise_or_reject() -> None:
    """Zero variance in both groups: the statistic does not exist. It must not count as evidence."""
    z = np.zeros((3, 8))
    for name in P.PRETESTS:
        p = P.pretest_p(name, z, z)
        assert np.all(np.isfinite(p))
        assert np.all(p >= 0.05), f"{name} treated an undefined statistic as a rejection"


# --------------------------------------------------------------------------- the simulation


@pytest.mark.parametrize("dist", P.DISTRIBUTIONS)
def test_draw_is_centred_and_scaled(dist: str) -> None:
    """The null under test is 'equal means'. If the draw is off-centre, the null is not true."""
    rng = np.random.default_rng(7)
    x, y = P.draw(dist, 300, 300, 1.0, 3.0, 4000, 0.0, rng)
    assert abs(x.mean()) < 0.02, f"{dist} group 1 off-centre: {x.mean():.4f}"
    assert abs(y.mean()) < 0.06, f"{dist} group 2 off-centre: {y.mean():.4f}"
    assert x.std() == pytest.approx(1.0, abs=0.05)
    assert y.std() == pytest.approx(3.0, rel=0.05)


def test_shift_moves_only_the_second_group() -> None:
    rng = np.random.default_rng(8)
    x, y = P.draw("normal", 200, 200, 1.0, 1.0, 3000, 0.75, rng)
    assert abs(x.mean()) < 0.02
    assert y.mean() == pytest.approx(0.75, abs=0.02)


def test_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError):
        P.draw("gaussian-ish", 5, 5, 1.0, 1.0, 2, 0.0, np.random.default_rng(0))


def test_calibration_cell_reads_nominal() -> None:
    """The known-truth anchor: Student's on a balanced equal-variance normal design is EXACT.

    If this drifts, nothing else in the study can be trusted - the harness would be reporting
    its own bug as a finding.
    """
    c = P.run_cell(P.Design(20, 20, 1.0), "brown-forsythe", 20_000, seed=42)
    assert c.student_error == pytest.approx(0.05, abs=0.005)
    assert c.welch_error == pytest.approx(0.05, abs=0.005)
    assert c.gated_error == pytest.approx(0.05, abs=0.005)
    assert c.pretest_reject_rate == pytest.approx(0.05, abs=0.012)


def test_noise_floor_is_narrower_than_the_band() -> None:
    lo, hi = P.noise_floor(runs=5, reps=20_000)
    assert hi - lo < (P.BAND_HI - P.BAND_LO), f"noise {hi - lo:.4f} cannot resolve the band"


def test_pretest_alpha_one_collapses_the_procedure_into_welch() -> None:
    """The known-answer anchor for the alpha sweep: a pretest that always fires IS Welch.

    If this does not hold, the gating logic is wired wrong and every other number is suspect.
    """
    c = P.run_cell(P.Design(50, 10, 2.0), "brown-forsythe", 4_000, pretest_alpha=1.0, seed=13)
    assert c.pretest_reject_rate == 1.0
    assert c.gated_error == c.welch_error


def test_pretest_alpha_zero_collapses_the_procedure_into_student() -> None:
    """And the other end: a pretest that never fires IS Student's."""
    c = P.run_cell(P.Design(50, 10, 2.0), "brown-forsythe", 4_000, pretest_alpha=0.0, seed=13)
    assert c.pretest_reject_rate == 0.0
    assert c.gated_error == c.student_error


# --------------------------------------------------------------------------- the findings


def test_the_gate_hands_student_its_worst_cases() -> None:
    """Finding 1, the mechanism: conditioning on 'pretest passed' makes Student's WORSE.

    This is the counterintuitive one, so it is asserted rather than described. The pretest and
    the pooled standard error are both functions of the sample variances, so the subset that
    passes is the subset where the pooled estimate is most flattering.
    """
    c = P.run_cell(P.Design(50, 10, 2.0), "brown-forsythe", 20_000, seed=17)
    assert c.share_passed > 0.05, "need a real passing subset to condition on"
    assert c.student_error_given_pass > c.student_error, (
        f"expected the passed subset to be worse: {c.student_error_given_pass:.4f} "
        f"vs overall {c.student_error:.4f}"
    )


def test_the_damage_peaks_at_a_moderate_variance_gap() -> None:
    """Finding 2: the gated error is NOT monotone in the variance ratio.

    At a huge gap the pretest always fires and the procedure becomes Welch; at no gap there is
    nothing to get wrong. The cost lives in the middle, which is where an analyst reads p > 0.05
    and writes down that the variances were checked.
    """
    rates = {r: P.run_cell(P.Design(50, 10, r), "brown-forsythe", 20_000, seed=19).gated_error
             for r in (1.0, 1.5, 6.0)}
    assert rates[1.5] > rates[1.0], rates
    assert rates[1.5] > rates[6.0], rates
    assert P.is_broken(rates[1.5], 20_000), "the peak must be a real departure, not noise"


def test_bartlett_reacts_to_SHAPE_not_variance() -> None:
    """Finding 3: Bartlett reads tail weight, on groups whose variances are IDENTICAL.

    Asserted as relationships rather than magic numbers - it is calibrated on normal data, and
    then moves in BOTH directions with shape: up on heavy tails, down on light ones. A test that
    were merely "liberal" would only go one way.
    """
    rate = {
        d: P.run_cell(P.Design(20, 20, 1.0, d), "bartlett", 20_000, seed=22).pretest_reject_rate
        for d in ("normal", "t5", "lognormal", "uniform")
    }
    assert rate["normal"] == pytest.approx(0.05, abs=0.012), rate
    assert P.is_broken(rate["t5"], 20_000) and rate["t5"] > 2 * rate["normal"], rate
    assert P.is_broken(rate["lognormal"], 20_000) and rate["lognormal"] > rate["t5"], rate
    assert P.is_broken(rate["uniform"], 20_000) and rate["uniform"] < rate["normal"] / 2, rate

    robust = P.run_cell(P.Design(20, 20, 1.0, "t5"), "brown-forsythe", 20_000, seed=22)
    assert robust.pretest_reject_rate < rate["t5"] / 2, "brown-forsythe should survive heavy tails"


def test_bartlett_and_the_f_test_are_one_test_at_equal_n() -> None:
    """They are taught as different tests, their p-values differ - and at equal n they are one.

    With two groups of the SAME size, Bartlett's statistic is a monotone function of the sample
    variance ratio, which is what the F-test uses, so the two rank every sample identically and
    split them in the same place: Spearman 1.0 and 100% decision agreement. Worth knowing before
    treating "we also ran an F-test" as corroboration.

    Unequal n breaks the equivalence, because Bartlett's statistic then also depends on the
    pooled variance - but only slightly, so it still is not independent evidence.
    """
    rng = np.random.default_rng(77)
    x, y = P.draw("t5", 15, 15, 1.0, 1.4, 3_000, 0.0, rng)
    bart, ftest = P.pretest_p("bartlett", x, y), P.pretest_p("f-test", x, y)
    assert not np.allclose(bart, ftest), "the p-values genuinely differ"
    assert stats.spearmanr(bart, ftest).statistic == pytest.approx(1.0, abs=1e-12)
    assert ((bart < 0.05) == (ftest < 0.05)).all(), "equal n must give identical decisions"

    rng = np.random.default_rng(77)
    xu, yu = P.draw("t5", 30, 10, 1.0, 1.4, 3_000, 0.0, rng)
    bu, fu = P.pretest_p("bartlett", xu, yu), P.pretest_p("f-test", xu, yu)
    agree = ((bu < 0.05) == (fu < 0.05)).mean()
    assert 0.90 < agree < 1.0, f"unequal n should diverge, but only slightly: {agree:.4f}"


# --------------------------------------------------------------------------- the scorers


def _cell(n1: int, n2: int, ratio: float, pre: float, student: float, welch: float,
          gated: float, reps: int = 20_000) -> P.CellResult:
    return P.CellResult(
        design=P.Design(n1, n2, ratio),
        pretest="brown-forsythe",
        reps=reps,
        alpha=0.05,
        pretest_alpha=0.05,
        pretest_reject_rate=pre,
        student_error=student,
        welch_error=welch,
        gated_error=gated,
        student_error_given_pass=student,
        student_error_given_reject=student,
        share_passed=1.0 - pre,
    )


def test_gated_beats_welch_can_return_true() -> None:
    """A comparison that can only come out one way is not a comparison.

    The headline claim is "the two-stage procedure never beat Welch on this grid". That claim is
    only worth anything if the predicate behind it is capable of saying otherwise.
    """
    better = _cell(20, 20, 2.0, 0.5, 0.09, 0.060, 0.0501)
    assert better.gated_beats_welch
    worse = _cell(20, 20, 2.0, 0.5, 0.09, 0.0501, 0.060)
    assert not worse.gated_beats_welch


def test_broken_flag_is_an_interval_not_a_compare() -> None:
    """Carried over from the sibling build, which shipped this bug and then fixed it."""
    assert not P.is_broken(0.0433, 1_500), "a point estimate inside Monte-Carlo noise is not a finding"
    assert P.is_broken(0.0433, 200_000), "the same rate measured properly IS a finding"
    assert not P.is_broken(0.05, 20_000)
    lo, hi = P.wilson(0.05, 20_000)
    assert lo < 0.05 < hi
    assert P.wilson(0.0, 1_000)[0] == 0.0, "an interval must not run below zero"


def test_grid_reps_can_resolve_the_band() -> None:
    lo, hi = P.wilson(0.05, P.REPS)
    assert (hi - lo) < 2 * (P.BAND_HI - P.BAND_LO), "grid replicate count is too low to judge"


def test_score_gate_counts_the_four_corners() -> None:
    cells = [
        _cell(50, 10, 3.0, 0.9, 0.20, 0.05, 0.06),   # fires, Student unsafe -> true positive
        _cell(20, 20, 3.0, 0.9, 0.050, 0.05, 0.05),  # fires, Student fine   -> false alarm
        _cell(50, 10, 1.5, 0.1, 0.20, 0.05, 0.10),   # quiet, Student unsafe -> miss
        _cell(20, 20, 1.0, 0.1, 0.050, 0.05, 0.05),  # quiet, Student fine   -> true negative
    ]
    s = P.score_gate(cells)
    assert s.cells == 4 and s.unsafe_cells == 2
    assert s.sensitivity == pytest.approx(0.5)
    assert s.specificity == pytest.approx(0.5)
    assert s.false_alarm_cells == 1 and s.missed_cells == 1


def test_score_gate_can_report_a_perfect_gate() -> None:
    cells = [_cell(50, 10, 3.0, 0.9, 0.20, 0.05, 0.06), _cell(20, 20, 1.0, 0.1, 0.050, 0.05, 0.05)]
    s = P.score_gate(cells)
    assert s.sensitivity == pytest.approx(1.0)
    assert s.specificity == pytest.approx(1.0)


def test_worst_cell_picks_the_largest_departure_either_way() -> None:
    # |0.001 - 0.05| = 0.049 beats |0.095 - 0.05| = 0.045, so the conservative cell must win.
    cells = [_cell(20, 20, 1.0, 0.1, 0.05, 0.05, 0.052),
             _cell(50, 10, 1.5, 0.3, 0.13, 0.05, 0.095),
             _cell(10, 50, 3.0, 0.9, 0.001, 0.05, 0.001)]
    assert P.worst_cell(cells).gated_error == 0.001, "a rate far BELOW nominal is also a departure"
    # And the inflated one wins when it is genuinely further from nominal.
    cells[1] = _cell(50, 10, 1.5, 0.3, 0.13, 0.05, 0.140)
    assert P.worst_cell(cells).gated_error == 0.140


def test_design_pairing_direction() -> None:
    """The direction of the n/sd pairing, not the ratio, is what breaks Student's."""
    assert P.Design(50, 10, 3.0).bigger_group_has_bigger_sd is False
    assert P.Design(10, 50, 3.0).bigger_group_has_bigger_sd is True
    assert P.Design(20, 20, 3.0).bigger_group_has_bigger_sd is None
    assert P.Design(50, 10, 1.0).bigger_group_has_bigger_sd is None


def test_material_win_ignores_a_gap_smaller_than_the_measurement() -> None:
    """A 0.0006 gap at 20,000 replicates is not a win, and the headline must not count it.

    The grid produced exactly one cell where the two-stage procedure was nominally closer to
    0.05 than Welch, by 0.0006 - about an eighth of the 99% interval width. Reporting that as
    "1 of 14" would be reading the measurement error as a result.
    """
    tie = _cell(50, 10, 1.0, 0.04, 0.048, 0.0506, 0.0500)
    assert tie.gated_beats_welch, "the naive comparison should still see it"
    assert not tie.gated_beats_welch_materially, "but it must not survive the noise floor"

    real = _cell(50, 10, 2.0, 0.6, 0.20, 0.0900, 0.0505)
    assert real.gated_beats_welch and real.gated_beats_welch_materially


def test_power_comparability_flags_an_inflated_test() -> None:
    """Power from a test that does not hold its false-positive rate is not power."""
    inflated = _cell(50, 10, 1.5, 0.3, 0.1361, 0.0505, 0.1029)
    assert not inflated.student_controls_type_one
    assert inflated.welch_controls_type_one

    fair = _cell(20, 20, 1.0, 0.04, 0.0502, 0.0500, 0.0502)
    assert fair.student_controls_type_one and fair.welch_controls_type_one


def test_type_one_direction_is_three_valued() -> None:
    """Inflated and conservative are opposite findings, so they must not share a label.

    An inflated test's power advantage is the inflation restated; a conservative test's power
    deficit is a real cost. The first draft of the evidence report collapsed both into "not
    comparable" and then contradicted itself in the prose underneath.
    """
    assert _cell(20, 20, 1.0, 0.04, 0.0502, 0.05, 0.05).student_type_one_direction == "ok"
    assert _cell(50, 10, 1.5, 0.30, 0.1361, 0.05, 0.10).student_type_one_direction == "inflated"
    assert _cell(10, 50, 3.0, 0.95, 0.0010, 0.05, 0.05).student_type_one_direction == "conservative"
