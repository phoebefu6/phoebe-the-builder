"""Every claim in the README is asserted here against the model that produced it.

Run:  python3 -m pytest test_codelay.py -q
"""

from __future__ import annotations

import math

import pytest

import codelay as C


@pytest.fixture(scope="module")
def items():
    return C.backlog()


@pytest.fixture(scope="module")
def lin(items):
    return C.linearised(items)


@pytest.fixture(scope="module")
def sr(items):
    return C.sweep(items)


@pytest.fixture(scope="module")
def sl(lin):
    return C.sweep(lin)


@pytest.fixture(scope="module")
def costs(items):
    return C.all_costs(items)


# ------------------------------------------------------------------ the shapes


def test_linear_cum_is_rate_times_time():
    cod = C.CoD("linear", r=38.0)
    assert cod.cum(10.0) == pytest.approx(380.0)
    assert cod.rate(0.0) == cod.rate(39.0) == 38.0


def test_deadline_is_free_before_the_date_and_not_after():
    cod = C.CoD("deadline", r2=180.0, t_break=26.0)
    assert cod.cum(25.9) == 0.0
    assert cod.cum(40.0) == pytest.approx(180.0 * 14)
    assert cod.rate(25.9) == 0.0 and cod.rate(26.0) == 180.0


def test_window_saturates_so_late_delay_is_nearly_free():
    cod = C.CoD("window", r=70.0, tau=10.0)
    early = cod.cum(10.0) - cod.cum(0.0)
    late = cod.cum(40.0) - cod.cum(30.0)
    assert early / late == pytest.approx(20.0, abs=0.5)
    assert cod.cum(1e6) == pytest.approx(700.0)  # bounded total


def test_step_escalates_at_the_break():
    cod = C.CoD("step", r=5.0, r2=45.0, t_break=20.0)
    assert cod.cum(20.0) == pytest.approx(100.0)
    assert cod.cum(40.0) == pytest.approx(1000.0)


def test_a_zero_rate_item_can_be_the_most_expensive_one(items):
    d, b = items["D"], items["B"]
    assert d.cod.rate(0.0) == 70.0 and b.cod.rate(0.0) == 0.0
    assert b.cod.cum(40.0) > d.cod.cum(40.0) * 3


# --------------------------------------------------------------- the backlog


def test_backlog_shape(items):
    assert len(items) == 9
    assert sum(i.duration for i in items.values()) == 40.0
    assert sum(1 for i in items.values() if i.cod.kind != "linear") == 4


def test_effort_and_duration_differ_for_parallelisable_items(items):
    assert items["A"].person_weeks == 2 * items["A"].duration
    assert items["C"].person_weeks == items["C"].duration


# ------------------------------------------------- Smith's rule, and its limits


def test_cd3_is_exactly_optimal_on_a_linear_backlog(lin, sl):
    cd3 = C.order_cd3_mean(lin)
    assert cd3 == sl["best_order"]
    assert C.cost_of(cd3, lin) == pytest.approx(sl["best"], abs=1e-9)


def test_the_enumeration_is_complete(sl):
    assert sl["count"] == math.factorial(9) == 362880


def test_sweep_best_agrees_with_the_public_cost_function(items, sr):
    assert C.cost_of(sr["best_order"], items) == pytest.approx(sr["best"])


def test_cd3_loses_18_7_percent_once_the_shapes_are_real(items, sr):
    cd3 = C.cost_of(C.order_cd3_mean(items), items)
    assert sr["best"] == pytest.approx(3288.5, abs=0.1)
    assert cd3 == pytest.approx(3904.6, abs=0.1)
    assert 100 * (cd3 / sr["best"] - 1) == pytest.approx(18.7, abs=0.1)


# ----------------------------------------------- one method, three orderings


def test_cd3_does_not_name_an_ordering(items):
    a = C.order_cd3_initial(items)
    b = C.order_cd3_mean(items)
    p = C.order_cd3_peak(items)
    assert a != b != p != a
    assert C.kendall_distance(a, b) == 18  # exactly half of the 36 pairs
    assert C.kendall_distance(b, p) == 8
    assert C.kendall_distance(a, p) == 14


def test_the_elicitation_costs_more_than_the_method_choice(items):
    spread = (C.cost_of(C.order_cd3_initial(items), items)
              - C.cost_of(C.order_cd3_mean(items), items))
    assert spread == pytest.approx(2323.0, abs=0.5)


# -------------------------------------------------------- worse than a hat


def test_four_orderings_lose_to_the_mean_of_all_orderings(items, sr):
    losers = {n for n, f in C.ORDERINGS.items()
              if C.cost_of(f(items), items) > sr["mean"]}
    assert losers == {"rice", "rice_duration", "cd3_initial", "hippo"}


def test_percentiles_are_exact(items, costs):
    assert len(costs) == 362880
    pct = {n: round(100 * C.percentile_of(costs, C.cost_of(f(items), items)), 1)
           for n, f in C.ORDERINGS.items()}
    assert pct["cd3_initial"] == 90.0
    assert pct["cd3_mean"] == 5.4
    assert pct["rice"] == 86.3
    assert pct["hippo"] == 99.8


def test_hippo_is_nearly_the_worst_ordering_available(items, sr, costs):
    h = C.cost_of(C.order_hippo(items), items)
    assert C.percentile_of(costs, h) > 0.99
    assert h < sr["worst"]


def test_rice_denominator_swap_is_a_rounding_error(items):
    r1, r2 = C.order_rice(items), C.order_rice_duration(items)
    assert C.kendall_distance(r1, r2) == 3
    delta = C.cost_of(r2, items) - C.cost_of(r1, items)
    assert delta == pytest.approx(26.8, abs=0.5)
    assert delta > 0  # the "fix" is very slightly worse


# ----------------------------------------------------------------- the date


def test_no_method_schedules_to_the_date(items, sr):
    date = items["B"].cod.t_break
    fins = {n: C.completions(f(items), items)["B"] for n, f in C.ORDERINGS.items()}
    assert fins["cd3_initial"] == 40.0 and fins["hippo"] == 40.0
    assert fins["cd3_mean"] == 4.0
    assert fins["rice"] == 36.0
    opt_fin = C.completions(sr["best_order"], items)["B"]
    assert opt_fin == 25.0 and date - opt_fin == 1.0
    assert all(abs(v - date) > 5 for v in fins.values())


def test_the_penalty_for_missing_the_date(items):
    o = C.order_cd3_initial(items)
    assert C.cost_breakdown(o, items)["B"] == pytest.approx(2520.0)


def test_the_highest_current_cost_of_delay_belongs_last(items, sr):
    assert max(items, key=lambda k: items[k].cod.rate(0.0)) == "D"
    assert sr["best_order"][-1] == "D"
    d_first = ["D"] + [k for k in sr["best_order"] if k != "D"]
    assert C.cost_of(d_first, items) - sr["best"] == pytest.approx(389.6, abs=0.5)


# ------------------------------------------------------------- two teams


def test_list_scheduling_is_near_optimal_on_two_teams(lin):
    exact = C.optimal_two_team_assignment(lin)["best"]
    greedy = C.parallel_cost(C.order_cd3_mean(lin), lin, 2)
    assert 100 * (greedy / exact - 1) < 0.5  # under half a percent


def test_the_assignment_search_agrees_with_full_enumeration(lin):
    assert (C.optimal_two_team_assignment(lin)["best"]
            == pytest.approx(C.sweep(lin, teams=2)["best"]))


def test_doubling_the_teams_does_not_halve_the_cost(lin, sl):
    two = C.optimal_two_team_assignment(lin)["best"]
    assert 100 * (1 - two / sl["best"]) == pytest.approx(41.5, abs=0.3)


def test_two_team_optimum_refuses_non_linear_shapes(items):
    with pytest.raises(ValueError):
        C.optimal_two_team_assignment(items)


# ------------------------------------------------------------ precedence


def test_precedence_rules_out_three_quarters_of_the_orderings(items):
    sp = C.sweep(items, edges=C.PRECEDENCE)
    assert sp["count"] == 90720
    assert sp["count"] / math.factorial(9) == pytest.approx(0.25)


def test_the_constraint_is_cheap_and_the_repair_is_not(items, sr):
    sp = C.sweep(items, edges=C.PRECEDENCE)
    assert sp["best"] - sr["best"] == pytest.approx(40.0, abs=0.5)
    raw = C.order_cd3_mean(items)
    assert not all(raw.index(a) < raw.index(b) for a, b in C.PRECEDENCE)
    rep = C.repair_precedence(raw, C.PRECEDENCE)
    assert all(rep.index(a) < rep.index(b) for a, b in C.PRECEDENCE)
    assert C.cost_of(rep, items) - C.cost_of(raw, items) == pytest.approx(52.3, abs=0.5)
    assert C.cost_of(rep, items) - sp["best"] == pytest.approx(628.4, abs=0.5)


def test_repair_preserves_the_item_set(items):
    rep = C.repair_precedence(C.order_cd3_initial(items), C.PRECEDENCE)
    assert sorted(rep) == sorted(items)


# ----------------------------------------------------------- estimate noise


def test_the_rank_never_survives_ordinary_estimate_error(items):
    r = C.noise_sweep(items, 0.35, 2000)
    assert r["reorder_rate"] > 0.99


def test_the_cost_survives_better_than_the_rank(items, sr):
    r = C.noise_sweep(items, 0.35, 2000)
    added = r["mean"] - r["truth_cost"]
    method_gap = C.cost_of(C.order_cd3_mean(items), items) - sr["best"]
    assert added == pytest.approx(263.2, abs=1.0)
    assert added < method_gap
    assert r["mean"] < sr["mean"]  # still beats a hat, despite the noise


def test_noise_sweep_is_reproducible(items):
    a = C.noise_sweep(items, 0.35, 300)
    b = C.noise_sweep(items, 0.35, 300)
    assert a["mean"] == b["mean"]
    assert a["mean"] != C.noise_sweep(items, 0.35, 300, seed=1)["mean"]


def test_added_cost_grows_with_sigma(items):
    means = [C.noise_sweep(items, s, 600)["mean"] for s in (0.2, 0.35, 0.5, 0.7)]
    assert means == sorted(means)


# ------------------------------------------------------------------ plumbing


def test_orderings_are_permutations(items):
    for name, f in C.ORDERINGS.items():
        o = f(items)
        assert sorted(o) == sorted(items), name


def test_orderings_are_deterministic(items):
    for name, f in C.ORDERINGS.items():
        assert f(items) == f(items), name


def test_kendall_distance_bounds(items):
    o = C.order_cd3_mean(items)
    assert C.kendall_distance(o, o) == 0
    assert C.kendall_distance(o, list(reversed(o))) == 36


def test_readme_claims_about_the_backlog_itself(items):
    """Numbers stated in prose in the README, so prose cannot drift from code."""
    assert sum(1 for i in items.values() if i.person_weeks != i.duration) == 3
    d = items["D"].cod
    assert (d.cum(10) - d.cum(0)) == pytest.approx(442.5, abs=0.1)
    assert (d.cum(40) - d.cum(30)) == pytest.approx(22.0, abs=0.1)
    assert items["B"].cod.cum(C.HORIZON) == pytest.approx(2520.0)
    assert items["D"].cod.cum(C.HORIZON) == pytest.approx(687.2, abs=0.1)


def test_readme_percentage_claims(items, sr):
    rice = C.cost_of(C.order_rice(items), items)
    hippo = C.cost_of(C.order_hippo(items), items)
    assert round(100 * (rice / sr["mean"] - 1)) == 23
    assert round(100 * (hippo / sr["mean"] - 1)) == 43
    assert sr["mean"] - sr["best"] == pytest.approx(1583.2, abs=1.0)
