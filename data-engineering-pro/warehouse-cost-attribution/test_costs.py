"""Every claim the README makes is pinned here.

The cost game is deterministic, so almost everything is asserted exactly. Only the
sampled-Shapley section carries a tolerance.
"""

from __future__ import annotations

import itertools
import math

import pytest

import costs as C


# ------------------------------------------------------------------ the cost function --
def test_the_invoice_is_what_the_full_coalition_costs():
    assert C.INVOICE == pytest.approx(C.coalition_cost(C.TEAM_NAMES))
    assert C.INVOICE == pytest.approx(30_550, abs=60)


def test_an_empty_warehouse_costs_nothing_and_one_user_owes_the_floor():
    assert C.coalition_cost([]) == 0.0
    for n in C.TEAM_NAMES:
        assert C.v([n]) > C.RESERVED_FLOOR


def test_the_cost_function_is_monotone():
    """Adding a team can never make the bill smaller."""
    for S in C.all_coalitions():
        for n in C.TEAM_NAMES:
            if n in S:
                continue
            assert C.v(list(S) + [n]) >= C.v(S) - 1e-9


def test_the_cost_function_is_subadditive_which_is_why_it_is_joint():
    """v(S u T) <= v(S) + v(T) for disjoint S, T -- sharing is real."""
    names = C.TEAM_NAMES
    for k in range(1, 4):
        for S in itertools.combinations(names, k):
            rest = [n for n in names if n not in S]
            for j in range(1, 3):
                for T in itertools.combinations(rest, j):
                    assert C.v(list(S) + list(T)) <= C.v(S) + C.v(T) + 1e-9


def test_the_cost_function_is_not_additive():
    """If it were, none of this would be interesting."""
    assert sum(C.v([n]) for n in C.TEAM_NAMES) > C.INVOICE * 2


def test_every_coalition_is_precomputed():
    assert len(C.COALITION_COST) == 2 ** len(C.TEAM_NAMES)


# ------------------------------------------------------------------ the methods --------
def test_every_method_bills_exactly_the_invoice():
    for name, fn in C.METHODS.items():
        assert sum(fn().values()) == pytest.approx(C.INVOICE, abs=0.01), name


def test_every_method_bills_every_team_a_non_negative_amount():
    for name, fn in C.METHODS.items():
        for team, amount in fn().items():
            assert amount >= -1e-9, (name, team)


def test_the_methods_disagree_by_more_than_a_hundredfold_on_one_team():
    allocs = {n: f() for n, f in C.METHODS.items()}
    ratios = {}
    for t in C.TEAM_NAMES:
        vals = [allocs[m][t] for m in allocs]
        ratios[t] = max(vals) / max(min(vals), 1.0)
    assert max(ratios.values()) > 100
    assert max(ratios, key=lambda k: ratios[k]) == "exec_reporting"


def test_three_different_teams_are_the_most_expensive_depending_on_method():
    tops = {m: max(f(), key=lambda k: f()[k]) for m, f in C.METHODS.items()}
    assert len(set(tops.values())) == 3


# ------------------------------------------------------------------ marginal / standalone
def test_marginal_cost_leaves_most_of_the_invoice_unfunded():
    raw = C.raw_marginal()
    total = sum(raw.values())
    assert total / C.INVOICE == pytest.approx(0.099, abs=0.01)
    assert C.INVOICE - total > 0.85 * C.INVOICE


def test_no_marginal_cost_is_negative():
    for n, m in C.raw_marginal().items():
        assert m >= 0, n


def test_standalone_cost_over_recovers_by_more_than_threefold():
    total = sum(C.raw_standalone().values())
    assert total / C.INVOICE == pytest.approx(3.25, abs=0.10)


def test_the_value_of_sharing_is_the_gap_between_them():
    gap = sum(C.raw_standalone().values()) - C.INVOICE
    assert gap == pytest.approx(68_666, rel=0.02)


# ------------------------------------------------------------------ shapley -------------
def test_shapley_is_efficient():
    assert sum(C.shapley().values()) == pytest.approx(C.INVOICE, abs=0.01)


def test_shapley_charges_a_dummy_team_exactly_its_own_marginal_cost():
    """A team that adds the same amount to every coalition must be billed that amount."""
    base = C.coalition_cost

    def patched(members):
        members = list(members)
        if "ghost" in members:
            return base([m for m in members if m != "ghost"]) + 100.0
        return base(members)

    names = C.TEAM_NAMES[:3] + ["ghost"]
    phi = {n: 0.0 for n in names}
    for order in itertools.permutations(names):
        run, prev = [], 0.0
        for n in order:
            run.append(n)
            cur = patched(run) if "ghost" in run else base(run)
            phi[n] += cur - prev
            prev = cur
    assert phi["ghost"] / math.factorial(len(names)) == pytest.approx(100.0, abs=1e-6)


def test_shapley_gives_equal_shares_to_symmetric_teams():
    """Give finance an exact twin -- same reads AND same model consumption -- and the two
    must be billed identically. Symmetry is one of the four axioms Shapley is unique for."""
    twin = C.Team("finance_twin", dict(C.TEAM_BY_NAME["finance"].reads))
    saved_models = C.MODELS
    C.TEAM_BY_NAME["finance_twin"] = twin
    C.MODELS = [
        C.Model(m.name, m.build_gb,
                m.consumers | ({"finance_twin"} if "finance" in m.consumers else set()))
        for m in saved_models
    ]
    try:
        assert C.coalition_cost(["finance"]) == pytest.approx(C.coalition_cost(["finance_twin"]))
        names = ["analytics", "finance", "finance_twin", "ml_platform"]
        phi = {n: 0.0 for n in names}
        for order in itertools.permutations(names):
            run, prev = [], 0.0
            for n in order:
                run.append(n)
                cur = C.coalition_cost(run)
                phi[n] += cur - prev
                prev = cur
        f = math.factorial(len(names))
        assert phi["finance"] / f == pytest.approx(phi["finance_twin"] / f, abs=1e-6)
        assert phi["finance"] / f > 0
    finally:
        C.MODELS = saved_models
        del C.TEAM_BY_NAME["finance_twin"]


def test_sampled_shapley_converges_on_the_exact_value():
    exact = C.shapley()
    coarse = C.sampled_shapley(50, seed=1)
    fine = C.sampled_shapley(5_000, seed=1)
    e_coarse = max(abs(coarse[n] - exact[n]) for n in C.TEAM_NAMES)
    e_fine = max(abs(fine[n] - exact[n]) for n in C.TEAM_NAMES)
    assert e_fine < e_coarse
    assert e_fine / C.INVOICE < 0.01, "5k orderings should land inside 1% of the invoice"


def test_sampled_shapley_is_still_efficient_at_any_sample_size():
    for m in (10, 100):
        assert sum(C.sampled_shapley(m, seed=3).values()) == pytest.approx(C.INVOICE, abs=0.01)


# ------------------------------------------------------------------ the core ------------
def test_the_core_is_not_empty():
    assert C.core_is_nonempty()


def test_shapley_lands_inside_the_core():
    assert C.in_core(C.shapley())


def test_the_core_is_wide_enough_to_be_useless_as_a_single_answer():
    widths = {}
    for n in C.TEAM_NAMES:
        lo, hi = C.core_range(n)
        assert lo is not None and hi is not None
        assert lo <= hi
        widths[n] = (hi - lo) / C.INVOICE
    assert max(widths.values()) > 0.60, "some team's fair range spans most of the invoice"
    assert widths["growth"] == pytest.approx(0.716, abs=0.03)


def test_the_core_lower_bound_is_the_marginal_cost():
    """The S = N\\{i} constraint forces every team to pay at least what it costs you."""
    raw = C.raw_marginal()
    for n in C.TEAM_NAMES:
        lo, _ = C.core_range(n)
        assert lo == pytest.approx(raw[n], abs=1.0), n


def test_the_core_rejects_exactly_the_two_rules_people_reach_for_first():
    rejected = [name for name, fn in C.METHODS.items() if not C.in_core(fn())]
    assert set(rejected) == {"direct_bytes", "equal_split"}


def test_equal_split_is_rejected_by_the_two_smallest_teams():
    viol = C.core_violations(C.method_equal_split())
    assert viol
    worst_coalition, excess = viol[0]
    assert set(worst_coalition) == {"finance", "exec_reporting"}
    assert excess == pytest.approx(3_450, rel=0.05)


def test_surviving_methods_still_disagree_enormously():
    survivors = [n for n, f in C.METHODS.items() if C.in_core(f())]
    allocs = {n: C.METHODS[n]() for n in survivors}
    worst = max(max(allocs[m][t] for m in survivors) / max(min(allocs[m][t] for m in survivors), 1.0)
                for t in C.TEAM_NAMES)
    assert worst > 50


# ------------------------------------------------------------------ unattributable -------
def test_the_reservation_is_a_sixth_of_the_bill_and_nobody_s_marginal_cost():
    assert C.RESERVED_FLOOR / C.INVOICE == pytest.approx(0.17, abs=0.01)
    for n in C.TEAM_NAMES:
        without = C.v([m for m in C.TEAM_NAMES if m != n])
        assert without >= C.RESERVED_FLOOR, "the floor survives losing any single team"


def test_blame_and_saving_are_an_order_of_magnitude_apart():
    sh = C.shapley()
    saving = C.unowned_cost()
    assert sh["scheduled_unowned"] / saving > 10
    assert saving / C.INVOICE < 0.02
    assert sh["scheduled_unowned"] / C.INVOICE > 0.20


def test_switching_off_the_orphaned_jobs_saves_almost_nothing():
    owned = [t.name for t in C.TEAMS if t.owned]
    assert C.INVOICE - C.v(owned) == pytest.approx(430, rel=0.05)


# ------------------------------------------------------------------ cache ---------------
def test_the_same_scan_costs_fifty_times_more_if_it_runs_first():
    assert C.cache_ratio("events_raw") == pytest.approx(50.0)
    assert C.query_cost_by_position("events_raw", 0) == pytest.approx(
        C.query_cost_by_position("events_raw", 1) * 50)


def test_first_toucher_bills_half_the_invoice_to_whoever_sorts_first():
    a = C.method_first_toucher()
    assert max(a, key=lambda k: a[k]) == "analytics"
    assert a["analytics"] / C.INVOICE > 0.45


# ------------------------------------------------------------------ end to end ----------
def test_evidence_runs_and_its_headline_numbers_hold():
    import evidence

    s3 = evidence.section_3()
    assert s3["recovery"] < 0.15
    s4 = evidence.section_4()
    assert s4["over_recovery"] > 3.0
    s6 = evidence.section_6()
    assert s6["nonempty"] and set(s6["rejects"]) == {"direct_bytes", "equal_split"}
    s7 = evidence.section_7()
    assert s7["ratio"] > 10
