"""Tests for the percentage/apportionment audit.

Three kinds of assertion, and the second and third are the point of the file:

1. per-method behaviour against that method's own rule
2. **disagreement** assertions - two methods handed one table, with both answers
   pinned. If an edit makes them agree, that is a modelling regression
3. **theorem** assertions - the properties Balinski and Young proved: the
   quota-respecting method takes the paradox, the paradox-free methods take the
   quota violation, and no method in the table escapes both
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from percentages import (
    CENSUS_AFTER,
    CENSUS_BEFORE,
    COMMITTEE,
    CORPUS,
    COUNCIL,
    GROUPED,
    METHODS,
    NEWCOMER,
    NEWCOMER_EXTRA,
    NEWCOMER_ROW,
    PNL,
    POWERS,
    QUARTERS,
    QUEUES,
    SHIFTS,
    SHORTLIST,
    SURVEY7,
    THIRDS,
    TRAFFIC,
    ZERO_BASE,
    Table,
    Verdict,
    alabama,
    audit,
    audit_corpus,
    largest_remainder,
    naive_half_even,
    naive_half_up,
    new_state_paradox,
    no_method_is_clean,
    percent_table,
    population_paradox,
    quota_violations,
    representable_step,
    seat_table,
    size_bias,
    subtotal_clash,
)

APPORTIONMENT = ("largest_remainder", "jefferson_dhondt", "webster_sainte_lague",
                 "adams", "huntington_hill")
DEFINED = tuple(t for t in CORPUS if t.is_definable()[0])


# ---------------------------------------------------------------------------
# The quota is the only unarguable number
# ---------------------------------------------------------------------------


def test_quotas_are_exact_fractions_not_floats():
    q = THIRDS.quotas()
    assert q == (Fraction(1000, 3),) * 3
    assert sum(q) == 1000  # exactly, which floats would not give


def test_quotas_sum_to_the_budget_on_every_defined_table():
    for t in DEFINED:
        assert sum(t.quotas()) == t.units


# ---------------------------------------------------------------------------
# Independent rounding is the only family that fails to sum
# ---------------------------------------------------------------------------


def test_three_equal_rows_sum_to_999():
    a = naive_half_up(THIRDS)
    assert a.units == (333, 333, 333)
    assert a.total == 999
    assert a.percents(THIRDS) == (33.3, 33.3, 33.3)


def test_apportionment_methods_always_sum_exactly():
    for t in DEFINED:
        for name in APPORTIONMENT:
            a = METHODS[name](t)
            if a.notes and any("no answer" in n for n in a.notes):
                continue  # adams/HH on a table with fewer units than rows
            assert a.total == t.units, (t.name, name, a.units)


def test_naive_rounding_fails_to_sum_on_most_of_the_corpus():
    fails = [t.name for t in DEFINED if not naive_half_up(t).sums_to(t)]
    assert len(fails) >= 8
    assert "quarters" not in fails  # the one table whose denominator divides


def test_half_up_and_half_even_differ_on_an_exact_half():
    up, even = naive_half_up(POWERS), naive_half_even(POWERS)
    assert POWERS.quotas()[0] == Fraction(125, 2)  # exactly 62.5
    assert up.units[0] == 63 and even.units[0] == 62
    assert up.units != even.units


# ---------------------------------------------------------------------------
# Largest remainder: in quota, and order-dependent on ties
# ---------------------------------------------------------------------------


def test_largest_remainder_never_leaves_the_quota():
    for t in DEFINED:
        assert quota_violations(t, largest_remainder(t)) == (), t.name


def test_tied_remainders_are_broken_by_input_order():
    a = largest_remainder(THIRDS)
    assert a.units == (334, 333, 333)
    reversed_table = Table(THIRDS.name, THIRDS.rows[::-1], THIRDS.units, THIRDS.kind,
                           THIRDS.decimals)
    b = largest_remainder(reversed_table)
    # The extra tenth of a point follows the input order, not the data.
    assert b.units == (334, 333, 333)
    assert dict(zip(THIRDS.labels, a.units)) != dict(zip(reversed_table.labels, b.units))
    assert any("input order" in n for n in a.notes)


# ---------------------------------------------------------------------------
# The paradoxes, on the tables found by exhaustive search
# ---------------------------------------------------------------------------


def test_alabama_paradox_on_the_committee():
    hits = alabama(COMMITTEE, "largest_remainder")
    assert hits == (("legal", 1, 0),)
    assert largest_remainder(COMMITTEE).units == (2, 4, 1)
    bigger = seat_table(COMMITTEE.name, [(r.label, r.value) for r in COMMITTEE.rows], 8)
    assert largest_remainder(bigger).units == (3, 5, 0)


def test_divisor_methods_are_free_of_the_alabama_paradox_on_the_corpus():
    for t in DEFINED:
        for name in ("jefferson_dhondt", "webster_sainte_lague", "huntington_hill"):
            assert alabama(t, name) == (), (t.name, name)


def test_population_paradox_the_faster_grower_loses():
    hits = population_paradox(CENSUS_BEFORE, CENSUS_AFTER, "largest_remainder")
    assert hits and hits[0][0] == "south" and hits[0][1] == "east"
    loser_rate, gainer_rate = hits[0][2], hits[0][3]
    assert loser_rate > gainer_rate
    assert largest_remainder(CENSUS_BEFORE).units == (7, 0, 6)
    assert largest_remainder(CENSUS_AFTER).units == (7, 1, 5)


def test_divisor_methods_are_free_of_the_population_paradox():
    for name in ("jefferson_dhondt", "webster_sainte_lague", "huntington_hill", "adams"):
        assert population_paradox(CENSUS_BEFORE, CENSUS_AFTER, name) == (), name


def test_new_state_paradox_moves_two_rows_that_did_not_change():
    hits = new_state_paradox(NEWCOMER, NEWCOMER_ROW, NEWCOMER_EXTRA, "largest_remainder")
    moved = {label: (b, a) for label, b, a in hits}
    assert moved == {"centre": (4, 3), "hills": (2, 3)}


# ---------------------------------------------------------------------------
# Balinski-Young: every method in the table has a witness against it
# ---------------------------------------------------------------------------


def test_jefferson_violates_the_upper_quota_on_the_council():
    v = quota_violations(COUNCIL, METHODS["jefferson_dhondt"](COUNCIL))
    assert v and v[0][0] == "blue"
    label, awarded, quota = v[0]
    assert awarded == 5 and 3.9 < quota < 4.0   # more than the ceiling of its share


def test_webster_and_huntington_hill_also_violate_quota_somewhere():
    # Balinski-Young says a paradox-free method must; these are the witnesses.
    assert quota_violations(QUEUES, METHODS["webster_sainte_lague"](QUEUES))
    assert quota_violations(SHIFTS, METHODS["huntington_hill"](SHIFTS))


def test_no_method_escapes_every_failure_on_this_corpus():
    table = no_method_is_clean()
    for method, (sum_fail, quota_fail, alabama_hits) in table.items():
        assert sum_fail + quota_fail + alabama_hits > 0, method


def test_the_two_families_fail_in_opposite_ways():
    t = no_method_is_clean()
    lr_sum, lr_quota, lr_alabama = t["largest_remainder"]
    assert (lr_quota, lr_sum) == (0, 0) and lr_alabama > 0   # keeps quota, takes the paradox
    for name in ("jefferson_dhondt", "webster_sainte_lague", "huntington_hill"):
        _, quota_fail, alabama_hits = t[name]
        assert alabama_hits == 0 and quota_fail > 0           # paradox-free, breaks quota


# ---------------------------------------------------------------------------
# Three failures specific to percentages rather than seats
# ---------------------------------------------------------------------------


def test_a_denominator_of_seven_cannot_produce_a_tenth_of_a_point():
    step = representable_step(SURVEY7)
    assert step == pytest.approx(100 / 7)
    codes = {f.code for f in audit(SURVEY7).findings}
    assert "UNREPRESENTABLE_PRECISION" in codes


def test_a_denominator_that_divides_raises_no_precision_finding():
    codes = {f.code for f in audit(QUARTERS).findings}
    assert "UNREPRESENTABLE_PRECISION" not in codes
    assert representable_step(QUARTERS) == pytest.approx(0.1)


def test_a_two_level_table_cannot_be_consistent_at_both_levels():
    clash = subtotal_clash(GROUPED)
    assert clash
    group, rows_sum, own = clash[0]
    assert abs(rows_sum - own) == 1  # one tenth of a point, unfixable by rounding
    assert "SUBTOTAL_CLASH" in {f.code for f in audit(GROUPED).findings}


def test_a_flat_table_has_no_subtotal_clash():
    assert subtotal_clash(TRAFFIC) == ()


def test_mixed_signs_have_no_share_at_all():
    a = audit(PNL)
    assert a.verdict is Verdict.UNDEFINED
    assert {f.code for f in a.findings} == {"MIXED_SIGN_BASE"}
    assert a.allocations == ()


def test_a_zero_base_is_undefined_not_zero_percent():
    a = audit(ZERO_BASE)
    assert a.verdict is Verdict.UNDEFINED
    assert {f.code for f in a.findings} == {"NO_SHARE_DEFINED"}


# ---------------------------------------------------------------------------
# Rows that exist and print as nothing
# ---------------------------------------------------------------------------


def test_a_real_row_prints_as_zero_under_most_methods():
    a = audit(TRAFFIC)
    zeros = [f for f in a.findings if f.code == "ZERO_ROW_NONZERO_VALUE"]
    assert zeros
    tail = TRAFFIC.rows[-1]
    assert 0 < tail.value < TRAFFIC.total / 1000


def test_adams_and_huntington_hill_cannot_print_a_zero_row():
    for name in ("adams", "huntington_hill"):
        a = METHODS[name](TRAFFIC)
        assert all(u > 0 for u in a.units), name
        assert a.sums_to(TRAFFIC)


def test_min_one_methods_have_no_answer_when_units_are_scarcer_than_rows():
    for name in ("adams", "huntington_hill"):
        a = METHODS[name](SHORTLIST)
        assert any("no answer" in n for n in a.notes), name
        assert not a.sums_to(SHORTLIST)
    assert "METHOD_HAS_NO_ANSWER" in {f.code for f in audit(SHORTLIST).findings}


# ---------------------------------------------------------------------------
# Bias and the residual hacks
# ---------------------------------------------------------------------------


def test_dhondt_serves_large_rows_better_than_adams_does():
    big = size_bias(COUNCIL, METHODS["jefferson_dhondt"](COUNCIL))
    small = size_bias(COUNCIL, METHODS["adams"](COUNCIL))
    assert big > small


def test_the_residual_hacks_name_the_row_they_charge():
    for name in ("last_row_dump", "largest_row_dump"):
        a = METHODS[name](THIRDS)
        assert a.sums_to(THIRDS)
        assert any("residual" in n for n in a.notes), name


def test_the_dump_hacks_can_leave_the_quota():
    # Pushing the whole residual onto one row is not a rounding of that row's share.
    offenders = [t.name for t in DEFINED
                 if quota_violations(t, METHODS["largest_row_dump"](t))]
    assert offenders


# ---------------------------------------------------------------------------
# Verdicts and corpus accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table,verdict",
    [
        (QUARTERS, Verdict.CONSISTENT),
        (THIRDS, Verdict.RESIDUAL),
        (COUNCIL, Verdict.CONTESTED),
        (SURVEY7, Verdict.CONTESTED),
        (PNL, Verdict.UNDEFINED),
        (ZERO_BASE, Verdict.UNDEFINED),
    ],
)
def test_verdicts(table, verdict):
    assert audit(table).verdict is verdict


def test_contested_means_more_than_a_residual_apart():
    a = audit(COUNCIL)
    assert a.max_row_gap() == 2          # blue gets 3, 4 or 5 seats
    assert "blue" in a.disagreeing_rows()


def test_consistent_table_has_no_disagreement_at_all():
    a = audit(QUARTERS)
    assert a.disagreeing_rows() == ()
    assert a.max_row_gap() == 0
    assert all(al.sums_to(QUARTERS) for al in a.allocations)


def test_every_finding_code_fires_somewhere_in_the_corpus():
    rep = audit_corpus()
    unused = [c for c, n in rep.finding_counts.items() if n == 0]
    assert unused == [], f"finding codes with no evidence: {unused}"


def test_corpus_accounting_adds_up():
    rep = audit_corpus()
    assert sum(rep.verdicts.values()) == rep.total == len(CORPUS)
    assert rep.verdicts["consistent"] == 1  # exactly one clean table in fifteen


def test_balinski_young_is_reported_on_every_defined_table():
    for t in DEFINED:
        assert "BALINSKI_YOUNG" in {f.code for f in audit(t).findings}, t.name


def test_audit_is_deterministic():
    for t in CORPUS:
        first, second = audit(t), audit(t)
        assert [a.units for a in first.allocations] == [a.units for a in second.allocations]
        assert [f.code for f in first.findings] == [f.code for f in second.findings]


def test_percent_and_seat_tables_use_the_same_engine():
    # A 1 dp percentage column is an apportionment of 1000 units. Same code path.
    as_percent = percent_table("x", [(r.label, r.value) for r in COUNCIL.rows])
    assert as_percent.units == 1000
    assert sum(largest_remainder(as_percent).units) == 1000
    assert sum(largest_remainder(COUNCIL).units) == 9


def test_a_single_row_takes_everything():
    t = percent_table("solo", [("only", 42)])
    for name in METHODS:
        a = METHODS[name](t)
        assert a.units == (1000,), name
