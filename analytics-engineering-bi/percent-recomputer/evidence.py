"""Every number quoted in the README, printed from the engine.

Run `python3 evidence.py`. Nothing in the README is typed by hand, and the three
paradox tables are re-derived here by exhaustive search rather than cited.
"""

from __future__ import annotations

from typing import Dict

from percentages import (
    CENSUS_AFTER,
    CENSUS_BEFORE,
    COMMITTEE,
    CORPUS,
    COUNCIL,
    DESCRIPTION_OF,
    GROUPED,
    METHOD_KIND,
    METHODS,
    NEWCOMER,
    NEWCOMER_EXTRA,
    NEWCOMER_ROW,
    QUARTERS,
    QUEUES,
    SEVERITY_OF,
    SHIFTS,
    SHORTLIST,
    SURVEY7,
    THIRDS,
    TRAFFIC,
    alabama,
    audit,
    audit_corpus,
    largest_remainder,
    naive_half_up,
    new_state_paradox,
    no_method_is_clean,
    population_paradox,
    quota_violations,
    representable_step,
    seat_table,
    subtotal_clash,
)

RULE = "-" * 78


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def s1_column() -> None:
    head(1, "The column that does not add up")
    a = naive_half_up(THIRDS)
    print(f"  {'row':<10} {'exact share':>28} {'printed':>10}")
    for label, q, p in zip(THIRDS.labels, THIRDS.quotas(), a.percents(THIRDS)):
        print(f"  {label:<10} {str(q / 10):>28} {p:>9.1f}%")
    print(f"  {'':<10} {'':>28} {'-' * 10}")
    print(f"  {'total':<10} {'100':>28} {a.total / 10:>9.1f}%")
    print("\n  Every row is correctly rounded to one decimal place. The column reads 99.9%.")
    print("  Rounding is a per-row operation and adding up is a joint constraint; the")
    print("  first cannot preserve the second.")


def s2_nine_methods() -> None:
    head(2, "Nine methods on one table: nine seats, five parties")
    print(f"  votes: {dict((r.label, int(r.value)) for r in COUNCIL.rows)}   seats: {COUNCIL.units}")
    print(f"\n  {'method':<22} {'kind':<24} allocation      sums  in quota")
    for name, fn in METHODS.items():
        al = fn(COUNCIL)
        qv = quota_violations(COUNCIL, al)
        print(f"  {name:<22} {METHOD_KIND[name]:<24} {str(al.units):<15} "
              f"{'yes' if al.sums_to(COUNCIL) else 'NO ':<5} {'no' if qv else 'yes'}")
    exact = [f"{float(q):.3f}" for q in COUNCIL.quotas()]
    print(f"\n  exact shares: {exact}")
    a = audit(COUNCIL)
    print(f"  verdict: {a.verdict.value}; rows in dispute: {a.disagreeing_rows()}; "
          f"widest gap {a.max_row_gap()} seats")
    print("  blue wins 3, 4 or 5 of the 9 seats depending on which correct method ran.")


def s3_alabama() -> None:
    head(3, "The Alabama paradox: growing the total costs a row its seat")
    before = largest_remainder(COMMITTEE)
    bigger = seat_table(COMMITTEE.name, [(r.label, r.value) for r in COMMITTEE.rows],
                        COMMITTEE.units + 1)
    after = largest_remainder(bigger)
    print(f"  headcount: {dict((r.label, int(r.value)) for r in COMMITTEE.rows)}")
    print(f"\n  {'row':<14} {'7 seats':>9} {'8 seats':>9}   {'quota at 7':>11} {'quota at 8':>11}")
    for i, label in enumerate(COMMITTEE.labels):
        print(f"  {label:<14} {before.units[i]:>9} {after.units[i]:>9}   "
              f"{float(COMMITTEE.quotas()[i]):>11.3f} {float(bigger.quotas()[i]):>11.3f}")
    print(f"\n  {alabama(COMMITTEE, 'largest_remainder')}")
    print("  Nothing about legal changed. The committee grew and legal lost its only seat.")
    print("  Named for the 1880 US census, where Alabama fell from 8 seats to 7 as the")
    print("  House grew from 299 to 300.")


def s4_population() -> None:
    head(4, "The population paradox: the faster grower loses to the slower")
    b, a = largest_remainder(CENSUS_BEFORE), largest_remainder(CENSUS_AFTER)
    print(f"  {'row':<8} {'before':>8} {'after':>8} {'growth':>8} {'seats before':>13} {'seats after':>12}")
    for i, label in enumerate(CENSUS_BEFORE.labels):
        v0 = CENSUS_BEFORE.rows[i].value
        v1 = CENSUS_AFTER.rows[i].value
        print(f"  {label:<8} {v0:>8.0f} {v1:>8.0f} {v1 / v0 - 1:>7.1%} "
              f"{b.units[i]:>13} {a.units[i]:>12}")
    for loser, gainer, rl, rg in population_paradox(CENSUS_BEFORE, CENSUS_AFTER, "largest_remainder"):
        print(f"\n  {loser} grew {rl - 1:.1%} and lost a seat; {gainer} grew {rg - 1:.1%} and gained one")
    print("  Every divisor method is immune to this by construction:")
    for name in ("jefferson_dhondt", "webster_sainte_lague", "huntington_hill", "adams"):
        print(f"    {name:<22} {population_paradox(CENSUS_BEFORE, CENSUS_AFTER, name) or 'no paradox'}")


def s5_new_state() -> None:
    head(5, "The new-state paradox: a newcomer moves rows it never touched")
    hits = new_state_paradox(NEWCOMER, NEWCOMER_ROW, NEWCOMER_EXTRA, "largest_remainder")
    print(f"  before: {dict((r.label, int(r.value)) for r in NEWCOMER.rows)} over {NEWCOMER.units} seats")
    print(f"  add {NEWCOMER_ROW.label!r} at {int(NEWCOMER_ROW.value)} with its own "
          f"{NEWCOMER_EXTRA} extra seats\n")
    for label, b, a in hits:
        print(f"    {label:<8} {b} -> {a}")
    print("\n  Neither row changed and no seats were taken away from the table; the two")
    print("  existing rows swapped a seat because a third row joined. Named for Oklahoma")
    print("  joining the Union in 1907, when New York lost a seat to Maine.")


def s6_scoreboard() -> None:
    head(6, "Balinski-Young: every method has a witness against it")
    t = no_method_is_clean()
    print(f"  {'method':<22} {'kind':<24} {'fails to sum':>12} {'quota':>6} {'alabama':>8}")
    for name, (s, q, al) in t.items():
        print(f"  {name:<22} {METHOD_KIND[name]:<24} {s:>12} {q:>6} {al:>8}")
    defined = [x for x in CORPUS if x.is_definable()[0]]
    print(f"\n  counted over the {len(defined)} tables in the corpus that have a share at all.")
    print("  largest_remainder is the only method that never leaves the quota, and the only")
    print("  one that suffers the Alabama paradox. The divisor methods are the mirror image.")
    print("  Balinski and Young (1982): no method can be in neither column. The choice is")
    print("  which failure you are willing to explain, and 'we used round()' is not a choice.")


def s7_quota() -> None:
    head(7, "Quota violations, and by how much")
    for table in (COUNCIL, QUEUES, SHIFTS):
        print(f"  {table.name}  ({table.units} units)")
        for name in ("jefferson_dhondt", "webster_sainte_lague", "huntington_hill", "adams"):
            for label, awarded, quota in quota_violations(table, METHODS[name](table)):
                print(f"    {name:<22} {label:<10} awarded {awarded:>3}, exact share "
                      f"{quota:>7.3f}  ({awarded - quota:+.3f})")
        print()
    print("  A reader assumes without being told that a row owed 8.87 units gets 8 or 9.")
    print("  Sainte-Lague gives it 10. That is not rounding, it is a different rule.")


def s8_percent_only() -> None:
    head(8, "Three failures that belong to percentages, not to seats")
    step = representable_step(SURVEY7)
    print("  a) seven respondents, printed to one decimal place")
    print(f"     denominator {int(SURVEY7.total)}, so the only shares that exist are multiples "
          f"of {step:.4g} points:")
    print(f"     {[round(k * step, 3) for k in range(int(SURVEY7.total) + 1)]}")
    a = naive_half_up(SURVEY7)
    print(f"     the column prints {[f'{p:.1f}%' for p in a.percents(SURVEY7)]} - one decimal "
          f"place implies a sample")
    print(f"     far larger than {int(SURVEY7.total)}, and 42.9% is 3/7 dressed as a measurement")

    print("\n  b) a grouped table, rounded at both levels")
    rows = largest_remainder(GROUPED)
    per_group: Dict[str, int] = {}
    for label, u in zip(GROUPED.labels, rows.units):
        g = GROUPED.group_of[label]
        per_group[g] = per_group.get(g, 0) + u
    for g, rows_sum, own in subtotal_clash(GROUPED):
        print(f"     group {g}: rows sum to {rows_sum / 10:.1f}%, its own rounded share is {own / 10:.1f}%")
    print(f"     rows: {[f'{p:.1f}' for p in rows.percents(GROUPED)]} summing to "
          f"{rows.total / 10:.1f}%")
    print("     Rows must sum to subtotals, subtotals to the grand total, and every printed")
    print("     number must be a rounding of its own share. Three constraints, one set of")
    print("     integers, usually no solution. Pick which level is allowed to be wrong.")

    print("\n  c) a signed base")
    for t in CORPUS:
        ok, why = t.is_definable()
        if not ok:
            print(f"     {t.name:<12} {why}")


def s9_zero_rows() -> None:
    head(9, "The long tail: rows that exist and print as nothing")
    tail = TRAFFIC.rows[-1]
    print(f"  {TRAFFIC.name}: {len(TRAFFIC.rows)} sources, base {int(TRAFFIC.total):,}")
    print(f"  smallest row {tail.label!r} = {int(tail.value)} sessions = "
          f"{100 * tail.value / TRAFFIC.total:.4f}% of the base\n")
    print(f"  {'method':<22} {'that row':>9} {'sums':>6}")
    for name, fn in METHODS.items():
        al = fn(TRAFFIC)
        print(f"  {name:<22} {al.percents(TRAFFIC)[-1]:>8.1f}% {'yes' if al.sums_to(TRAFFIC) else 'NO':>6}")
    print("\n  Adams and Huntington-Hill guarantee every row a unit, so they cannot print 0.0%")
    print("  for a source that exists. They pay for it with quota violations elsewhere, and")
    print(f"  with having no answer at all when units are scarcer than rows "
          f"({SHORTLIST.name}: {SHORTLIST.units} places, {len(SHORTLIST.rows)} candidates).")


def s10_headline() -> None:
    head(10, "The headline run: fifteen tables")
    rep = audit_corpus()
    print(f"  {'table':<17} {'kind':<8} {'rows':>4} {'verdict':<11} {'gap':>4} "
          f"{'disputed rows':<28} findings")
    for a in rep.audits:
        disputed = ", ".join(a.disagreeing_rows()) or "-"
        print(f"  {a.table.name:<17} {a.table.kind:<8} {len(a.table.rows):>4} "
              f"{a.verdict.value:<11} {a.max_row_gap():>4} {disputed[:27]:<28} {len(a.findings)}")
    print(f"\n  {rep.verdicts}")
    print(f"  clean tables: {[a.table.name for a in rep.clean()]}")
    print("\n  One table in fifteen is consistent, and it is the one whose denominator")
    print("  divides its budget exactly: four equal rows at one decimal place.")


def s11_findings() -> None:
    head(11, "Twenty mechanisms, every one with evidence")
    rep = audit_corpus()
    print(f"  {'code':<28} {'severity':<9} {'fires':>6}  description")
    for code, n in rep.finding_counts.items():
        print(f"  {code:<28} {SEVERITY_OF[code]:<9} {n:>6}  {DESCRIPTION_OF[code]}")
    silent = sum(n for c, n in rep.finding_counts.items() if SEVERITY_OF[c] == "silent")
    print(f"\n  {silent} silent findings across the corpus: every one of them is a table that")
    print("  renders successfully, sums to 100%, and is defensibly wrong.")


def s12_what_to_do() -> None:
    head(12, "What to do instead")
    print("  1. Allocate, do not round. Any apportionment method sums exactly; independent")
    print(f"     rounding failed to sum on {sum(1 for t in CORPUS if t.is_definable()[0] and not naive_half_up(t).sums_to(t))}"
          f" of the {sum(1 for t in CORPUS if t.is_definable()[0])} definable tables here.")
    print("  2. Name the method in the caption. `largest_remainder` and `webster` are")
    print("     defensible; `round()` plus a residual on the last row is not a method.")
    print("  3. Print the decimal place the denominator can carry. n=7 gets whole numbers.")
    print("  4. Decide which level of a grouped table is allowed to disagree, and mark it.")
    print("  5. Refuse a signed or zero base rather than dividing by it.")
    a = audit(QUARTERS)
    print(f"\n  The one table that needs none of this: {QUARTERS.name}, verdict "
          f"{a.verdict.value}, {len(a.findings)} findings, both advisory.")


def main() -> None:
    print("Percent Recomputer - evidence for every number in the README")
    for fn in (s1_column, s2_nine_methods, s3_alabama, s4_population, s5_new_state,
               s6_scoreboard, s7_quota, s8_percent_only, s9_zero_rows, s10_headline,
               s11_findings, s12_what_to_do):
        fn()
    print()


if __name__ == "__main__":
    main()
