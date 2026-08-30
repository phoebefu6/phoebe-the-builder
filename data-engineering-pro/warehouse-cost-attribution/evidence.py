"""Ten sections. Every number in the README is printed here and asserted in the tests."""

from __future__ import annotations

import math
import time
from typing import Dict


import costs as C


def _rule(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _row(name: str, alloc: Dict[str, float]) -> str:
    return f"{name:16} " + " ".join(f"{alloc[n]:>9,.0f}" for n in C.TEAM_NAMES)


def _head() -> str:
    return f"{'method':16} " + " ".join(f"{n[:9]:>9}" for n in C.TEAM_NAMES)


# ======================================================================================
def section_1() -> Dict:
    _rule("1. A month with a known answer")
    scan = storage = 0.0
    reads = C._reads_by_table(C.TEAM_NAMES)
    for table, r in reads.items():
        t = C.TABLES[table]
        cold = min(r, C.WORKDAYS * C.COLD_SLOTS_PER_DAY)
        scan += cold * t.scan_gb * C.SCAN_RATE + (r - cold) * t.scan_gb * C.SCAN_RATE * C.CACHE_RATE
        storage += t.storage_gb * C.STORAGE_RATE
    models = sum(C.WORKDAYS * m.build_gb * C.SCAN_RATE for m in C.MODELS)

    print(f"The invoice: ${C.INVOICE:,.2f}\n")
    print(f"{'component':22} {'$':>12} {'share':>8}   what makes it JOINT")
    for label, amount, why in [
        ("query scans", scan, "the 2nd read of a table that day costs 2% of the 1st"),
        ("storage", storage, "paid once, however many teams need the table"),
        ("upstream models", models, "built once, for everyone who consumes them"),
        ("reservation", C.RESERVED_FLOOR, "owed the moment ANYBODY uses the warehouse"),
    ]:
        print(f"{label:22} {amount:12,.2f} {amount/C.INVOICE:7.1%}   {why}")

    print(f"\nSix teams, {len(C.TABLES)} tables, {len(C.MODELS)} shared models, {C.WORKDAYS} working days.")
    print(f"{'team':20} {'reads/mo':>9} {'tables':>7}   what they do")
    for t in C.TEAMS:
        flag = "" if t.owned else "   <- nobody claims these"
        print(f"{t.name:20} {sum(t.reads.values()):9,} {len(t.reads):7}   {t.label}{flag}")

    print(f"\nThe cost function is not additive, and that is the whole problem. Summing what "
          f"each team would cost alone gives ${sum(C.raw_standalone().values()):,.0f} against an "
          f"invoice of ${C.INVOICE:,.0f}. The difference - ${sum(C.raw_standalone().values())-C.INVOICE:,.0f} "
          f"- is the value of sharing, and it does not belong to anybody in particular.")
    return {"scan": scan, "storage": storage, "models": models,
            "reservation": C.RESERVED_FLOOR, "invoice": C.INVOICE}


# ======================================================================================
def section_2() -> Dict:
    _rule("2. Seven defensible methods, one invoice")
    allocs = {name: fn() for name, fn in C.METHODS.items()}
    print("Each is a sentence somebody says out loud in a cost review. Each bills the full "
          f"${C.INVOICE:,.0f}.\n")
    print(_head())
    for name in C.METHODS:
        print(_row(name, allocs[name]))

    print(f"\n{'team':20} {'min':>10} {'max':>10} {'ratio':>8}   {'spread as % of invoice':>22}")
    spreads = {}
    for n in C.TEAM_NAMES:
        vals = [allocs[m][n] for m in C.METHODS]
        lo, hi = min(vals), max(vals)
        spreads[n] = (lo, hi, hi / max(lo, 1.0))
        print(f"{n:20} {lo:10,.0f} {hi:10,.0f} {hi/max(lo,1.0):7.1f}x {(hi-lo)/C.INVOICE:21.1%}")

    tops = {m: max(allocs[m], key=lambda k: allocs[m][k]) for m in C.METHODS}
    distinct = sorted(set(tops.values()))
    worst = max(spreads, key=lambda k: spreads[k][2])
    print(f"\n{worst} pays between ${spreads[worst][0]:,.0f} and ${spreads[worst][1]:,.0f} "
          f"depending on nothing but which sentence was said - a factor of "
          f"{spreads[worst][2]:.0f}.")
    print(f"And the single most consequential output of a cost review - WHICH TEAM TO GO TALK "
          f"TO - is {len(distinct)} different answers across the seven methods: "
          f"{', '.join(distinct)}.")
    return {"allocs": allocs, "spreads": spreads, "tops": tops, "distinct_tops": distinct}


# ======================================================================================
def section_3() -> Dict:
    _rule("3. Bill everyone what they actually cost you, and 90% of the invoice is unfunded")
    raw = C.raw_marginal()
    total = sum(raw.values())
    print("Marginal cost is the most defensible number in the building: what would stop being "
          "spent if this team stopped querying. Nobody can argue with it.\n")
    print(f"{'team':20} {'marginal $':>12} {'% of invoice':>13}   cost that vanishes if they leave")
    for n in C.TEAM_NAMES:
        print(f"{n:20} {raw[n]:12,.2f} {raw[n]/C.INVOICE:12.2%}")
    print(f"{'TOTAL':20} {total:12,.2f} {total/C.INVOICE:12.2%}")
    print(f"\nCharge every team exactly what it costs you and you collect ${total:,.0f} against "
          f"an invoice of ${C.INVOICE:,.0f}. **${C.INVOICE-total:,.0f} - {1-total/C.INVOICE:.1%} of "
          f"the bill - has no payer.**")
    print("This is not a modelling artefact. It is what 'joint' means: the storage is paid once, "
          "the model is built once, and the reservation exists whether or not any particular team "
          "shows up. No allocation that respects marginal cost can also add up to the invoice.")
    return {"raw_marginal": raw, "total": total, "recovery": total / C.INVOICE,
            "unfunded": C.INVOICE - total}


# ======================================================================================
def section_4() -> Dict:
    _rule("4. Bill everyone what they would have cost alone, and you collect three times over")
    raw = C.raw_standalone()
    total = sum(raw.values())
    print(f"{'team':20} {'standalone $':>13} {'x its shapley':>14}   would-be invoice on its own")
    sh = C.shapley()
    for n in C.TEAM_NAMES:
        print(f"{n:20} {raw[n]:13,.0f} {raw[n]/sh[n]:13.1f}x")
    print(f"{'TOTAL':20} {total:13,.0f} {'':14}   = {total/C.INVOICE:.1%} of the actual invoice")
    print("\nEvery one of those numbers is true. Every team really would face that bill alone, "
          "because it would pay the reservation, the storage and the cold scans by itself.")
    print(f"The gap - ${total - C.INVOICE:,.0f} - is the value the teams create by being in the "
          f"same warehouse. Marginal cost hands all of it back to nobody; standalone cost charges "
          f"for it {total/C.INVOICE:.1f} times. The truth is between them and the method decides where.")
    return {"raw_standalone": raw, "total": total, "over_recovery": total / C.INVOICE,
            "sharing_value": total - C.INVOICE}


# ======================================================================================
def section_5() -> Dict:
    _rule("5. Shapley: the average over every order teams could have arrived in")
    n = len(C.TEAM_NAMES)
    t0 = time.perf_counter()
    sh = C.shapley()
    elapsed = time.perf_counter() - t0
    print(f"{math.factorial(n):,} orderings, {2**n} coalitions, computed exactly in {elapsed:.2f}s.\n")
    print(f"{'team':20} {'shapley $':>11} {'% of invoice':>13} {'vs marginal':>12} {'vs standalone':>14}")
    raw_m, raw_s = C.raw_marginal(), C.raw_standalone()
    for t in C.TEAM_NAMES:
        print(f"{t:20} {sh[t]:11,.0f} {sh[t]/C.INVOICE:12.1%} {sh[t]/max(raw_m[t],1):11.1f}x "
              f"{sh[t]/raw_s[t]:13.2f}x")
    print(f"{'TOTAL':20} {sum(sh.values()):11,.0f} {sum(sh.values())/C.INVOICE:12.1%}")
    print("\nIt is the unique rule satisfying four axioms at once: it adds up to the invoice, it "
          "gives equal shares to teams that contribute identically, it charges nothing to a team "
          "that adds nothing, and it is additive across separable bills.")
    print("Uniqueness is a strong claim about the METHOD. Section 6 is about how much weaker a "
          "claim it is about whether anybody would accept the bill.")
    print(f"Cost of exactness: 2^n coalitions. At 6 teams that is {2**n}. At 30 teams it is "
          f"{2**30:,}, which is why section 9 asks whether the exact version is worth building.")
    return {"shapley": sh, "seconds": elapsed, "orderings": math.factorial(n)}


# ======================================================================================
def section_6() -> Dict:
    _rule("6. 'Fair' is not a point. It is a polytope, and it is nearly as wide as the bill.")
    nonempty = C.core_is_nonempty()
    print("The CORE is the set of allocations no group of teams would walk out of: no coalition "
          "is asked to pay more than it would cost that coalition alone.\n")
    print(f"Is the core non-empty? {nonempty}\n")
    print(f"{'team':20} {'core min':>10} {'core max':>10} {'width':>10} {'width/invoice':>14} {'shapley':>10}")
    sh = C.shapley()
    ranges = {}
    for t in C.TEAM_NAMES:
        lo, hi = C.core_range(t)
        ranges[t] = (lo, hi)
        print(f"{t:20} {lo:10,.0f} {hi:10,.0f} {hi-lo:10,.0f} {(hi-lo)/C.INVOICE:13.1%} {sh[t]:10,.0f}")

    widest = max(ranges, key=lambda k: ranges[k][1] - ranges[k][0])
    lo, hi = ranges[widest]
    print(f"\nEvery allocation inside those bounds is defensible on the strongest fairness test "
          f"cooperative game theory offers. {widest} could be billed ${lo:,.0f} or ${hi:,.0f} and "
          f"nobody has grounds to object either way - a spread of {(hi-lo)/C.INVOICE:.0%} of the "
          f"entire invoice.")

    print("\nWhich of the seven methods survive the test?\n")
    print(f"{'method':16} {'in core':>8} {'worst objection':>17}   objecting coalition")
    survivors, rejects = [], []
    for name, fn in C.METHODS.items():
        a = fn()
        viol = C.core_violations(a)
        ok = not viol
        (survivors if ok else rejects).append(name)
        if ok:
            print(f"{name:16} {'yes':>8} {'-':>17}")
        else:
            S, ex = viol[0]
            print(f"{name:16} {'NO':>8} {ex:17,.0f}   {'+'.join(s[:8] for s in S)}")
    print(f"\nThe core rejects {len(rejects)} of {len(C.METHODS)} methods ({', '.join(rejects)}) and "
          f"accepts {len(survivors)}. Those {len(survivors)} survivors disagree with each other by up "
          f"to {max((max(C.METHODS[m]()[n] for m in survivors)) / max(min(C.METHODS[m]()[n] for m in survivors), 1) for n in C.TEAM_NAMES):.0f}x "
          f"on a single team.")
    print("So the strongest fairness test in the theory is a real filter and a weak one: it rules "
          "out the two rules people actually reach for first, and leaves everything else.")
    return {"nonempty": nonempty, "ranges": ranges, "survivors": survivors, "rejects": rejects}


# ======================================================================================
def section_7() -> Dict:
    _rule("7. The part of the bill that belongs to nobody")
    sh = C.shapley()
    unowned = C.unowned_cost()
    res = C.RESERVED_FLOOR
    print(f"{'component':26} {'$':>10} {'% of invoice':>13}")
    print(f"{'reservation floor':26} {res:10,.0f} {res/C.INVOICE:12.1%}")
    print(f"{'orphaned jobs (saving if':26} {unowned:10,.0f} {unowned/C.INVOICE:12.1%}")
    print(f"{'   they were switched off)':26}")
    print(f"\nThe reservation is owed the moment anybody uses the warehouse and does not go away "
          f"when any single team leaves. Its marginal cost to every team is zero and its total is "
          f"{res/C.INVOICE:.0%} of the invoice. There is no non-arbitrary way to divide it.")

    print("\nAnd then the sharpest number in this build:")
    print(f"  Shapley says the orphaned scheduled jobs are responsible for   ${sh['scheduled_unowned']:,.0f}"
          f"  ({sh['scheduled_unowned']/C.INVOICE:.0%} of the invoice)")
    print(f"  Switching those same jobs off would save                       ${unowned:,.0f}"
          f"  ({unowned/C.INVOICE:.1%} of the invoice)")
    print(f"  Ratio                                                          "
          f"{sh['scheduled_unowned']/max(unowned,1):.0f}x")
    print(f"\nBoth are correct. The jobs really do consume {sh['scheduled_unowned']/C.INVOICE:.0%} of the "
          f"warehouse by any consumption-based measure, and turning them off really would save "
          f"almost nothing, because everything they read is read by somebody else anyway.")
    print("A cost review that acts on the first number cancels the jobs and does not see the "
          "saving. One that acts on the second never cancels anything. Both readings come off "
          "the same invoice.")
    return {"reservation": res, "unowned_saving": unowned,
            "unowned_shapley": sh["scheduled_unowned"],
            "ratio": sh["scheduled_unowned"] / max(unowned, 1)}


# ======================================================================================
def section_8() -> Dict:
    _rule("8. The same query, priced by who happened to run first")
    ratio = C.cache_ratio("events_raw")
    t = C.TABLES["events_raw"]
    cold = t.scan_gb * C.SCAN_RATE
    warm = cold * C.CACHE_RATE
    print(f"Under direct-bytes billing, one scan of {t.name} ({t.scan_gb:,.0f} GB):\n")
    print(f"  first read of the day   ${cold:8,.2f}")
    print(f"  every read after that   ${warm:8,.2f}")
    print(f"  ratio                   {ratio:8,.0f}x   for byte-identical work")

    print("\nSo the cost of a query is not a property of the query. It is a property of the "
          "queue. The team that runs at 06:00 subsidises everyone who runs at 09:00, and the "
          "chargeback report reads that subsidy as profligacy.")
    fake = C.method_first_toucher()
    print(f"\n`first_toucher` prices that honestly and the result is absurd: analytics is billed "
          f"${fake['analytics']:,.0f} ({fake['analytics']/C.INVOICE:.0%} of the invoice) because it "
          f"sorts first alphabetically - which is exactly as principled as sorting by who happens "
          f"to have the earliest cron.")
    print(f"\nThe incentive this creates is the real cost. Any billing rule that charges the cold "
          f"scan to whoever triggers it pays every team {ratio:,.0f}x to wait for somebody else to "
          f"go first. If everybody waits, the dashboards are late and the bill is unchanged.")
    return {"cold": cold, "warm": warm, "ratio": ratio,
            "first_toucher_analytics": fake["analytics"]}


# ======================================================================================
def section_9() -> Dict:
    _rule("9. Does the sophistication pay?")
    exact = C.shapley()
    print("Shapley is the principled answer and it costs 2^n coalition evaluations. Two "
          "questions worth asking before building it: does the sampled version get you there, "
          "and does the exact one buy anything the cheap rules do not?\n")
    print(f"{'permutations':>13} {'max abs error':>14} {'% of invoice':>13} {'seconds':>9}")
    conv = []
    for m in [50, 200, 1_000, 5_000]:
        t0 = time.perf_counter()
        s = C.sampled_shapley(m, seed=1)
        el = time.perf_counter() - t0
        err = max(abs(s[n] - exact[n]) for n in C.TEAM_NAMES)
        conv.append({"draws": m, "err": err, "seconds": el})
        print(f"{m:13,} {err:14,.0f} {err/C.INVOICE:12.2%} {el:9.2f}")

    good = next(r for r in conv if r["err"] / C.INVOICE < 0.01)
    print(f"\nNEGATIVE RESULT: {good['draws']:,} sampled orderings land within "
          f"{good['err']/C.INVOICE:.2%} of the exact value. The 2^n objection to Shapley is real "
          f"in complexity and mostly irrelevant in practice - do NOT build the exact version for a "
          f"warehouse with thirty teams, sample it.")

    print("\nAnd the cheap rules against Shapley, as a % of the invoice per team:\n")
    print(f"{'method':16} {'max deviation from shapley':>28} {'in core':>9}")
    rows = []
    for name, fn in C.METHODS.items():
        if name == "shapley":
            continue
        a = fn()
        dev = max(abs(a[n] - exact[n]) for n in C.TEAM_NAMES) / C.INVOICE
        rows.append((name, dev, C.in_core(a)))
        print(f"{name:16} {dev:27.1%} {str(C.in_core(a)):>9}")
    best = min(rows, key=lambda r: r[1])
    smallest = min(exact, key=lambda k: exact[k])
    err_dollars = best[1] * C.INVOICE
    print(f"\nThe closest cheap rule is {best[0]}, still {best[1]:.1%} of the invoice away from "
          f"Shapley on some team - ${err_dollars:,.0f} on a ${C.INVOICE:,.0f} bill. That error alone "
          f"is {err_dollars/exact[smallest]:.0%} of {smallest}'s entire Shapley share "
          f"(${exact[smallest]:,.0f}). No cheap rule here approximates the principled one, so the "
          f"choice is between sampling Shapley and admitting you picked a different rule.")
    return {"convergence": conv, "cheap": rows, "closest": best, "good": good}


# ======================================================================================
def section_10(s2, s3, s4, s6, s7, s9) -> Dict:
    _rule("10. What a cost allocation has to carry")
    print(f"1. THE METHOD, NAMED. Seven defensible rules put one team anywhere between "
          f"${min(v[0] for v in s2['spreads'].values()):,.0f} and "
          f"${max(v[1] for v in s2['spreads'].values()):,.0f}, and name "
          f"{len(s2['distinct_tops'])} different teams as the most expensive.")
    print(f"2. THE ADMISSION THAT IT CANNOT ADD UP. Marginal cost recovers "
          f"{s3['recovery']:.1%} of the invoice; standalone cost recovers "
          f"{s4['over_recovery']:.0%}. No rule is both defensible per team and exact in total.")
    print(f"3. A RANGE, NOT A NUMBER. The core is non-empty and up to "
          f"{max((hi-lo) for lo, hi in s6['ranges'].values())/C.INVOICE:.0%} of the invoice wide "
          f"per team. Every point in it is fair.")
    print(f"4. WHAT IT REFUSES TO ATTRIBUTE. The reservation is "
          f"{s7['reservation']/C.INVOICE:.0%} of the bill with a marginal cost of zero to "
          f"everyone. Forcing it onto teams is a political act, not an accounting one.")
    print(f"5. THE GAP BETWEEN BLAME AND SAVING. The orphaned jobs are charged "
          f"${s7['unowned_shapley']:,.0f} and switching them off saves ${s7['unowned_saving']:,.0f} "
          f"- {s7['ratio']:.0f}x apart. Attribution answers 'who consumed it'. It does not answer "
          f"'what would we save'. Budget decisions need the second.")
    print("6. THE INCENTIVE IT CREATES. Charging the cold scan to whoever triggers it pays every "
          "team to run late.")
    print(f"7. ITS OWN COST. {s9['good']['draws']:,} sampled orderings get within "
          f"{s9['good']['err']/C.INVOICE:.2%} of exact Shapley; the exact 2^n version is not worth "
          f"building above a handful of teams.")
    return {}


def main() -> Dict:
    s1 = section_1()
    s2 = section_2()
    s3 = section_3()
    s4 = section_4()
    s5 = section_5()
    s6 = section_6()
    s7 = section_7()
    s8 = section_8()
    s9 = section_9()
    s10 = section_10(s2, s3, s4, s6, s7, s9)
    return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5,
            "s6": s6, "s7": s7, "s8": s8, "s9": s9, "s10": s10}


if __name__ == "__main__":
    main()
