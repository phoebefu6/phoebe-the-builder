"""Every number in the README, printed from the model that produced it.

Run:  python3 evidence.py
"""

from __future__ import annotations

import codelay as C

RULE = "=" * 78


def h(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def fmt_order(o) -> str:
    return " -> ".join(o)


def main() -> None:
    items = C.backlog()
    lin = C.linearised(items)

    # ---------------------------------------------------------------- section 1
    h(1, "THE BACKLOG")
    print("Nine items, 40 weeks of work for one team, cost of delay in $k/week.\n")
    print(f"{'':3} {'item':20} {'dur':>4} {'p-wks':>6} {'shape':>9} "
          f"{'rate@0':>7} {'mean':>7} {'total@40':>9} {'RICE':>7}")
    for k in sorted(items):
        it = items[k]
        print(f"{k:3} {it.name:20} {it.duration:4.0f} {it.person_weeks:6.0f} "
              f"{it.cod.kind:>9} {it.cod.rate(0.0):7.1f} {it.cod.mean_rate():7.1f} "
              f"{it.cod.cum(C.HORIZON):9.1f} {it.rice:7.0f}")
    print(f"\nTotal duration: {sum(i.duration for i in items.values()):.0f} weeks. "
          f"Precedence: {', '.join(a + '->' + b for a, b in C.PRECEDENCE)}.")

    # ---------------------------------------------------------------- section 2
    h(2, "COST OF DELAY IS NOT A SCALAR")
    print("Four items, each quoted at the rate the room would give on day one.")
    print("What each one actually costs depends on the week it lands.\n")
    probes = ["D", "A", "H", "B"]
    print(f"{'':3} {'item':20} {'quoted@0':>9} " +
          "".join(f"{'wk' + str(w):>9}" for w in (5, 10, 20, 30, 40)))
    for k in probes:
        it = items[k]
        row = "".join(f"{it.cod.cum(w):9.1f}" for w in (5, 10, 20, 30, 40))
        print(f"{k:3} {it.name:20} {it.cod.rate(0.0):9.1f} {row}")
    d, b_ = items["D"], items["B"]
    print(f"\n'{d.name}' is quoted at {d.cod.rate(0.0):.0f}/week and "
          f"'{b_.name}' at {b_.cod.rate(0.0):.0f}/week.")
    print(f"  At week 10 the {d.cod.rate(0.0):.0f}/week item has cost "
          f"{d.cod.cum(10):.1f} and the 0/week item has cost {b_.cod.cum(10):.1f}.")
    print(f"  At week 40 the {d.cod.rate(0.0):.0f}/week item has cost "
          f"{d.cod.cum(40):.1f} and the 0/week item has cost {b_.cod.cum(40):.1f}.")
    late = d.cod.cum(40) - d.cod.cum(30)
    early = d.cod.cum(10) - d.cod.cum(0)
    print(f"\nSaturation: delaying '{d.name}' from week 0 to 10 costs {early:.1f}. "
          f"Delaying it from 30 to 40 costs {late:.1f} - a factor of "
          f"{early / late:.0f}.")
    print("Urgency is not a property of the item. It is a property of when you are.")

    # ---------------------------------------------------------------- section 3
    h(3, "WSJF IS OPTIMAL, AND IT NEEDS FOUR CONDITIONS")
    print("Smith's rule (1956): on one machine, with linear delay costs, no")
    print("deadlines and no precedence, sorting by weight/duration descending")
    print("minimises total weighted completion time. That is exactly CD3.\n")
    sl = C.sweep(lin)
    cd3_lin = C.order_cd3_mean(lin)
    print(f"Linearised backlog, all {sl['count']:,} orderings enumerated:")
    print(f"  exhaustive optimum   {sl['best']:9.4f}   {fmt_order(sl['best_order'])}")
    print(f"  CD3 (weight/dur)     {C.cost_of(cd3_lin, lin):9.4f}   {fmt_order(cd3_lin)}")
    print(f"  identical order?     {cd3_lin == sl['best_order']}")
    print(f"  gap                  {C.cost_of(cd3_lin, lin) - sl['best']:9.4f}")
    print("\nThe theorem holds exactly. Now the same backlog with its real cost")
    print("shapes put back - nothing else changed:\n")
    sr = C.sweep(items)
    cd3_real = C.order_cd3_mean(items)
    gap = C.cost_of(cd3_real, items) - sr["best"]
    print(f"  exhaustive optimum   {sr['best']:9.1f}   {fmt_order(sr['best_order'])}")
    print(f"  CD3 (mean rate/dur)  {C.cost_of(cd3_real, items):9.1f}   {fmt_order(cd3_real)}")
    print(f"  gap                  {gap:9.1f}  (+{100 * gap / sr['best']:.1f}%)")
    print(f"\nThe four conditions this backlog violates: non-linear cost shapes "
          f"({sum(1 for i in items.values() if i.cod.kind != 'linear')} of "
          f"{len(items)} items), a fixed date, {len(C.PRECEDENCE)} precedence "
          f"edges, and more than one team.")
    print("Each is measured separately in sections 6, 7 and 8.")

    # ---------------------------------------------------------------- section 4
    h(4, "'CD3' DOES NOT NAME AN ORDERING")
    print("The method is cost-of-delay divided by duration. 'Cost of delay' is")
    print("one number extracted from a room. Three defensible extractions:\n")
    variants = [("cd3_initial", "what a week costs us right now"),
                ("cd3_mean", "averaged over the planning window"),
                ("cd3_peak", "the worst week in the window")]
    costs = C.all_costs(items)
    for name, gloss in variants:
        o = C.ORDERINGS[name](items)
        v = C.cost_of(o, items)
        print(f"  {name:12} {gloss:36} {v:8.1f}  {''.join(o)}")
    a = C.order_cd3_initial(items)
    b = C.order_cd3_mean(items)
    p = C.order_cd3_peak(items)
    npairs = len(items) * (len(items) - 1) // 2
    print(f"\nPairs of items the variants disagree about (of {npairs}):")
    print(f"  initial vs mean  {C.kendall_distance(a, b):2}    "
          f"mean vs peak  {C.kendall_distance(b, p):2}    "
          f"initial vs peak  {C.kendall_distance(a, p):2}")
    print(f"\nCost spread across the three: "
          f"{C.cost_of(a, items) - C.cost_of(b, items):.1f} "
          f"({100 * (C.cost_of(a, items) / C.cost_of(b, items) - 1):.1f}%).")
    print("Naming the method does not determine the answer. The elicitation does.")

    # ---------------------------------------------------------------- section 5
    h(5, "FOUR OF NINE ORDERINGS LOSE TO DRAWING THE BACKLOG OUT OF A HAT")
    print(f"All {len(costs):,} orderings, exact - not sampled. A method's")
    print("percentile is the share of orderings that are cheaper than it.\n")
    print(f"{'method':16} {'cost':>8} {'vs optimum':>11} {'percentile':>11}")
    print(f"{'exhaustive optimum':16} {sr['best']:8.1f} {'':>11} {0.0:10.1f}%")
    rows = sorted(((n, C.cost_of(f(items), items)) for n, f in C.ORDERINGS.items()),
                  key=lambda r: r[1])
    shown = set()
    for n, v in rows:
        pct = 100 * C.percentile_of(costs, v)
        print(f"{n:16} {v:8.1f} {100 * (v / sr['best'] - 1):10.1f}% {pct:10.1f}%")
        shown.add(n)
    print(f"{'random (mean)':16} {sr['mean']:8.1f} "
          f"{100 * (sr['mean'] / sr['best'] - 1):10.1f}% {'':>11}")
    print(f"{'worst possible':16} {sr['worst']:8.1f} "
          f"{100 * (sr['worst'] / sr['best'] - 1):10.1f}% {100.0:10.1f}%")
    losers = [(n, v) for n, v in rows if v > sr["mean"]]
    print(f"\nBeaten by the average of all orderings: "
          f"{', '.join(n for n, _ in losers)}.")
    ci = C.cost_of(C.order_cd3_initial(items), items)
    cm = C.cost_of(C.order_cd3_mean(items), items)
    print(f"The same method spans the {100 * C.percentile_of(costs, ci):.0f}th "
          f"percentile (initial rate) and the "
          f"{100 * C.percentile_of(costs, cm):.1f}th (mean rate).")
    r1, r2 = C.order_rice(items), C.order_rice_duration(items)
    v1, v2 = C.cost_of(r1, items), C.cost_of(r2, items)
    print("\nThe popular critique of RICE is its denominator: effort is in "
          "person-months\nand delay is paid in calendar weeks. Swapping the "
          "denominator for duration moves")
    print(f"{C.kendall_distance(r1, r2)} of {npairs} pairs and costs "
          f"{v2 - v1:+.1f} - it makes things very slightly worse.")
    print("RICE's problem is not its denominator. Reach x Impact x Confidence is a")
    print("value estimate, and value is not cost of delay: it says how much the "
          "thing\nis worth, not what each week of not having it costs.")

    # ---------------------------------------------------------------- section 6
    h(6, "NOBODY SCHEDULES TO THE DATE")
    it = items["B"]
    print(f"'{it.name}' has a fixed date at week {it.cod.t_break:.0f} and costs")
    print(f"{it.cod.r2:.0f}/week after it. Before it, delay is free. Where each")
    print("method lands it:\n")
    print(f"{'method':16} {'position':>8} {'finish wk':>10} {'vs date':>8} {'penalty':>9}")
    for n, f in C.ORDERINGS.items():
        o = f(items)
        fin = C.completions(o, items)["B"]
        print(f"{n:16} {o.index('B') + 1:8} {fin:10.0f} "
              f"{fin - it.cod.t_break:+8.0f} {it.cod.cum(fin):9.1f}")
    fin_opt = C.completions(sr["best_order"], items)["B"]
    print(f"{'optimum':16} {sr['best_order'].index('B') + 1:8} {fin_opt:10.0f} "
          f"{fin_opt - it.cod.t_break:+8.0f} {it.cod.cum(fin_opt):9.1f}")
    print(f"\nThe optimum finishes it at week {fin_opt:.0f} with "
          f"{it.cod.t_break - fin_opt:.0f} week of slack and pays nothing.")
    print("Two methods finish it at week 40 and pay "
          f"{it.cod.cum(40):.0f}. Three finish it at week 4, "
          f"{it.cod.t_break - 4:.0f} weeks early, and pay nothing either -")
    print("but 4 weeks of the queue went in front of everything that was bleeding.")
    opt = sr["best_order"]
    d_first = ["D"] + [k for k in opt if k != "D"]
    print(f"\nHow tight the slack is: the item with the highest cost of delay "
          f"*right now*\n('{items['D'].name}', {items['D'].cod.rate(0):.0f}/week) "
          f"is last in the optimum. Moving it to the front")
    print(f"costs {C.cost_of(d_first, items) - sr['best']:.1f} - because its "
          f"{items['D'].duration:.0f} weeks push the fixed date past week "
          f"{it.cod.t_break:.0f}.")

    # ---------------------------------------------------------------- section 7
    h(7, "WHERE THE METHOD DOES NOT BREAK: TWO TEAMS")
    print("Two teams on the linearised backlog. Within one team WSPT is provably")
    print("optimal, so the exact two-team optimum is a search over assignments.\n")
    o2 = C.optimal_two_team_assignment(lin)
    greedy = C.parallel_cost(C.order_cd3_mean(lin), lin, 2)
    one = sl["best"]
    print(f"  one team, optimum            {one:9.1f}")
    print(f"  two teams, exact optimum     {o2['best']:9.1f}   "
          f"{'/'.join(''.join(g) for g in o2['split'])}")
    print(f"  two teams, CD3 list-schedule {greedy:9.1f}")
    print(f"  list-scheduling gap          {greedy - o2['best']:9.1f}  "
          f"({100 * (greedy / o2['best'] - 1):.2f}%)")
    print("\nA negative result worth having: on this backlog the parallel-capacity")
    print(f"break costs {100 * (greedy / o2['best'] - 1):.2f}%. Walking the CD3 "
          f"order and handing each item to")
    print("whichever team is free is very nearly optimal, and simpler than the")
    print("assignment search. Do not fix this one.")
    print(f"\nWhat does change: doubling the teams cuts delay cost by "
          f"{100 * (1 - o2['best'] / one):.1f}%, not 50%.")
    print("Delay cost is not linear in capacity, so 'add a team' is not a lever")
    print("with a predictable price.")

    # ---------------------------------------------------------------- section 8
    h(8, "PRECEDENCE: THE REPAIR IS NOT FREE, AND IT IS NOT THE PROBLEM")
    sp = C.sweep(items, edges=C.PRECEDENCE)
    print(f"Two edges ({', '.join(a + '->' + b for a, b in C.PRECEDENCE)}) rule "
          f"out {len(costs) - sp['count']:,} of {len(costs):,} orderings")
    print(f"({100 * (1 - sp['count'] / len(costs)):.0f}%). Feasible orderings: "
          f"{sp['count']:,}.\n")
    print(f"  unconstrained optimum        {sr['best']:9.1f}   "
          f"{fmt_order(sr['best_order'])}")
    print(f"  precedence-feasible optimum  {sp['best']:9.1f}   "
          f"{fmt_order(sp['best_order'])}")
    print(f"  cost of the constraint       {sp['best'] - sr['best']:9.1f}  "
          f"({100 * (sp['best'] / sr['best'] - 1):.1f}%)")
    raw = C.order_cd3_mean(items)
    ok = all(raw.index(x) < raw.index(y) for x, y in C.PRECEDENCE)
    rep = C.repair_precedence(raw, C.PRECEDENCE)
    print(f"\n  CD3 order                    {C.cost_of(raw, items):9.1f}   "
          f"{''.join(raw)}   feasible: {ok}")
    print(f"  after the usual repair       {C.cost_of(rep, items):9.1f}   "
          f"{''.join(rep)}")
    print(f"  repair cost                  {C.cost_of(rep, items) - C.cost_of(raw, items):9.1f}")
    print(f"  remaining gap to feasible optimum "
          f"{C.cost_of(rep, items) - sp['best']:.1f} "
          f"(+{100 * (C.cost_of(rep, items) / sp['best'] - 1):.1f}%)")
    print("\nThe constraint itself is cheap. Ranking as if it were absent and then")
    print("pushing blocked items down the list is what costs money.")

    # ---------------------------------------------------------------- section 9
    h(9, "THE RANK IS NOT REPRODUCIBLE. THE COST IS.")
    print("Durations are estimates. Rank on the estimate, pay on the truth.")
    print("2,000 trials per row, lognormal multiplicative error, seeded.\n")
    print(f"{'sigma':>6} {'reorder rate':>13} {'mean cost':>10} {'p90':>9} "
          f"{'worst':>9} {'added vs truth':>15}")
    for sg in (0.20, 0.35, 0.50, 0.70):
        r = C.noise_sweep(items, sg, 2000)
        print(f"{sg:6.2f} {100 * r['reorder_rate']:12.1f}% {r['mean']:10.1f} "
              f"{r['p90']:9.1f} {r['max']:9.1f} {r['mean'] - r['truth_cost']:15.1f}")
    r35 = C.noise_sweep(items, 0.35, 2000)
    print(f"\nAt sigma=0.35 - an ordinary software estimate - the CD3 ranking "
          f"changes in\n{100 * r35['reorder_rate']:.1f}% of trials, so the "
          f"*order* is essentially never reproducible.")
    print(f"The cost it delivers moves by {r35['mean'] - r35['truth_cost']:.1f} "
          f"on average.")
    print(f"For comparison, the CD3-to-optimum gap is {gap:.1f} and the "
          f"RICE-to-CD3 gap is\n"
          f"{C.cost_of(C.order_rice(items), items) - cm:.1f}.")
    print("\nSo: arguing about whether item 4 or item 5 goes first is inside the")
    print("noise. Arguing about which method to use is not. The reproducible")
    print("output of a prioritisation exercise is its cost, never its rank.")

    # --------------------------------------------------------------- section 10
    h(10, "WHAT SURVIVES")
    for i, line in enumerate((
        "An ordering is not a schedule. Score the schedule.",
        "Cost of delay is a rate over time. A scalar throws away the shape, and "
        "the shape is where the deadlines and the saturating windows live.",
        f"CD3 is exactly optimal under Smith's rule - reproduced here to a gap of "
        f"{abs(C.cost_of(cd3_lin, lin) - sl['best']):.4f} against a full "
        f"enumeration - and every one of its four conditions is violated by an "
        f"ordinary backlog.",
        f"'CD3' does not name an ordering: three defensible elicitations of the "
        f"same input disagree about {C.kendall_distance(a, b)} of {npairs} pairs "
        f"and span the {100 * C.percentile_of(costs, ci):.0f}th to the "
        f"{100 * C.percentile_of(costs, cm):.1f}th percentile.",
        f"Four of the nine orderings are beaten by the mean of all "
        f"{len(costs):,} orderings; RICE by "
        f"{100 * (C.cost_of(C.order_rice(items), items) / sr['mean'] - 1):.0f}%.",
        f"The optimum holds the fixed date with "
        f"{items['B'].cod.t_break - fin_opt:.0f} week of slack. No method finds "
        f"that; they miss by 14 weeks or hit it 22 weeks early.",
        "Parallel capacity is the condition you can safely ignore, and it is the "
        "one people reach for first.",
        "The rank is noise. The cost is the number to report.",
    ), 1):
        print(f"  {i}. {line}")
    print()


if __name__ == "__main__":
    main()
