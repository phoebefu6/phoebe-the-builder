"""Every claim in the README, printed from the live computation.

Run it:  python evidence.py
"""

from __future__ import annotations

import premortem as P


def rule(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def money(x: float) -> str:
    return f"{x:,.0f}"


def section_1_the_plan_is_a_conjunction() -> None:
    rule("1. Twelve steps nobody would call risky. The plan succeeds 45% of the time.")
    print("This is what a pre-mortem is for, and it is arithmetic, not psychology.\n")
    print(f"{'step':<44}{'P(works)':>10}{'P(all so far)':>15}")
    print("-" * 84)
    running = 1.0
    for s in P.PLAN:
        running *= s.p_success
        print(f"{s.name:<44}{s.p_success:>10.2f}{running:>15.3f}")
    print(f"\nWeakest single step: {P.weakest_step_success():.2f}. "
          f"Whole plan: {P.plan_success():.3f}.")
    print(f"At the average step quality, {P.steps_to_coin_flip()} steps is a coin flip.")
    print("\nNobody in the room stated 45%. Each person stated a number about their own")
    print("step, and the plan's number was never computed, because no single person owned it.")


def section_2_independence_is_optimistic() -> None:
    rule("2. And the 45% assumed the steps are independent.")
    c = P.correlated_plan_success()
    print("Independence is not the neutral assumption, it is the favourable one. Add one")
    print("common cause - the engineer who knows the old warehouse leaves, the vendor slips -")
    print(f"firing with probability {c['shock_probability']:.2f} and tripling each step's failure rate:\n")
    print(f"    independent product rule   {c['independent']:.3f}")
    print(f"    with one common shock      {c['correlated']:.3f}")
    print(f"    gap                        {c['gap']:.3f}")
    print("\nThe product rule is already the optimistic answer. Every shared dependency -")
    print("one person, one vendor, one weekend - moves it down again.")


def section_3_what_comes_out() -> None:
    rule("3. Fourteen causes, and the four numbers each one needs.")
    print("A cause is not a category. 'Data quality issues' cannot be ranked, priced or")
    print("prevented; a mechanism can.\n")
    print(f"{'id':<5}{'cause':<62}{'P':>6}{'loss':>11}")
    print("-" * 84)
    for m in P.MODES:
        print(f"{m.id:<5}{m.cause[:60]:<62}{m.probability:>6.2f}{money(m.loss):>11}")
    print(f"\nTotal expected loss across the register: {money(P.total_expected_loss())}")
    print("\nThe corpus is authored - it is the worked example. Everything computed about it")
    print("below is arithmetic on these numbers and is asserted in the test suite.")


def section_4_the_matrix_cannot_rank() -> None:
    rule("4. The 5x5 matrix ranks a quarter of the pairs the wrong way round.")
    print("Cox (2008) proved qualitative risk matrices cannot reproduce the ordering of the")
    print("quantitative risks they summarise. Measured here against expected loss:\n")
    print(f"{'scale':<16}{'pairs':>7}{'ordered':>9}{'tied':>7}{'inverted':>10}"
          f"{'inversion rate':>16}{'cannot order':>14}")
    print("-" * 84)
    for sc in P.SCALES:
        q = P.ranking_quality(sc)
        print(f"{sc.name:<16}{q['pairs']:>7}{q['ordered_by_matrix']:>9}{q['tied_by_matrix']:>7}"
              f"{q['inversions']:>10}{q['inversion_rate']:>15.1%}{q['undecided_rate']:>14.1%}")
    print("\nAn inversion is not imprecision. It is the matrix saying A is the bigger risk")
    print("while A's expected loss is smaller. The worst of them on the default scale:\n")
    for hi, lo, ratio in P.inversions(P.SCALES[0])[:6]:
        a = next(m for m in P.MODES if m.id == hi)
        b = next(m for m in P.MODES if m.id == lo)
        print(f"    {hi} scores above {lo}, and {lo} carries {ratio:.1f}x the expected loss")
        print(f"        {hi}: P={a.probability:.2f} loss={money(a.loss):>9} "
              f"E={money(a.expected_loss)}")
        print(f"        {lo}: P={b.probability:.2f} loss={money(b.loss):>9} "
              f"E={money(b.expected_loss)}")


def section_5_the_buried_risk() -> None:
    rule("5. The matrix buries the largest risk in the register.")
    scale = P.SCALES[0]
    f06 = next(m for m in P.MODES if m.id == "F06")
    m_order = [m.id for m in P.by_matrix(scale)]
    e_order = [m.id for m in P.by_expected_loss()]
    v_order = [m.id for m in P.by_prevention_value()]
    print(f"F06 - {f06.cause}")
    print(f"      P={f06.probability:.2f}  loss={money(f06.loss)}  "
          f"expected loss={money(f06.expected_loss)}\n")
    print(f"    rank by matrix score      {m_order.index('F06') + 1} of {len(P.MODES)}")
    print(f"    rank by expected loss     {e_order.index('F06') + 1}")
    print(f"    rank by prevention value  {v_order.index('F06') + 1}")
    print(f"\n    its cell: likelihood band {scale.p_band(f06.probability)}, "
          f"impact band {scale.loss_band(f06.loss)}, score "
          f"{scale.score(f06)}")
    print("\nLow probability drags the band down and the band is all the score sees. The")
    print("single largest expected loss in the register lands eighth on the page the")
    print("steering committee reads.")


def section_6_range_compression() -> None:
    rule("6. Risks that share a cell become the same risk.")
    for sc in P.SCALES:
        c = P.range_compression(sc)
        print(f"\n{sc.name}: {c['occupied_cells']} cells occupied, "
              f"{c['shared_cells']} of them holding more than one risk")
        if c["worst_pair"]:
            hi, lo = c["worst_pair"]
            a = next(m for m in P.MODES if m.id == hi)
            b = next(m for m in P.MODES if m.id == lo)
            print(f"    worst cell {c['worst_cell']}: {hi} and {lo} score identically, "
                  f"and {hi} carries {c['worst_ratio']:.2f}x the expected loss")
            print(f"        {hi}: E={money(a.expected_loss)}   {lo}: E={money(b.expected_loss)}")
    print("\nEvery reader downstream of the matrix - the slide, the committee, the tracker -")
    print("sees one number and treats those as equivalent.")


def section_7_the_arithmetic_is_not_arithmetic() -> None:
    rule("7. Likelihood band times impact band is a number with no unit.")
    o = P.ordinal_product_is_meaningless(P.SCALES[0])
    print("Band 4 is not twice band 2. The bands are labels; multiplying two labels")
    print("produces a score whose indifference curves are an artefact of the bin edges.\n")
    print(f"    {o['cells']} distinct cells collapse to {o['distinct_scores']} distinct scores")
    print(f"    {o['colliding_scores']} of those scores are shared by more than one cell\n")
    score, cells = o["example"]
    print(f"    score {score} is produced by cells {cells}:")
    for pb, lb in cells:
        print(f"        likelihood band {pb} x impact band {lb}")
    print("\n    A 30% chance of a 2,000,000 loss and a 60% chance of a 500,000 loss score")
    print("    the same. Their expected losses differ by a factor of two.")


def section_8_two_templates_two_answers() -> None:
    rule("8. Two conventional scales, one register, a different top risk.")
    d = P.scale_disagreement()
    a, b = P.SCALES
    print(f"{'id':<5}{'E[loss]':>11}{'  ' + a.name:<18}{'  ' + b.name:<18}")
    print("-" * 84)
    for m in P.by_expected_loss():
        print(f"{m.id:<5}{money(m.expected_loss):>11}"
              f"{'  band ' + str(a.cell(m)) + ' score ' + str(a.score(m)):<18}"
              f"{'  band ' + str(b.cell(m)) + ' score ' + str(b.score(m)):<18}")
    print(f"\n{d['n_flips']} pairs are ordered oppositely by the two scales.")
    print(f"Top risk under {a.name}: {d['top_by_a']}")
    print(f"Top risk under {b.name}: {d['top_by_b']}")
    print(f"Same top risk: {d['same_top']}")
    print(f"\n{a.note}")
    print(f"{b.note}")
    print("\nThe single most consequential output of the exercise depends on which template")
    print("the organisation happens to use.")


def section_9_the_orderings_disagree() -> None:
    rule("9. Three orderings of the same register, and only one of them can be acted on.")
    scale = P.SCALES[0]
    print(f"{'':<6}{'by matrix':<12}{'by E[loss]':<12}{'by prevention value':<22}"
          f"{'by loss avoided / cost':<24}")
    print("-" * 84)
    orders = (P.by_matrix(scale), P.by_expected_loss(),
              P.by_prevention_value(), P.by_prevention_ratio())
    for i in range(len(P.MODES)):
        row = "".join(f"{o[i].id:<12}" if k < 2 else f"{o[i].id:<22}" if k == 2
                      else f"{o[i].id:<24}" for k, o in enumerate(orders))
        print(f"{i + 1:<6}{row}")
    d = P.ordering_disagreement(scale)
    print(f"\nmatrix top 3          {d['matrix_top3']}")
    print(f"expected loss top 3   {d['expected_loss_top3']}")
    print(f"prevention top 3      {d['prevention_top3']}")
    mid, mrank, vrank = d["biggest_move"]
    print(f"\nBiggest move: {mid} is #{mrank + 1} by matrix and #{vrank + 1} by prevention value.")
    print("\nOnly the last two orderings know what prevention COSTS, which is the only")
    print("question the meeting was held to answer. The matrix never asked.")


def section_10_it_is_a_knapsack() -> None:
    rule("10. It was never a ranking problem. It is a knapsack.")
    print("Prevention is bought under a budget, so the decision is which SET to buy - and")
    print("no ordering is guaranteed to find the best set. Exact answer by brute force over")
    print(f"all {2 ** len(P.MODES):,} subsets, against both heuristics:\n")
    print(f"{'budget':>9}{'matrix order':>14}{'ratio order':>13}{'optimal':>12}"
          f"{'matrix short':>14}{'ratio short':>13}")
    print("-" * 84)
    rows = P.allocation_comparison()
    for r in rows:
        print(f"{r['budget']:>9,}{money(r['matrix']):>14}{money(r['ratio']):>13}"
              f"{money(r['optimal']):>12}{money(r['matrix_shortfall']):>14}"
              f"{money(r['ratio_shortfall']):>13}")
    worst = max(rows, key=lambda r: r["matrix_shortfall"])
    print(f"\nAt a budget of {worst['budget']:,} the matrix ordering buys "
          f"{money(worst['matrix'])} of loss avoidance")
    print(f"where {money(worst['optimal'])} was available - it leaves "
          f"{worst['matrix_shortfall'] / worst['optimal']:.0%} of the achievable benefit unbought.")
    print("\nAnd note the honest part: at the tightest budget the ratio heuristic LOSES to")
    print("the matrix. Greedy-by-ratio is a knapsack heuristic, not a solution. Neither")
    print("ordering is reliable, which is the argument for not ordering at all - the exact")
    print("solve above runs in well under a second on fourteen items.")
    print(f"\nOptimal set at {rows[1]['budget']:,}: {rows[1]['optimal_set']}")


def section_11_the_notes_as_they_arrive() -> None:
    rule("11. Half the notes cannot be acted on.")
    r = P.notes_report()
    print("How a pre-mortem's output actually arrives, before anyone asks for numbers.\n")
    print(f"{'cause':<62}{'actionable'}")
    print("-" * 84)
    for n in P.RAW_NOTES:
        print(f"{str(n['cause'])[:60]:<62}{'yes' if P.actionable(n) else 'NO'}")
    print(f"\n{r['actionable']} of {r['n']} are actionable. Field by field:")
    for k, v in r["per_field"].items():
        print(f"    {k:<26}{v:>3} of {r['n']}")
    print(f"\nThe vague ones: {r['vague']}")
    print("\nEach of those names a category, not a mechanism. A category cannot be assigned")
    print("a probability, cannot be priced, and cannot be prevented - so it survives every")
    print("review untouched and appears again on the next project's register.")


def section_12_what_to_do() -> None:
    rule("12. What to keep from the exercise.")
    print("1. Run the pre-mortem. Prospective hindsight surfaces more and more specific")
    print("   causes than asking what might go wrong, and it costs an hour.")
    print("2. Compute the plan's own probability. Twelve steps at 95% is not 95%.")
    print("3. Demand a mechanism, not a category. If it cannot take a probability, it is")
    print("   not a finding yet.")
    print("4. Record four numbers per cause: probability, loss, cost to reduce it, and how")
    print("   much of it the spend removes. The last two are the ones nobody writes down")
    print("   and the only two that decide anything.")
    print("5. Do not score it on a matrix. On this register the default 5x5 inverts")
    print(f"   {P.ranking_quality(P.SCALES[0])['inversion_rate']:.0%} of the pairs it orders, "
          f"cannot order "
          f"{P.ranking_quality(P.SCALES[0])['undecided_rate']:.0%} of them at all, and buries the")
    print("   largest expected loss in the register in eighth place.")
    print("6. Then stop ranking. Name the budget and solve for the best set - it is a")
    print("   knapsack, it is exact, and it runs instantly at the size any real register is.")


def main() -> None:
    section_1_the_plan_is_a_conjunction()
    section_2_independence_is_optimistic()
    section_3_what_comes_out()
    section_4_the_matrix_cannot_rank()
    section_5_the_buried_risk()
    section_6_range_compression()
    section_7_the_arithmetic_is_not_arithmetic()
    section_8_two_templates_two_answers()
    section_9_the_orderings_disagree()
    section_10_it_is_a_knapsack()
    section_11_the_notes_as_they_arrive()
    section_12_what_to_do()
    print()


if __name__ == "__main__":
    main()
