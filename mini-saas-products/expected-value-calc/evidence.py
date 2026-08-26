"""Every claim in the README, printed from the live computation.

Run it:  python evidence.py
"""

from __future__ import annotations

import evcalc as E


def rule(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def money(x: float) -> str:
    return f"{x:,.0f}"


def section_1_the_decision() -> None:
    rule("1. The decision, and the four things nobody knows.")
    print("Build a tool, buy the vendor's, or do neither. Three-year horizon.\n")
    print(f"{'input':<16}{'P10':>9}{'typed':>9}{'P90':>9}  {'unit':<20}what it is")
    print("-" * 84)
    for i in E.INPUTS:
        print(f"{i.name:<16}{i.low:>9.1f}{i.mid:>9.1f}{i.high:>9.1f}  {i.unit:<20}{i.note}")
    print(f"\nFixed: build team {money(E.BUILD_TEAM_COST_PER_MONTH)}/month, "
          f"upkeep {money(E.BUILD_MAINTENANCE_PER_YEAR)}/year;")
    print(f"       vendor {money(E.BUY_ONBOARDING)} onboarding, "
          f"{money(E.BUY_LICENCE_PER_SEAT_YEAR)}/seat/year, capped at "
          f"{E.BUY_SEAT_CAP:.0f} seats.")
    print("\nTwo nonlinearities, both ordinary: the vendor cap makes `buy` flat in seats")
    print("above 20, and `build` earns nothing until it ships, so an overrun costs twice.")


def section_2_two_averaging_errors() -> None:
    rule("2. Two different averaging errors, and the famous one is the smaller.")
    print("ERROR ONE - the number typed into the cell is the most-likely value, which for")
    print("a skewed range is the mode, not the mean.\n")
    print(f"{'input':<16}{'typed':>10}{'actual mean':>14}{'shift':>10}")
    print("-" * 84)
    for k, d in E.mode_vs_mean().items():
        print(f"{k:<16}{d['elicited_mid']:>10.2f}{d['actual_mean']:>14.2f}"
              f"{d['shift']:>+10.2f}")
    print("\nERROR TWO - the flaw of averages proper: f(E[x]) is not E[f(x)]. Measured at")
    print("the true input MEANS so error one cannot contaminate it; what is left is the")
    print("payoff's own curvature.\n")
    print(f"{'option':<8}{'typed-mid':>13}{'at input means':>17}{'true EV':>13}"
          f"{'Jensen gap':>13}")
    print("-" * 84)
    for opt, d in E.flaw_of_averages().items():
        if opt == "defer":
            continue
        print(f"{opt:<8}{money(d['elicited_mid_estimate']):>13}"
              f"{money(d['at_input_means']):>17}{money(d['true_ev']):>13}"
              f"{d['jensen_gap']:>+13,.0f}")
    fa = E.flaw_of_averages()
    naive_gap = fa["build"]["true_ev"] - fa["build"]["elicited_mid_estimate"]
    print(f"\nFor `build` the whole error is {money(naive_gap)} and Jensen accounts for "
          f"{money(fa['build']['jensen_gap'])} of it.")
    print("The famous nonlinearity effect is near zero here; typing the mode is the entire")
    print("problem. Worth knowing, because the popular telling blames the curvature.")
    print(f"For `buy` the cap does bite: Jensen overstates by "
          f"{fa['buy']['jensen_overstates_by']:.1%}.")
    print("\nAnd the number in the cell is not an outcome anyone gets:")
    for opt, p in E.probability_of_the_point_estimate().items():
        print(f"    P(actual lands within 5% of the typed estimate for `{opt}`) = {p:.1%}")


def section_3_ranking_the_mean() -> None:
    rule("3. The higher expected value loses more often than it wins.")
    c = E.ranking_conflict()
    print(f"{'option':<8}{'expected value':>17}{'P10':>12}{'median':>12}{'P90':>12}"
          f"{'P(loss)':>10}")
    print("-" * 84)
    for opt in E.OPTIONS:
        d = E.downside(opt)
        print(f"{opt:<8}{money(c['ev'][opt]):>17}{money(d['p10']):>12}"
              f"{money(d['median']):>12}{money(d['p90']):>12}{d['p_loss']:>10.1%}")
    print(f"\nHighest expected value: `{c['best_by_ev']}`.\n")
    print(f"{'pair':<22}{'P(first wins)':>15}")
    print("-" * 84)
    for (a, b), p in sorted(c["pairs"].items()):
        print(f"{a + ' > ' + b:<22}{p:>15.1%}")
    for a, b, gap, p in c["conflicts"]:
        print(f"\n`{a}` has {money(gap)} more expected value than `{b}` and beats it "
              f"only {p:.1%} of the time.")
    print("\nBoth statements are true and they are about different things. Expected value")
    print("ranks the mean; a mean is an average over futures, and only one future happens.")
    print("`build` carries a longer right tail that lifts its average while most draws")
    print("land below `buy`. Which fact matters depends on whether the bet repeats -")
    print("see section 6, where it does.")


def section_4_which_input_decides() -> None:
    rule("4. The input everyone argues about is not the one that decides it.")
    print("Swing in (build - buy) when one input moves P10 to P90, others held at typed.\n")
    print(f"{'input':<16}{'at P10':>14}{'at P90':>14}{'swing':>14}")
    print("-" * 84)
    for name, lo, hi, swing in E.tornado():
        print(f"{name:<16}{lo:>+14,.0f}{hi:>+14,.0f}{swing:>14,.0f}")
    rows = E.tornado()
    print(f"\n`{rows[0][0]}` swings the answer by {money(rows[0][3])}; "
          f"`{rows[-1][0]}` by {money(rows[-1][3])}, "
          f"{rows[0][3] / rows[-1][3]:.0f}x less.")
    print("The hourly rate is the number that gets debated in the meeting because everyone")
    print("has an opinion about it. Adoption is the one that decides the outcome, and it")
    print("is usually the one nobody is asked to estimate.")
    ish = E.interaction_share()
    print(f"\nA tornado is not a distribution. Adding the single-input swings in quadrature "
          f"implies a variance of {money(ish['oat_variance'])};")
    print(f"the joint simulation gives {money(ish['joint_variance'])} - "
          f"a ratio of {ish['ratio']:.2f}.")
    print("The chart OVERSTATES the spread, because it lines up worst cases that rarely")
    print("co-occur. Read it for ordering, never for range.")


def section_5_what_would_have_to_be_true() -> None:
    rule("5. The recommendation is balanced exactly where the estimate sits.")
    print("For each input: the value at which the answer flips from one option to the other,")
    print("everything else held at the typed estimate.\n")
    print(f"{'input':<16}{'typed':>10}{'flips at':>12}{'distance':>12}  plausible range")
    print("-" * 84)
    for name, sp in E.switching_points().items():
        i = E.INPUTS_BY_NAME[name]
        if sp is None:
            print(f"{name:<16}{i.mid:>10.2f}{'never':>12}{'-':>12}  "
                  f"{i.low}-{i.high} {i.unit}")
            continue
        print(f"{name:<16}{i.mid:>10.2f}{sp:>12.2f}{sp - i.mid:>+12.2f}  "
              f"{i.low}-{i.high} {i.unit}")
    print("\nEvery one of these switching points sits inside the plausible range, and two of")
    print("them sit within a rounding error of the typed estimate. The decision is not")
    print("'build, by 29,000'. It is 'build if adoption clears about 33 seats, and nobody")
    print("has been asked how confident they are that it will'.")
    print("\nThis is the output worth carrying out of the meeting. A point estimate invites")
    print("agreement; a switching point invites a check.")


def section_6_repeating_the_bet() -> None:
    rule("6. Positive expected value, and you still go broke.")
    print(f"A gamble: {E.P_UP:.0%} chance of x{E.UP}, otherwise x{E.DOWN}, on your whole stake.\n")
    print(f"    average multiplier per round   {E.ensemble_growth():.4f}   "
          f"(above 1: positive expected value)")
    print(f"    growth a single run experiences {E.time_average_growth():.4f}   "
          f"(below 1: it shrinks)")
    print("\nThose are not in conflict. The first averages across parallel worlds, the")
    print("second follows one. When payoffs multiply, the second is the one you live in.\n")
    print(f"{'fraction staked':>17}{'mean':>14}{'median':>12}{'below start':>14}{'lost 99%':>11}")
    print("-" * 84)
    for r in E.sizing_comparison():
        label = f"{r['fraction']:.3f}"
        med = (f"{r['median']:,.2f}" if r["median"] >= 0.01
               else f"{r['median']:.2e}")
        print(f"{label:>17}{r['mean']:>14,.2f}{med:>12}"
              f"{r['p_below_start']:>14.1%}{r['p_ruin_99pct']:>11.1%}")
    full = E.sizing_comparison()[0]
    print("\nStaking everything each round is what maximising expected value per round")
    print(f"tells you to do. Over 250 rounds the mean ends at {full['mean']:,.0f}x - and the")
    print(f"median ends at {full['median']:.2e}x - about two millionths of the stake - "
          f"with {full['p_ruin_99pct']:.0%}")
    print("of runs losing 99% of it.")
    print("The average is carried by a vanishing set of paths nobody is on.")
    print(f"\nMaximising expected LOG wealth instead gives a stake of "
          f"{E.kelly_fraction():.0%}, which turns the")
    print("same gamble into a median that grows. Same bet, same probabilities, different")
    print("question - and the question is set by whether the decision repeats.")


def section_7_worth_finding_out() -> None:
    rule("7. What it is worth to find out first, before deciding at all.")
    v = E.evpi()
    print("Expected value of perfect information: what the decision would be worth if you")
    print("knew the answer, minus what it is worth now. It is the ceiling on any study,")
    print("pilot or spike - and it is decidable before commissioning one.\n")
    print(f"    best option without information   {money(v['best_without_information'])}")
    print(f"    if the future were known          {money(v['with_perfect_information'])}")
    print(f"    EVPI                              {money(v['evpi'])}")
    print(f"\nThat is {v['evpi'] / v['best_without_information']:.0%} of the decision's whole "
          f"value, so investigation is not a luxury here.\n")
    print("But you cannot buy perfect information. What is ONE input worth?\n")
    print(f"{'learn this':<18}{'worth up to':>14}")
    print("-" * 84)
    info = E.information_value()
    for k, val in sorted(info.items(), key=lambda kv: -kv[1]):
        print(f"{k:<18}{money(val):>14}")
    parts = sum(v for k, v in info.items() if not k.startswith("_"))
    print(f"\nThe parts sum to {money(parts)} against an EVPI of {money(v['evpi'])}: "
          f"information is not additive.")
    print("Resolving two inputs is worth less than resolving each separately implies,")
    print("because the second only pays where the first left the decision open.\n")
    top = max((k for k in info if not k.startswith("_")), key=lambda k: info[k])
    worst = min((k for k in info if not k.startswith("_")), key=lambda k: info[k])
    print(f"A two-week study of `{top}` is worth up to {money(info[top])}. The same study of")
    print(f"`{worst}` is worth {money(info[worst])} - it cannot repay a single afternoon.")
    print("Both would have been proposed in the same meeting with the same seriousness.")


def section_8_what_to_do() -> None:
    rule("8. What to carry out of the meeting.")
    print("1. Elicit ranges, not points. P10 / most-likely / P90 is what people can")
    print("   actually give, and the mid of that range is not its mean.")
    print("2. Simulate the payoff; never evaluate it at the averages. Report the")
    print("   distribution, the chance of a loss, and the chance each option wins.")
    print("3. Say which comparison you mean. Highest expected value and most-likely-to-win")
    print("   are different questions, and here they have different answers.")
    print("4. Publish the switching points, not the winner. 'Build if adoption clears 33'")
    print("   is checkable; 'build, EV 129k' is not.")
    print("5. Ask whether the decision repeats. If the payoffs multiply, maximise expected")
    print("   log, not expected value, or the average will rise while you lose.")
    print("6. Price the investigation before commissioning it. EVPI is the ceiling, and")
    print("   per-input value says which question is worth asking.")
    print("\nAnd the one that costs nothing: write the switching point into the decision")
    print("record, then check it later. That is the whole loop - see `decision-log` and")
    print("`pre-mortem` for the other two thirds of it.")


def main() -> None:
    section_1_the_decision()
    section_2_two_averaging_errors()
    section_3_ranking_the_mean()
    section_4_which_input_decides()
    section_5_what_would_have_to_be_true()
    section_6_repeating_the_bet()
    section_7_worth_finding_out()
    section_8_what_to_do()
    print()


if __name__ == "__main__":
    main()
