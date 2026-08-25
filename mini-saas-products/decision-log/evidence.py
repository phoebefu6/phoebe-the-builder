"""Every claim in the README, printed from the live computation.

Run it:  python evidence.py
"""

from __future__ import annotations

import numpy as np

import declog as D


def rule(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def section_1_the_diary_problem() -> None:
    rule("1. Half the records cannot be scored at all.")
    rep = D.resolvability_report()
    print("A decision log without a prediction is a diary. Before any scoring rule matters,")
    print("the record has to contain something that can turn out to be wrong.\n")
    print(f"{'id':<7} {'prob':>6} {'resolve by':<12} {'metric':<28} scoreable")
    print("-" * 84)
    for r in D.RECORDS:
        p = f"{r.probability:.2f}" if r.probability is not None else "-"
        print(f"{r.id:<7} {p:>6} {(r.resolve_by or '-'):<12} "
              f"{(r.metric or '-'):<28} {'yes' if D.resolvable(r) else 'NO'}")
    print(f"\n{rep['resolvable']} of {rep['n']} records are scoreable.")
    print("Field by field, of", rep["n"], "records:")
    for k, v in rep["per_field"].items():
        print(f"    {k:<22} {v:>3}")
    print("\nThe corpus is illustrative - it is written, not sampled - so the rate above is a")
    print("property of these 20 records, not a measurement of the world. The LINTER is the")
    print("reusable part: four fields, and a record missing any one of them can never be")
    print("scored, no matter how carefully the decision was reasoned.")
    print("\nNote the pattern in the pairs: every even-numbered record is the odd one above it,")
    print("rewritten. 'query costs will fall' and 'median dashboard query latency drops below")
    print("4s' record the same decision. Only one of them can ever be wrong.")


def section_2_the_rules() -> None:
    rule("2. Six scoring rules. Three of them pay your team to lie.")
    print(f"{'rule':<19} {'bounded':<8} where you meet it")
    print("-" * 84)
    for r in D.RULES:
        print(f"{r.name:<19} {str(r.bounded):<8} {r.seen_in}")
    print("\nA rule is PROPER when the report that minimises expected loss is the report the")
    print("forecaster actually believes. Propriety is not asserted here - it is computed, by")
    print("optimising the expected score over 1001 candidate reports for 99 true beliefs.\n")
    print(f"{'rule':<19} {'proper':<8} {'worst gap':>10} {'beliefs misreported':>21}")
    print("-" * 84)
    for r in D.RULES:
        ok, gap, bad = D.propriety(r.name)
        print(f"{r.name:<19} {str(ok):<8} {gap:>10.3f} {len(bad):>15} of 99")


def section_3_the_optimal_lie() -> None:
    rule("3. What an optimising forecaster reports, under each rule.")
    print("Read the rows: 'I believe 55%, so I will say ___ because that scores best.'\n")
    beliefs = (0.55, 0.6, 0.7, 0.8, 0.9)
    print(f"{'rule':<19}" + "".join(f"{'p=' + str(p):>10}" for p in beliefs))
    print("-" * 84)
    for r in D.RULES:
        row = dict(D.optimal_lie_table(r.name, beliefs))
        print(f"{r.name:<19}" + "".join(f"{row[p]:>10.2f}" for p in beliefs))
    print("\n`absolute` and `confidence_points` both pay 1.00 for every belief above a coin")
    print("flip. A forecaster who believes 55% and reports 55% is scored WORSE than one who")
    print("believes 55% and claims certainty. That is not a subtle distortion - it is the")
    print("rule instructing your team to hide what they know.")
    print("\n`threshold_01` is different and no better. It has no single optimum at all:")
    for r in D.RULES:
        lo, hi, w = D.optimal_report_set(r, 0.7)
        if w > 0:
            print(f"    {r.name}: every report in [{lo:.2f}, {hi:.2f}] scores identically "
                  f"- a plateau {w:.2f} wide.")
    print("A rule with a wide plateau does not punish confidence, it cannot SEE confidence.")
    print("\nBelow a coin flip the two improper rules fail by different mechanisms:")
    for name in ("absolute", "confidence_points"):
        row = D.optimal_lie_table(name, (0.10, 0.30, 0.45))
        print(f"    {name:<19} " + "  ".join(f"believe {p:.2f} -> say {q:.3f}" for p, q in row))
    print("`absolute` collapses to 0. The points game wagers the NUMBER WRITTEN DOWN, so the")
    print("best play is the largest stake still on the favoured side - 0.499. Two mechanisms,")
    print("one outcome: under neither rule is the report ever the belief.")
    print("'What was our hit rate?' throws away the only number that made the log an")
    print("instrument rather than a list.")


def section_4_ranking_flips() -> None:
    rule("4. The rule does not measure who is best. It decides who is best.")
    t = D.score_table()
    print(f"{'forecaster':<16}" + "".join(f"{r.name[:10]:>12}" for r in D.RULES))
    print("-" * 84)
    for f in D.FORECASTERS:
        print(f"{f.name:<16}" + "".join(f"{t[f.name][r.name]:>12.4f}" for r in D.RULES))
    print("\nSame six forecasters, same 4000 events, six rules. The winner changes:\n")
    for r in D.RULES:
        order = D.ranking(r.name)
        print(f"    {r.name:<19} 1st: {order[0]:<15} last: {order[-1]}")
    print("\n`absolute` - 'average error', the one that ends up in a spreadsheet - ranks the")
    print("OVERCONFIDENT forecaster first and the calibrated one second. A team scoring its")
    print("decision log that way will promote the person who is most often wrongly certain.")
    print("`confidence_points`, the in-house prediction game, ranks the UNDERCONFIDENT one")
    print("first. Two homebrew rules, two opposite wrong answers.\n")
    rd = D.ranking_disagreement()
    off = {k: v for k, v in rd.items() if k[0] != k[1]}
    worst = max(off.values())
    pairs = sorted({tuple(sorted(k)) for k, v in off.items() if v == worst})
    total = len(D.FORECASTERS) * (len(D.FORECASTERS) - 1) // 2
    print(f"Worst rule pair flips {worst} of the {total} forecaster pairings: {pairs}")


def section_4b_log_loss_reversal() -> None:
    rule("4b. Log loss ranks real information BELOW no information.")
    t = D.score_table()
    outcomes, reports = D.simulate()
    q = reports["noisy_expert"]
    misses = int(((q > 0.9) & (outcomes == 0)).sum() + ((q < 0.1) & (outcomes == 1)).sum())
    qb = reports["base_rate"]
    print("`noisy_expert` is unbiased and inconsistent - it has genuine signal and high")
    print("variance. `base_rate` reports the same number every time and knows nothing.\n")
    print(f"{'':<16}{'brier':>10}{'log':>10}")
    print("-" * 40)
    for n in ("noisy_expert", "base_rate"):
        print(f"{n:<16}{t[n]['brier']:>10.4f}{t[n]['log']:>10.4f}")
    print("\nBy Brier, noisy_expert wins. By log loss it finishes LAST of all six - below the")
    print("forecaster that knows nothing.")
    print(f"\nWhy: it makes {misses} confident misses in {len(outcomes)} events "
          f"(reported >0.9 and it did not happen, or <0.1 and it did).")
    print(f"`base_rate` makes {int(((qb > 0.9) | (qb < 0.1)).sum())} - it never commits, so it "
          "can never be caught out.")
    print("Log loss is unbounded, so those few events dominate the mean.")
    print("\nThat is not a bug in log loss - it is what log loss is FOR. It is the right rule")
    print("when a confident miss is genuinely catastrophic, and the wrong one when you are")
    print("trying to find out who in the room is worth listening to. Both are proper. Proper")
    print("is not the whole specification.")


def section_5_calibration_is_not_skill() -> None:
    rule("5. Perfect calibration, zero skill.")
    d = D.decompositions()
    print("Murphy: Brier = reliability - resolution + uncertainty.")
    print("Reliability is 'when you said 70%, did it happen 70% of the time' - the thing")
    print("everyone means by calibration, and the only part a recalibration step can fix.")
    print("Resolution is 'did you separate the cases at all'. It carries the information.\n")
    print(f"{'forecaster':<16}{'brier':>9}{'reliability':>13}{'resolution':>12}"
          f"{'uncertainty':>13}{'residual':>10}")
    print("-" * 84)
    for f in D.FORECASTERS:
        m = d[f.name]
        print(f"{f.name:<16}{m['brier']:>9.4f}{m['reliability']:>13.4f}"
              f"{m['resolution']:>12.4f}{m['uncertainty']:>13.4f}"
              f"{m['check'] - m['brier']:>10.4f}")
    print("\n(The residual is the cost of binning into 10 buckets - the identity is exact only")
    print("when bins are the distinct forecast values. It is reported rather than hidden.)\n")
    br = d["base_rate"]
    print(f"`base_rate` has reliability {br['reliability']:.4f} - PERFECTLY calibrated, because")
    print("it reports the base rate and the base rate is exactly right on average. It also has")
    print(f"resolution {br['resolution']:.4f} and the worst Brier score of the six. It is useless.")
    print("\n'Improve your calibration' would give this forecaster full marks.\n")
    flips = D.reliability_beats_resolution()
    print(f"{len(flips)} ordered pairs where the MORE reliable forecaster has the WORSE Brier:")
    for a, b in flips:
        print(f"    {a:<15} is better calibrated than {b:<15} and scores worse overall")
    print("\nCalibration is necessary and it is not sufficient, and the difference is the whole")
    print("reason to record a probability rather than a direction.")


def section_6_reliability_curve() -> None:
    rule("6. The reliability curve, for the forecaster you probably have.")
    for name in ("calibrated", "overconfident", "base_rate"):
        xs, ys, ns = D.reliability_curve(name)
        print(f"\n{name}")
        print(f"  {'said':>6} {'happened':>10} {'n':>7}   gap")
        for x, y, n in zip(xs, ys, ns):
            bar = "+" * int(abs(x - y) * 60)
            print(f"  {x:>6.2f} {y:>10.2f} {n:>7}   {bar}")
    print("\n`overconfident` is the default human shape: says 90%, happens ~75%; says 10%,")
    print("happens ~25%. The curve is flatter than the diagonal at both ends.")


def section_7_resulting() -> None:
    rule("7. Judging a decision by its outcome is wrong 4 times in 10.")
    r = D.resulting()
    print(f"A decision with a {r['p_win']:.0%} chance of +100 and {1 - r['p_win']:.0%} of -100.")
    print(f"Expected value {r['expected_value']:+.0f}. It is a good decision, every time it is taken.")
    print(f"Over {r['n']} takings it loses {r['share_judged_bad']:.1%} of the time, and an")
    print("outcome-based review calls every one of those a bad decision.\n")
    p = D.resulting_portfolio()
    print(f"A mixed portfolio of {p['n']} real decisions - some genuinely good, some genuinely")
    print("bad - reviewed purely on how they turned out:\n")
    print(f"    truly good decisions      {p['truly_good']:>4} of {p['n']}")
    print(f"    verdicts that are wrong   {p['misjudged']:>4}  ({p['misjudged_rate']:.1%})")
    print(f"      good decisions punished {p['good_called_bad']:>4}")
    print(f"      bad decisions rewarded  {p['bad_called_good']:>4}")
    print("\nThis is why the record has to capture the *prediction*, not just the choice. The")
    print("outcome is a single noisy sample from a distribution the decision only shifted.")
    print("Without the stated probability there is nothing to separate a good decision that")
    print("lost from a bad one that lost, and a review of the log will do it by outcome.")


def section_8_power() -> None:
    rule("8. How many decisions before the log can conclude anything?")
    m = D.power_matrix()
    lo, ln, hi, hn = D.cheapest_and_dearest_comparison()
    print("Paired sample size to separate two forecasters by Brier score, at 80% power.\n")
    print(f"{'forecaster A':<17}{'forecaster B':<17}{'decisions needed':>18}")
    print("-" * 84)
    for (a, b), n in sorted(m.items(), key=lambda kv: kv[1]):
        print(f"{a:<17}{b:<17}{int(n):>18,}")
    med = int(np.median(list(m.values())))
    print(f"\nMedian over all {len(m)} pairings: {med:,} decisions.")
    print(f"Easiest: {lo[0]} vs {lo[1]}, {int(ln):,}. Hardest: {hi[0]} vs {hi[1]}, {int(hn):,}.")
    print("\nA team that logs one decision a week reaches 260 in five years. At that volume it")
    print("can separate a forecaster from the base rate, and almost nothing else. Ranking")
    print("your people by their decision log is the one use it is least able to support.")
    print("\nThat is not an argument against keeping one. It relocates the value: the log is")
    print("worth keeping because it makes the reasoning retrievable and forces the claim to")
    print("be falsifiable at the moment of writing - both of which pay off at n=1.")


def section_9_what_the_record_needs() -> None:
    rule("9. The record, and why each field is there.")
    print("Every field below is load-bearing for something above it.\n")
    print(f"{'field':<20} {'earns it':<12} what breaks without it")
    print("-" * 84)
    rows = [
        ("decision", "§1", "you cannot find the record again"),
        ("claim", "§1", "nothing can turn out to be wrong"),
        ("metric + threshold", "§1", "two readers resolve it differently"),
        ("resolve_by", "§1", "it is never scored, only remembered"),
        ("probability", "§7", "a good decision that lost is indistinguishable"),
        ("scoring rule", "§2-4", "the rule picks the winner, and you did not pick the rule"),
    ]
    for f, s, w in rows:
        print(f"{f:<20} {s:<12} {w}")
    print("\nAnd the rule to use: any of the three proper ones. Brier if the numbers get shown")
    print("to people - it is bounded and reads as an error rate. Log loss if the log feeds a")
    print("model, accepting that one confident miss dominates. Never `absolute`, never a")
    print("hit-rate, and never the points game, however much fun it is.")
    ok = [r.name for r in D.RULES if D.propriety(r.name)[0]]
    bad = [r.name for r in D.RULES if not D.propriety(r.name)[0]]
    print(f"\n  proper:   {ok}")
    print(f"  improper: {bad}")


def main() -> None:
    section_1_the_diary_problem()
    section_2_the_rules()
    section_3_the_optimal_lie()
    section_4_ranking_flips()
    section_4b_log_loss_reversal()
    section_5_calibration_is_not_skill()
    section_6_reliability_curve()
    section_7_resulting()
    section_8_power()
    section_9_what_the_record_needs()
    print()


if __name__ == "__main__":
    main()
