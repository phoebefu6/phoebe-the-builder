"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import escalation as E


def main() -> None:
    table = E.policy_table()
    curve = E.load_curve()
    up = E.upgrade()
    value = E.value_curve()
    cal = E.calibrate()
    res = {"policies": table, "load_curve": curve, "upgrade": up, "value_curve": value, "calibration": cal,
           "base": E.BASE, "segments": E.SEGMENTS, "centers": E.CENTERS.tolist()}
    json.dump(res, open("results.json", "w"), indent=1)

    by = {r["policy"]: r for r in table}
    never, best1, joint = by["never escalate"], by["best single threshold"], by["priced + capacity (joint)"]
    b = E.BASE
    out = []
    w = out.append
    w(f"human-escalation-policy evidence  (seed {E.SEED}, exact on {E.BINS} confidence bins, Monte Carlo {E.MC_DAYS} days)")
    w(f"world: {int(b['tickets']):,} tickets/day, review ${b['review_cost']:.0f}, capacity {int(b['capacity'])}/day, "
      f"human error {b['human_err']:.0%} calm -> {b['human_err_full']:.0%} at full load")
    w("")
    w("1. The segments")
    w(f"{'segment':<9} {'share':>6} {'cost of a wrong action':>23} {'true accuracy':>14} {'mean reported conf':>19}")
    tab = E.segment_tables()
    for s in E.SEGS:
        v = E.SEGMENTS[s]
        acc = float((tab[s]["pmf"] * tab[s]["p_true"]).sum())
        conf = float((tab[s]["pmf"] * E.CENTERS).sum())
        w(f"{s:<9} {v['share']:>6.0%} {'$' + format(v['cost'], '.0f'):>23} {acc:>14.1%} {conf:>19.1%}")
    w("")
    w("2. Seven escalation policies, one day")
    w(f"{'policy':<28} {'escalated':>9} {'reviewed':>9} {'human err':>10} {'error rate':>11} {'cost/day':>9} {'vs never':>9}")
    for r in table:
        w(f"{r['policy']:<28} {r['escalated']:>9.0f} {r['reviewed_share']:>9.0%} {r['human_err']:>10.1%} "
          f"{r['error_rate']:>11.2%} {'$' + format(r['cost'], ',.0f'):>9} {r['cost'] / never['cost'] - 1:>+9.1%}")
    w("joint optimum thresholds: " + ", ".join(f"{s} < {joint['thr'][s]:.3f}" for s in E.SEGS)
      + f"  -> {joint['cost'] / best1['cost'] - 1:+.1%} vs the best single threshold")
    unl = {r["policy"]: r for r in E.policy_table(dict(b, capacity=1e9))}
    w("   same policies with unlimited capacity (the textbook case):")
    for n in ("best single threshold", "priced, stated confidence", "priced, calibrated", "priced + capacity (joint)"):
        w(f"   {n:<28} ${unl[n]['cost']:>9,.0f}")
    res["unlimited"] = {n: unl[n]["cost"] for n in unl}
    json.dump(res, open("results.json", "w"), indent=1)
    w("")
    w("3. Escalating more is not always safer: scale the calibrated priced cutoffs")
    w(f"{'scale':>6} {'escalated':>10} {'demand/capacity':>16} {'human err':>10} {'cost/day':>10}")
    best_pt = min(curve, key=lambda r: r["cost"])
    for r in curve:
        w(f"{r['scale']:>6.2f} {r['escalated']:>10.0f} {r['utilisation_demand']:>16.2f} {r['human_err']:>10.1%} "
          f"{'$' + format(r['cost'], ',.0f'):>10}")
    w(f"cheapest point: scale {best_pt['scale']:.2f}, demand {best_pt['utilisation_demand']:.2f}x capacity, "
      f"${best_pt['cost']:,.0f}/day")
    w("")
    w(f"4. A model upgrade with the same skill that reports higher confidence (inflate {E.INFLATE_V2})")
    w(f"{'':<22} {'escalation rate':>16} {'error rate':>11} {'refund error $':>15} {'cost/day':>10}")
    for k, label in (("before", "v1, joint thresholds"), ("after", "v2, same thresholds"), ("refit", "v2, refitted")):
        r = up[k]
        w(f"{label:<22} {r['escalation_rate']:>16.2%} {r['error_rate']:>11.2%} {'$' + format(r['refund_err_cost'], ',.0f'):>15} "
          f"{'$' + format(r['cost'], ',.0f'):>10}")
    w(f"same thresholds: escalation rate {up['after']['escalation_rate'] - up['before']['escalation_rate']:+.2%} points, "
      f"cost {up['after']['cost'] / up['before']['cost'] - 1:+.1%}")
    w("")
    w("5. What one review is worth, calm queue (dollars saved, reported confidence bin)")
    for s in E.SEGS:
        vals = [(c, v) for c, v in zip(E.CENTERS, value[s]) if v == v]
        pos = [c for c, v in vals if v > 0]
        hi = f"pays below conf {max(pos) + 0.5 / E.BINS:.3f}" if pos else "never pays"
        w(f"   {s:<8} {hi:<26} best bin ${max(v for _, v in vals):.1f}")
    w("")
    inside = sum(r["inside"] for r in cal)
    w(f"6. Calibration: {inside}/{len(cal)} exact per-segment error rates inside the 99% Wilson interval of the simulation")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
