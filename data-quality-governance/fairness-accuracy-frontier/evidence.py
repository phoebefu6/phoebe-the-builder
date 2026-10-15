"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import frontier as F


def main() -> None:
    r = F.study()
    r["world"] = F.WORLD
    json.dump(r, open("results.json", "w"), indent=1)

    P, pr = r["points"], r["prices"]
    out = []
    w = out.append
    w(f"fairness-accuracy-frontier evidence  (seeds {', '.join(map(str, r['seeds']))}, n = {r['n']:,} each, "
      f"half train / half test, gradient boosting, top {F.SELECT:.0%} selected; mean [min-max] across seeds)")
    w("")
    w("1. Where each model lands (one cut-off for everyone; hit rate = share of the selected who are positive)")
    w(f"{'model':<30} {'B/A ratio':>16} {'hit vs history':>15} {'hit vs truth':>13} {'pts below frontier':>19}")
    for k in F.MODELS:
        v = P[k]
        rng = f"{v['ratio']['mean']:.3f} [{v['ratio']['lo']:.2f}-{v['ratio']['hi']:.2f}]"
        gb = r["gap_below"].get(k)
        gbs = f"{gb['mean']:+.2f}" if gb else "off curve"
        w(f"{k:<30} {rng:>16} {v['hit_hist']['mean']:>15.1%} {v['hit_true']['mean']:>13.1%} "
          f"{gbs:>19}")
    w(f"selecting by true skill itself gives a ratio of {r['oracle_ratio']['mean']:.3f}")
    w("")
    w("2. The frontier (group-specific cut-offs on the aware score, total held at 30%)")
    w(f"hit vs history peaks at ratio {r['peak_hist']['ratio']['mean']:.2f} ({r['peak_hist']['hit']['mean']:.1%}); "
      f"hit vs truth peaks at ratio {r['peak_true']['ratio']['mean']:.2f} ({r['peak_true']['hit']['mean']:.1%})")
    w(f"moving from the unconstrained model (ratio {r['start']['mean']:.2f}) to four-fifths (0.80): "
      f"vs history {-pr['fourfifths_hit_hist']['mean']:+.2f} pts, vs truth {-pr['fourfifths_hit_true']['mean']:+.2f} pts")
    w(f"moving from the unconstrained model to parity (1.00): "
      f"vs history {-pr['parity_hit_hist']['mean']:+.2f} pts [{-pr['parity_hit_hist']['hi']:+.2f} to {-pr['parity_hit_hist']['lo']:+.2f}], "
      f"vs truth {-pr['parity_hit_true']['mean']:+.2f} pts [{-pr['parity_hit_true']['hi']:+.2f} to {-pr['parity_hit_true']['lo']:+.2f}]")
    w("")
    w("3. The price as the history gets more biased (label_bias = 0 is the control: the history IS the truth)")
    w(f"{'label_bias':>10} {'start ratio':>12} {'parity vs history':>18} {'parity vs truth':>16} {'truth peak at':>14}")
    for row in r["sweep"]:
        w(f"{row['label_bias']['mean']:>10.2f} {row['start']['mean']:>12.2f} {-row['parity_hit_hist']['mean']:>+17.2f}  "
          f"{-row['parity_hit_true']['mean']:>+15.2f} {row['peak_true']['mean']:>14.2f}")
    c = r["sweep"][0]
    w("control, hit vs truth along the curve: " + "  ".join(f"r={x}: {c[f'true_at_{x}']['mean']:.1f}%" for x in F.READ_AT))
    w("")
    w(f"4. The price of four-fifths (vs history) as a team would estimate it on a validation sample ({F.AUDIT_REPS} draws)")
    for m, v in r["audit"].items():
        w(f"n = {int(m):>6,}: median {v['p50']['mean']:.2f} pts, 90% of samples between {v['p05']['mean']:.2f} and "
          f"{v['p95']['mean']:.2f}; samples showing no cost at all {v['share_le_0']['mean']:.1%}")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
