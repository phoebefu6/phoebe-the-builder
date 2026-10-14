"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import proxy as P


def model_lines(models: dict) -> list:
    out = [f"{'model':<36} {'B/A ratio':>16} {'AUC vs history':>15} {'AUC vs truth':>13} "
           f"{'group from score':>17} {'qualified sel A':>16} {'qualified sel B':>16}"]
    for k in P.MODELS:
        v = models[k]
        rng = f"{v['ratio']['mean']:.3f} [{v['ratio']['lo']:.2f}-{v['ratio']['hi']:.2f}]"
        out.append(f"{k:<36} {rng:>16} {v['auc_hist']['mean']:>15.3f} {v['auc_true']['mean']:>13.3f} "
                   f"{v['group_from_score']['mean']:>17.3f} {v['qual_sel_a']['mean']:>16.1%} {v['qual_sel_b']['mean']:>16.1%}")
    return out


def main() -> None:
    r = P.study()
    fl = P.flagged(r["screen"])
    r["flagged"] = fl
    r["world"] = P.WORLD
    json.dump(r, open("results.json", "w"), indent=1)

    t = r["truth"]
    s = r["screen"]
    out = []
    w = out.append
    w(f"proxy-variable-finder evidence  (seeds {', '.join(map(str, r['seeds']))}, n = {r['n']:,} each, "
      f"half train / half test, gradient boosting; mean [min-max] across seeds)")
    w("")
    w("1. The world: same skill, biased history")
    w(f"skill gap A minus B {t['skill_gap']:+.3f} sd; truly qualified A {t['true_rate_a']:.1%}, B {t['true_rate_b']:.1%}")
    w(f"past decisions (the training labels) selected A {t['hist_rate_a']:.1%}, B {t['hist_rate_b']:.1%}")
    w("")
    w("2. Three proxy screens (AUC for predicting the group; 0.5 = no information)")
    w("   one feature at a time:  " + "  ".join(f"{f} {v['mean']:.3f}" for f, v in s["single"].items()))
    top_pairs = sorted(s["pairs"].items(), key=lambda kv: -kv[1]["mean"])
    hidden = [(p, v) for p, v in top_pairs if p in fl["pairs_hidden_from_single"]]
    w("   pairs (hidden from the single screen):  " + "  ".join(f"{p} {v['mean']:.3f}" for p, v in hidden))
    w(f"   all five together: {s['joint']['mean']:.3f}")
    w(f"   flagged at AUC > 0.60 by the single screen: {', '.join(fl['single'])}; "
      f"missed by it, caught by pairs: {', '.join(fl['pairs_hidden_from_single'])}")
    w("   drop-one importance (joint AUC lost when the feature is removed):  "
      + "  ".join(f"{f} {v['mean']:.3f}" for f, v in sorted(s["drop_one"].items(), key=lambda kv: -kv[1]["mean"])))
    w("")
    w(f"3. Seven models, all trained on the biased history, top {P.SELECT:.0%} selected")
    for line in model_lines(r["models"]):
        w(line)
    a, best = r["models"]["aware (group as a feature)"], r["models"]["drop zip + pair, repair exp"]
    w(f"aware -> drop zip + pair, repair exp: AUC vs history {a['auc_hist']['mean']:.3f} -> {best['auc_hist']['mean']:.3f}, "
      f"AUC vs truth {a['auc_true']['mean']:.3f} -> {best['auc_true']['mean']:.3f}")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
