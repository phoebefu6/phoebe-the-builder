"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import audit as A


def pct(v: dict) -> str:
    return f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"


def main() -> None:
    r = A.study()
    r["world"] = A.WORLD
    json.dump(r, open("results.json", "w"), indent=1)

    out = []
    w = out.append
    w(
        f"sampling-bias-audit evidence  (seeds {', '.join(map(str, r['seeds']))}; training set n = {r['n_train']:,}, "
        f"deployment population n = {r['n_pop']:,}; gradient boosting, approve when p >= 0.5; mean [min-max] across seeds)"
    )
    w("")
    w(
        "1. The audit (training share / population share; flag outside [0.80, 1.25] when the 99% Wilson interval agrees)"
    )
    w(f"{'segment':<14} {'kind':<9} {'rows':>6} {'train':>7} {'population':>11} {'ratio':>6}  flag")
    for name, v in r["audit"].items():
        w(
            f"{name:<14} {v['kind']:<9} {v['rows']:>6,} {v['train_share']:>7.1%} {v['pop_share']:>11.1%} "
            f"{v['ratio']:>6.2f}  {'FLAG' if v['flag'] else 'ok'}"
        )
    ctrl = [k for k, v in r["audit_control"].items() if v["flag"]]
    w(f"control (a proportional sample of the same size): {len(ctrl)} segments flagged")
    w("")
    w("2. What the miss costs (share of good payers rejected, scored on the deployment population)")
    w(f"{'cell':<10} {'quota-trained':>22} {'proportional-trained':>22}")
    for c in ["overall"] + A.CELLS:
        w(
            f"{c:<10} {pct(r['quota'][c]['good_rejected']):>22} {pct(r['proportional'][c]['good_rejected']):>22}"
        )
    w(
        f"overall accuracy: the team's own test split {pct(r['quota_on_own_test']['overall']['acc'])}, "
        f"deployment {pct(r['quota']['overall']['acc'])}, proportional-trained {pct(r['proportional']['overall']['acc'])}"
    )
    w("")
    w("3. Two fixes for rural B (good payers rejected)")
    w(
        f"reweight the 100 rows to the population cell shares: {pct(r['reweighted']['B-rural']['good_rejected'])}"
    )
    w(
        f"a training set with NO rural B rows: {pct(r['empty']['B-rural']['good_rejected'])}; "
        f"reweighted: {pct(r['empty_reweighted']['B-rural']['good_rejected'])}"
    )
    w("collect more rural B rows (quota sample + k extra):")
    for k, v in zip(A.CURVE, r["curve"]):
        w(f"  {100 + k:>5,} rural B rows: {pct(v)}")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
