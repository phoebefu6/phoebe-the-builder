"""Run the study (main world and control), write results.json (read by the chart, the app and the tests) and evidence.txt."""

from __future__ import annotations

import json

import selective as S

EST = [
    ("truth", "truth (all outcomes)"),
    ("labeled", "labeled rows only"),
    ("rejected_bad", "rejected = bad"),
    ("ipw", "IPW on x, group"),
    ("explore_0.05", "5% random explore"),
]


def pct(v: dict) -> str:
    return f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"


def pts(v: dict) -> str:
    return f"{v['mean'] * 100:+.1f} [{v['lo'] * 100:+.1f} to {v['hi'] * 100:+.1f}]"


def main() -> None:
    r = S.study()
    c = S.study(w=S.CONTROL)
    json.dump(
        {"main": r, "control": c, "world": S.WORLD, "control_world": S.CONTROL},
        open("results.json", "w"),
        indent=1,
    )

    out = []
    w = out.append
    w(
        f"selective-labels evidence  (seeds {', '.join(map(str, r['seeds']))}; {r['n']:,} applicants per period; "
        f"{S.APPROVE:.0%} approved; gradient boosting on x and group; mean [min-max] across seeds)"
    )
    w("")
    w("1. The label set (outcomes observed only for the 40% the officers approved)")
    w(
        f"repay rate, all applicants: {pct(r['pop_repay'])}; labeled (approved) rows: {pct(r['hist_repay_labeled'])}"
    )
    w(
        f"labeled repay rate by group: A {pct(r['a_labeled_repay'])}, B {pct(r['b_labeled_repay'])} (B cleared a {S.WORLD['bar_b']} sd higher bar)"
    )
    w("")
    w("2. Two models scored on the truth (approve the top 40%)")
    w(f"{'model':<22} {'repay among approved':>22} {'share of B approved':>22} {'TPR A':>8} {'TPR B':>8}")
    for key, name, sb in (
        ("eval_full_model", "oracle (all outcomes)", "oracle"),
        ("eval_lab_model", "trained on labels", "labeled"),
    ):
        t = r[key]["truth"]
        w(
            f"{name:<22} {pct(t['repay']):>22} {pct(r['share_b_approved'][sb]):>22} {t['tpr_A']['mean']:>8.1%} {t['tpr_B']['mean']:>8.1%}"
        )
    w("")
    w("3. The same two models as each estimator reports them")
    w(
        f"{'estimator':<22} {'oracle repay':>22} {'label-trained repay':>22} {'oracle TPR gap A-B (pts)':>28}  ranks right?"
    )
    for k, name in EST:
        o, lm = r["eval_full_model"][k], r["eval_lab_model"][k]
        # seed ranges that overlap are a tie, not a ranking
        right = (
            "yes"
            if o["repay"]["lo"] > lm["repay"]["hi"]
            else "NO"
            if o["repay"]["hi"] < lm["repay"]["lo"]
            else "tie"
        )
        w(f"{name:<22} {pct(o['repay']):>22} {pct(lm['repay']):>22} {pts(o['tpr_gap']):>28}  {right}")
    w("")
    w(
        "4. How big the random-explore slice must be (oracle model, repay among approved; truth "
        f"{r['eval_full_model']['truth']['repay']['mean']:.1%})"
    )
    for e in S.EXPLORE:
        v = r["eval_full_model"][f"explore_{e:g}"]
        w(
            f"  explore {e:>4.0%}: {v['n_approved']['mean']:>6,.0f} labeled approvals, estimate {pct(v['repay'])}, "
            f"95% CI +/- {v['ci_half']['mean'] * 100:.1f} pts"
        )
    w("")
    w("5. Control: officers used recorded data only (sel_u = 0)")
    for k, name in EST[:4]:
        o = c["eval_full_model"][k]
        w(f"{name:<22} oracle repay {pct(o['repay'])}   TPR B {pct(o['tpr_B'])}")
    w(
        f"label-trained model, true repay {pct(c['eval_lab_model']['truth']['repay'])}; share of B approved "
        f"{pct(c['share_b_approved']['labeled'])} vs oracle {pct(c['share_b_approved']['oracle'])}"
    )
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
