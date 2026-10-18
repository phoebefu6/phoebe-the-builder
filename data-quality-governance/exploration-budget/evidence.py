"""Run the study, write results.json (read by the chart, the app and the tests) and evidence.txt."""

from __future__ import annotations

import json

import explore as E


def pct(v: dict) -> str:
    return f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"


def usd(v: float) -> str:
    return f"${v:,.0f}"


def main() -> None:
    r = E.study()
    json.dump({"main": r, "world": E.WORLD}, open("results.json", "w"), indent=1)

    out = []
    w = out.append
    w(
        f"exploration-budget evidence  (seeds {', '.join(map(str, r['seeds']))}; {r['n']:,} applicants per period; "
        f"{r['reps']} replayed slices per point; +${E.GAIN:,.0f} per repaid loan, -${E.LOSS:,.0f} per default; "
        f"costs per {E.PER:,} applicants; mean [min-max] across seeds)"
    )
    w("")
    w("1. What each design has to lend into")
    w(
        f"true repay among approvals: oracle {pct(r['truth']['oracle'])}, label-trained {pct(r['truth']['label'])}"
    )
    w(f"repay rate of rows production refused (uniform's extra loans): {pct(r['rejected_repay'])}")
    w(
        f"repay rate of rows a challenger wants and production refused (targeted's pool): {pct(r['challenger_only_repay'])}"
    )
    w(f"oracle approvals the label-trained model also approves: {pct(r['agree'])}")
    w("")
    w("2. Cost and power by design (P = chance the slice ranks the oracle above the label-trained model)")
    w(
        f"{'design':<16} {'extra loans':>12} {'net cost':>10} {'P right (worst seed)':>22} {'95% CI on gap':>15} {'bias oracle':>12}"
    )
    for design, grid in (("uniform", E.UNIFORM), ("targeted", E.TARGETED)):
        for s in grid:
            p = r[f"{design}_{s:g}"]
            w(
                f"{design + ' ' + format(s, '.2%'):<16} {p['extra_loans']['mean']:>12,.0f} {usd(p['net_cost']['mean']):>10} "
                f"{p['p_right']['mean']:>12.1%} ({p['p_right']['lo']:.1%}) {'+/-' + format(p['diff_ci']['mean'] * 100, '.1f') + ' pts':>15} "
                f"{p['bias_oracle']['mean'] * 100:>+10.2f} pts"
            )
    w("")
    w("3. Cheapest point that ranks the models right with 95% probability at every seed")
    best = {}
    for design in ("uniform", "targeted"):
        b = E.cheapest(r, design)
        best[design] = b
        w(
            f"{design:<9} {b['share']:.2%} slice: {b['extra_loans']:,.0f} extra loans, net cost {usd(b['net_cost'])}"
            if b
            else f"{design}: none on the grid"
        )
    if best["uniform"] and best["targeted"]:
        w(
            f"targeted is {best['uniform']['net_cost'] / best['targeted']['net_cost']:.1f}x cheaper for the same verdict"
        )
    u, t = r["uniform_0.005"], r["targeted_0.02"]
    w(
        f"at about the same spend: uniform 0.50% {usd(u['net_cost']['mean'])} -> P {u['p_right']['mean']:.1%}, CI +/-{u['diff_ci']['mean'] * 100:.1f}; "
        f"targeted 2.00% {usd(t['net_cost']['mean'])} -> P {t['p_right']['mean']:.1%}, CI +/-{t['diff_ci']['mean'] * 100:.1f}"
    )
    w("")
    w("4. What targeted cannot do: the fairness audit (oracle's TPR for group B)")
    w(
        f"true TPR B {pct(r['true_tpr_b'])}; B good payers no design but uniform can ever label: {pct(r['unreachable_good_b'])}"
    )
    for design, s in (("uniform", 0.05), ("targeted", 0.05), ("targeted", 0.5)):
        p = r[f"{design}_{s:g}"]
        w(
            f"  {design} {s:.0%}: TPR B bias {p['tpr_b_bias']['mean'] * 100:+.1f} pts [{p['tpr_b_bias']['lo'] * 100:+.1f} to {p['tpr_b_bias']['hi'] * 100:+.1f}]"
        )
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
