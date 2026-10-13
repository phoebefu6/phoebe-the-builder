"""Run the study once, write results.json (read by the chart, the app and the notebook test) and evidence.txt."""

from __future__ import annotations

import json

import fairness as F

POLICY_ORDER = ["one threshold (max accuracy)", "demographic parity", "equal opportunity (TPR)",
                "predictive parity (PPV)", "equalized odds (TPR+FPR)"]
AUDIT_N = [25, 50, 100, 200, 500, 1000, 2000]


def _clean(p: dict) -> dict:
    t = p["tA"]
    return {"tA": list(t) if isinstance(t, tuple) else t, "tB": p["tB"], "per": p["per"], "overall": p["overall"],
            "gaps": p["gaps"]}


def policy_lines(pols: dict) -> list:
    out = [f"{'policy':<30} {'accuracy':>9} {'4/5 ratio':>10} {'sel A':>6} {'sel B':>6} {'TPR A':>6} {'TPR B':>6} "
           f"{'FPR A':>6} {'FPR B':>6} {'PPV A':>6} {'PPV B':>6}"]
    for n in POLICY_ORDER:
        p, o = pols[n]["per"], pols[n]["overall"]
        out.append(f"{n:<30} {o['acc']:>9.1%} {o['impact_ratio']:>10.3f} {p['A']['sel']:>6.1%} {p['B']['sel']:>6.1%} "
                   f"{p['A']['tpr']:>6.1%} {p['B']['tpr']:>6.1%} {p['A']['fpr']:>6.1%} {p['B']['fpr']:>6.1%} "
                   f"{p['A']['ppv']:>6.1%} {p['B']['ppv']:>6.1%}")
    return out


def main() -> None:
    pols = F.policies()
    imp = F.impossibility()
    one = pols["one threshold (max accuracy)"]
    audit = F.audit_power(one["tA"], AUDIT_N)
    fix = F.data_fix()
    cal = F.calibrate()
    res = {"world": F.WORLD, "auc": {g: F.auc(g) for g in F.GROUPS}, "pooled_auc": F.pooled_auc(),
           "policies": {n: _clean(pols[n]) for n in POLICY_ORDER}, "impossibility": imp, "audit": audit,
           "fix": {"auc_B": fix["auc_B"], "policies": {n: _clean(fix["policies"][n]) for n in POLICY_ORDER}},
           "calibration": cal}
    json.dump(res, open("results.json", "w"), indent=1)

    out = []
    w = out.append
    w(f"disparate-impact evidence  (seed {F.SEED}, exact from binormal scores, Monte Carlo n = {F.MC_REPS:,} per policy)")
    w("")
    w("1. The model is fine overall")
    w(f"{'group':<6} {'share':>6} {'base rate':>10} {'AUC':>6}")
    for g in F.GROUPS:
        w(f"{g:<6} {F.WORLD[g]['share']:>6.0%} {F.WORLD[g]['base']:>10.0%} {F.auc(g):>6.3f}")
    o = one["overall"]
    w(f"pooled AUC {F.pooled_auc():.3f}, accuracy {o['acc']:.1%} at one threshold ({one['tA']:.2f}); "
      f"four-fifths ratio {o['impact_ratio']:.3f} (rule: >= {F.FOUR_FIFTHS})")
    w(f"qualified applicants selected: A {one['per']['A']['tpr']:.1%}, B {one['per']['B']['tpr']:.1%}")
    w("")
    w("2. Five policies, each enforcing one definition")
    for line in policy_lines(pols):
        w(line)
    eo = pols["equalized odds (TPR+FPR)"]
    t1, t2, q = eo["tA"]
    w(f"equalized odds: A is randomised - threshold {t1:.2f} for {q:.0%} of A applicants, {t2:.2f} for the rest; "
      f"B at {eo['tB']:.2f}")
    w("accuracy cost vs one threshold: " + ", ".join(
        f"{n.split(' (')[0]} {pols[n]['overall']['acc'] - o['acc']:+.1%}" for n in POLICY_ORDER[1:]))
    w("")
    w("3. Three definitions cannot all hold (base rates differ)")
    c = imp["closest"]
    w(f"threshold pairs checked: {imp['pairs']:,}; within {imp['tol']:.0%} on two of PPV/TPR/FPR: {imp['two_of_three']:,}; "
      f"on all three: {imp['all_three']}")
    w(f"closest pair (A {c['tA']:.2f}, B {c['tB']:.2f}): PPV gap {c['ppv_gap']:.3f}, TPR gap {c['tpr_gap']:.3f}, "
      f"FPR gap {c['fpr_gap']:.3f}")
    w("")
    w("4. Can an audit sample tell? Four-fifths verdict on a sample vs the population (one threshold)")
    w(f"{'n of B':>7} {'verdict agrees':>15} {'ratio 5th pct':>14} {'ratio 95th pct':>15}")
    for r in audit:
        w(f"{r['n_B']:>7} {r['agree']:>15.1%} {r['ratio_p5']:>14.3f} {r['ratio_p95']:>15.3f}")
    w("")
    w(f"5. Fix the data instead: B's score as informative as A's (AUC B {F.auc('B'):.3f} -> {fix['auc_B']:.3f}), base rates unchanged")
    for line in policy_lines(fix["policies"]):
        w(line)
    fo = fix["policies"]["one threshold (max accuracy)"]["overall"]
    w(f"one threshold now: accuracy {o['acc']:.1%} -> {fo['acc']:.1%}, four-fifths ratio {o['impact_ratio']:.3f} -> "
      f"{fo['impact_ratio']:.3f}")
    w("")
    inside = sum(r["inside"] for r in cal)
    w(f"6. Calibration: {inside}/{len(cal)} exact rates inside the 99% Wilson interval of the simulation")
    open("evidence.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
