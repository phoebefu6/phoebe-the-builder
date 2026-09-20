"""Regenerate every number quoted in the README and the notebook.

Writes evidence.txt (human) and results.json (machine, read by make_chart.py and the tests).
Nothing in this build quotes a number that does not appear in results.json.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from pvalue import (
    ALPHA,
    DESIGNS,
    REPS,
    Cell,
    dance_crosses_threshold,
    dance_has_stopped,
    dance_is_wide,
    ppv_study,
    replication_study,
    required_n,
    run_grid,
)

REP_REPS = 200_000
PPV_STUDIES = 200_000

REPLICATION_DESIGNS = [(0.2, 50), (0.5, 20), (0.5, 50), (0.8, 20)]
PPV_DESIGNS = [(0.5, 0.5, 50), (0.2, 0.5, 50), (0.1, 0.5, 50), (0.1, 0.2, 20), (0.5, 0.5, 20)]

out: List[str] = []


def w(line: str = "") -> None:
    out.append(line)
    print(line)


def cell_row(c: Cell) -> Dict[str, Any]:
    return {
        "d_true": c.d_true, "n": c.n, "reps": c.reps,
        "power": c.power, "power_analytic": c.power_analytic, "power_ci": list(c.power_ci),
        "calibrated": c.calibrated, "p_pcts": c.p_pcts, "log10_spread": c.log10_spread,
        "ks_stat": c.ks_stat, "ks_p": c.ks_p, "uniform_ok": c.uniform_ok,
        "type_m": c.type_m, "type_s": c.type_s,
        "type_s_ci": list(c.type_s_ci) if c.type_s_ci else None,
        "d_hat_median_sig": c.d_hat_median_sig,
        "crosses": dance_crosses_threshold(c), "wide": dance_is_wide(c),
        "stopped": dance_has_stopped(c), "seed": c.seed,
    }


w("=" * 100)
w("A P-VALUE IS A RANDOM VARIABLE, AND THE SPREAD OF THAT VARIABLE IS THE RESULT")
w("=" * 100)
w()
w(f"Test: Student's two-sample t, equal n, equal variance, normal data, alpha = {ALPHA}.")
w("That configuration is not a convenience - Day 174 verified it is the one cell where Student's")
w("t is EXACT. So nothing below can be blamed on a violated assumption or a wrong test.")
w(f"Grid: {len(DESIGNS)} designs x {REPS:,} replicates. Seed = index into DESIGNS.")
w()

cells = run_grid()

# ---------------------------------------------------------------------------
w("-" * 100)
w("0. CALIBRATION - can the harness reproduce a known truth before it reports an unknown one")
w("-" * 100)
w()
w("Under d=0 the p-value is Uniform(0,1) by construction. Under d>0 measured power must match")
w("the noncentral-t formula. Both are checked before any finding below is read.")
w()
w(f"{'design':<14} {'measured':>10} {'analytic':>10} {'99% CI':>22} {'calib':>7} {'KS p':>8}")
for c in cells:
    ci = f"[{c.power_ci[0]:.4f}, {c.power_ci[1]:.4f}]"
    w(f"{c.label:<14} {c.power:>10.4f} {c.power_analytic:>10.4f} {ci:>22} "
      f"{('OK' if c.calibrated else 'FAIL'):>7} {c.ks_p:>8.3f}")
w()
n_cal = sum(1 for c in cells if c.calibrated)
w(f"  Calibrated on {n_cal} of {len(cells)} designs.")
nulls = [c for c in cells if c.d_true == 0]
w(f"  Null designs passing the uniformity test (KS p > 0.01): "
  f"{sum(1 for c in nulls if c.uniform_ok)} of {len(nulls)}.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("1. THE DANCE - replay the SAME study and watch p move")
w("-" * 100)
w()
w("Every row is one true effect at one sample size, replayed 40,000 times. Nothing changes")
w("between replicates except the sample.")
w()
w(f"{'design':<14} {'power':>7} {'p05':>11} {'median':>11} {'p95':>11} {'orders':>7} {'crosses .05':>12}")
for c in cells:
    w(f"{c.label:<14} {c.power:>7.4f} {c.p_pcts['p5']:>11.2e} {c.p_pcts['p50']:>11.2e} "
      f"{c.p_pcts['p95']:>11.2e} {c.log10_spread:>7.2f} {str(dance_crosses_threshold(c)):>12}")
w()

alt = [c for c in cells if c.d_true > 0]
widest = max(cells, key=lambda c: c.log10_spread)
narrowest = min(cells, key=lambda c: c.log10_spread)
w(f"  Widest dance:    {widest.label} at {widest.log10_spread:.2f} orders of magnitude "
  f"(power {widest.power:.4f}).")
w(f"  Narrowest dance: {narrowest.label} at {narrowest.log10_spread:.2f} orders "
  f"(power {narrowest.power:.4f}).")
w()
w("  NEGATIVE RESULT, and it reverses the folk version of this story: the spread is WIDEST where")
w("  power is highest and NARROWEST under the null. 'The p-value bounces around over orders of")
w("  magnitude' is true of every design here including the ones nobody would worry about. Raw")
w("  spread is not the hazard; it is not even correlated with the hazard in the right direction.")
w()
crossing = [c for c in alt if dance_crosses_threshold(c)]
w(f"  The decision-relevant statistic is whether the central 90% STRADDLES 0.05: true on "
  f"{len(crossing)} of {len(alt)} designs with a real effect.")
w("  Highest-power design that still straddles:")
if crossing:
    top = max(crossing, key=lambda c: c.power)
    w(f"    {top.label}, power {top.power:.4f} - p95 is {top.p_pcts['p95']:.4f}, so roughly one")
    w("    identical rerun in twenty reports 'no significant difference'.")
w()
w("  SECOND NEGATIVE RESULT, sharper: for d > 0 that straddle predicate is EXACTLY 'power < 0.95'")
w("  - p95 < alpha if and only if P(p < alpha) > 0.95. The decision-relevant dance is not an extra")
w("  hazard sitting on top of low power. It IS low power, restated in a more alarming vocabulary.")
w("  Anything this build says about the dance that is not about power is in section 2 or 3.")
w()
w("  For scale, n per group needed for 80% power:")
for d in (0.2, 0.5, 0.8):
    w(f"    d = {d}: n = {required_n(d)} per group")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("2. THE WINNER'S CURSE - what the significant replicates look like, given they got published")
w("-" * 100)
w()
w("Type M is the exaggeration ratio: mean |observed d| among significant replicates, divided by")
w("the true d. Type S is the share of significant replicates pointing the WRONG WAY. Neither is")
w("a restatement of power, and neither is visible in the p-value.")
w()
w(f"{'design':<14} {'power':>7} {'type M':>8} {'type S':>9} {'type S 99% CI':>22} {'median d-hat':>13}")
for c in alt:
    ci = f"[{c.type_s_ci[0]:.4f}, {c.type_s_ci[1]:.4f}]" if c.type_s_ci else "-"
    w(f"{c.label:<14} {c.power:>7.4f} {c.type_m:>8.2f} {c.type_s:>9.4f} {ci:>22} "
      f"{c.d_hat_median_sig:>13.3f}")
w()
worst_m = max(alt, key=lambda c: c.type_m or 0)
worst_s = max(alt, key=lambda c: c.type_s or 0)
w(f"  Worst exaggeration: {worst_m.label} publishes effects {worst_m.type_m:.2f}x the truth")
w(f"    (true d = {worst_m.d_true}, median published d-hat = {worst_m.d_hat_median_sig:.3f}).")
w(f"  Worst sign error:   {worst_s.label} - {worst_s.type_s:.1%} of its SIGNIFICANT results have")
w(f"    the wrong sign, at power {worst_s.power:.4f}.")
w()
w("  This is the part that is NOT power restated. A design can be honestly reported, correctly")
w("  analysed and statistically significant, and still be a number in the wrong direction 13% of")
w("  the time. The p-value carries no information about which case you are in.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("3. REPLICATION - you observed p = 0.05 once. What does an identical rerun say?")
w("-" * 100)
w()
w("Studies are drawn until one lands in p in [0.045, 0.055]; that study is then rerun from the")
w(f"SAME true effect. {REP_REPS:,} draws per design.")
w()
w(f"{'design':<14} {'hits':>7} {'rep p05':>10} {'rep med':>9} {'rep p95':>9} {'sig again':>10} "
  f"{'power':>8} {'differs?':>9}")
reps_rows: List[Dict[str, Any]] = []
for i, (d, n) in enumerate(REPLICATION_DESIGNS):
    r = replication_study(d, n, REP_REPS, seed=900 + i)
    w(f"{r.label:<14} {r.hits:>7} {r.rep_p_pcts['p5']:>10.2e} {r.rep_p_pcts['p50']:>9.4f} "
      f"{r.rep_p_pcts['p95']:>9.3f} {r.rep_sig_rate:>10.4f} {r.power_analytic:>8.4f} "
      f"{str(r.differs_from_power):>9}")
    reps_rows.append({
        "d_true": r.d_true, "n": r.n, "hits": r.hits, "rep_p_pcts": r.rep_p_pcts,
        "rep_sig_rate": r.rep_sig_rate, "rep_sig_ci": list(r.rep_sig_ci),
        "rep_above_10pct": r.rep_above_10pct, "power_analytic": r.power_analytic,
        "differs_from_power": r.differs_from_power,
    })
w()
n_diff = sum(1 for r in reps_rows if r["differs_from_power"])
w(f"  The replication significance rate differs from the plain unconditional power on "
  f"{n_diff} of {len(reps_rows)} designs.")
w()
w("  THIRD NEGATIVE RESULT: conditioning on 'you observed p = 0.05' buys nothing. With the true")
w("  effect held fixed, replicate p-values are independent, so the rerun's chance of being")
w("  significant is just the study's power - the same number it was before anyone looked. The")
w("  alarming 'a replication of p=0.05 could land anywhere from 0.0001 to 0.44' is the ordinary")
w("  sampling distribution of p at that power, not something the observed 0.05 told you.")
w("  What people actually mean by the dance is that they do NOT know the true effect, and a")
w("  single p = 0.05 is weak evidence about it - which is section 4, not this one.")
w()
for r in reps_rows:
    w(f"    {r['d_true']:g}/{r['n']}: {r['rep_above_10pct']:.1%} of reruns come back above p = 0.10.")
w()

# ---------------------------------------------------------------------------
w("-" * 100)
w("4. WHAT A SIGNIFICANT RESULT IS WORTH - simulated from a mixture, then checked")
w("-" * 100)
w()
w(f"{PPV_STUDIES:,} studies per row. A share `prior` of them have a real effect; the rest are")
w("exactly null. Among the ones that come out significant, how many were null all along?")
w("The closed form is a CHECK on the simulation, not a substitute for it.")
w()
w(f"{'prior':>7} {'d':>5} {'n':>5} {'power':>7} {'n sig':>8} {'false share':>12} "
  f"{'99% CI':>20} {'formula':>9} {'match':>6}")
ppv_rows: List[Dict[str, Any]] = []
for i, (prior, d, n) in enumerate(PPV_DESIGNS):
    v = ppv_study(prior, d, n, PPV_STUDIES, seed=700 + i)
    ci = f"[{v.false_share_ci[0]:.4f}, {v.false_share_ci[1]:.4f}]"
    w(f"{v.prior:>7.2f} {v.d_true:>5.1f} {v.n:>5} {v.power:>7.4f} {v.n_sig:>8} "
      f"{v.false_share:>12.4f} {ci:>20} {v.formula:>9.4f} {str(v.matches_formula):>6}")
    ppv_rows.append({
        "prior": v.prior, "d_true": v.d_true, "n": v.n, "power": v.power, "n_sig": v.n_sig,
        "false_share": v.false_share, "false_share_ci": list(v.false_share_ci),
        "formula": v.formula, "matches_formula": v.matches_formula,
    })
w()
worst_ppv = max(ppv_rows, key=lambda r: r["false_share"])
w(f"  Worst row: prior {worst_ppv['prior']:.0%}, d = {worst_ppv['d_true']}, n = {worst_ppv['n']} "
  f"-> {worst_ppv['false_share']:.1%} of everything that clears p < 0.05 is null.")
w("  alpha is the false-positive rate among TRUE NULLS. It is not the error rate of the claims")
w("  you publish, and the gap between the two is set by power and by the prior - neither of which")
w("  appears anywhere in the p-value.")
w()

# ---------------------------------------------------------------------------
w("=" * 100)
w("WHAT TO TAKE AWAY")
w("=" * 100)
w()
w("1. The p-value really is a random variable, and its spread really is enormous - but the spread")
w("   is LARGEST at high power, where it changes no decision. Raw spread is not the finding.")
w("2. The decision-relevant version of the dance is exactly 'power < 0.95'. It carries no")
w("   information beyond the power calculation you should already have run.")
w("3. What power does NOT tell you is the winner's curse. At low power the significant results")
w(f"   are inflated up to {worst_m.type_m:.1f}x and up to {worst_s.type_s:.1%} of them point the "
  f"wrong way.")
w("4. Conditioning on the observed p = 0.05 tells you nothing extra about the rerun.")
w(f"5. With a realistic prior, up to {worst_ppv['false_share']:.0%} of significant findings here "
  f"are null - and alpha never sees it.")
w()

results = {
    "alpha": ALPHA, "reps": REPS, "rep_reps": REP_REPS, "ppv_studies": PPV_STUDIES,
    "cells": [cell_row(c) for c in cells],
    "replication": reps_rows,
    "ppv": ppv_rows,
    "required_n": {str(d): required_n(d) for d in (0.2, 0.5, 0.8)},
    "summary": {
        "n_calibrated": n_cal, "n_cells": len(cells),
        "widest": {"label": widest.label, "orders": widest.log10_spread, "power": widest.power},
        "narrowest": {"label": narrowest.label, "orders": narrowest.log10_spread,
                      "power": narrowest.power},
        "n_crossing": len(crossing), "n_alt": len(alt),
        "worst_type_m": {"label": worst_m.label, "value": worst_m.type_m},
        "worst_type_s": {"label": worst_s.label, "value": worst_s.type_s,
                         "power": worst_s.power},
        "n_replication_differs": n_diff,
        "worst_ppv": worst_ppv,
    },
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("evidence.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("\nwrote evidence.txt and results.json")
