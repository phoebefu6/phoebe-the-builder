"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Nothing downstream recomputes a verdict.
The study numbers are exact integrals; the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import equiv as E
import numpy as np

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 96)
    say(title)
    say("=" * 96)


def cell(rows: List[Dict], n: int, f: float) -> Dict:
    return next(r for r in rows if r["n"] == n and r["delta_frac"] == f)


results: Dict[str, Any] = {"config": {
    "alpha": E.ALPHA, "margin_sd": E.MARGIN, "ns": list(E.NS), "delta_fracs": list(E.DELTAS),
    "mc_reps": E.MC_REPS, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

# ------------------------------------------------------------------------ 1. calibration
rule("1.  CALIBRATION  -  the exact integral checked against three independent things")
cal = E.calibrate()
results["calibration"] = cal
say(f"  vectorised decisions vs scipy ttest_ind (one- and two-sided)  {cal['scipy_decision_mismatches']} "
    f"mismatches in {cal['decisions_compared']}")
say(f"  TOST verdict vs 'the 90% CI sits inside the margin'            {cal['tost_vs_ci90_mismatches']} mismatches in 300")
say(f"  integral P(not significant) vs noncentral-t closed form       max gap {cal['max_gap_vs_noncentral_t']:.1e}")
say(f"  four outcome probabilities sum to 1                           max gap {cal['max_sum_gap']:.1e}")
mc = E.monte_carlo()
results["monte_carlo"] = mc
say(f"\n  raw-data Monte Carlo, {E.MC_REPS:,} reps a design; every outcome inside its 99% Wilson interval?")
for r in mc:
    say(f"    n={r['n']:>3} delta={r['delta_frac']:.1f}*margin  exact equiv {r['exact']['equiv_only'] + r['exact']['both']:.4f}"
        f"  MC {r['mc']['equiv_only'] + r['mc']['both']:.4f}   {'inside' if r['inside_99'] else 'OUTSIDE'}")

# ------------------------------------------------------------------------ 2. the grid
rows = E.grid()
results["grid"] = rows
rule(f"2.  'NO SIGNIFICANT DIFFERENCE'  -  how often the t-test says it, margin = {E.MARGIN} SD, alpha = {E.ALPHA}")
say("   n/group" + "".join(f"   delta={f:.1f}M" for f in E.DELTAS))
for n in E.NS:
    say(f"  {n:>7}" + "".join(f"{cell(rows, n, f)['not_significant']:>13.4f}" for f in E.DELTAS))
say("\n  delta = 1.0M is a difference EXACTLY as large as the one the business said matters.")

rule("3.  TOST  -  P(declared equivalent). delta = 1.0M is the size (<= alpha); delta = 0 is the power")
say("   n/group" + "".join(f"   delta={f:.1f}M" for f in E.DELTAS))
for n in E.NS:
    say(f"  {n:>7}" + "".join(f"{cell(rows, n, f)['equivalent']:>13.4f}" for f in E.DELTAS))

# ------------------------------------------------------------------------ 4. evidence ratio
rule("4.  HOW MUCH EVIDENCE IS EACH REPORT?  likelihood ratio  P(report | delta=0) / P(report | delta=M)")
say("   n/group   'not significant'   'equivalent (TOST)'")
lr = []
for n in E.NS:
    a0, a1 = cell(rows, n, 0.0), cell(rows, n, 1.0)
    lr_ns = a0["not_significant"] / a1["not_significant"] if a1["not_significant"] > 0 else float("inf")
    lr_eq = a0["equivalent"] / a1["equivalent"]
    lr.append({"n": n, "lr_not_significant": lr_ns, "lr_equivalent": lr_eq})
    say(f"  {n:>7} {lr_ns:>19.2f} {lr_eq:>20.2f}")
results["likelihood_ratio"] = lr
say("\n  TOST's ratio is capped near power/alpha = 20; the t-test's is unbounded but only once n is large.")

# ------------------------------------------------------------------------ 5. n needed
rule("5.  THE n AT WHICH ABSENCE OF EVIDENCE BECOMES EVIDENCE OF ABSENCE  (80% chance TOST says 'equivalent')")
need = []
for f in (0.0, 0.2, 0.5):
    n_exact = E.n_for_power(0.8, f * E.MARGIN)
    n_apx = E.n_normal_approx(0.8, f)
    need.append({"delta_frac": f, "n_exact": n_exact, "n_normal_approx": n_apx})
    say(f"  true delta = {f:.1f}*margin   exact n/group = {n_exact:>4}   normal shortcut = {n_apx:6.1f}")
results["n_needed"] = need

# ------------------------------------------------------------------------ 6. the four outcomes
rule("6.  FOUR OUTCOMES, NOT TWO  -  at true delta = 0.2*margin (a real but immaterial difference)")
say("   n/group" + "".join(f"{E.LABEL[o]:>28}" for o in E.OUTCOMES))
for n in E.NS:
    c = cell(rows, n, 0.2)
    say(f"  {n:>7}" + "".join(f"{c[o]:>28.4f}" for o in E.OUTCOMES))

# ------------------------------------------------------------------------ 7. exemplar
rule("7.  ONE REPORT  -  basket value, SD $20, margin $10, TRUE difference $10, n = 20 a group")
ex = E.exemplar()
results["exemplar"] = ex
say(f"  seed {ex['seed']}: observed difference ${ex['delta']:.2f}, t-test p = {ex['p_diff']:.4f}  "
    "-> 'no significant difference'")
say(f"  TOST p = {ex['p_tost']:.4f}, 90% CI (${ex['ci90'][0]:.2f}, ${ex['ci90'][1]:.2f}) vs margin +-${ex['margin']:.0f}"
    f"  -> {ex['outcome']}")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
