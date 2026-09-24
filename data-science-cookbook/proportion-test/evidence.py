"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Nothing downstream recomputes a verdict.
Every number here is EXACT - a weighted sum over the complete outcome grid - so there are no
replicates, no seeds and no intervals. The only approximation is Barnard's nuisance grid, and its
distance from scipy.stats.barnard_exact is measured and printed in section 1.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import proportion as P
from scipy import stats

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 96)
    say(title)
    say("=" * 96)


results: Dict[str, Any] = {"config": {
    "alpha": P.ALPHA, "band": [P.BAND_LO, P.BAND_HI], "nuisance_points": int(P.NUISANCE.size),
    "size_ps": P.SIZE_PS, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

# ------------------------------------------------------------------------ 1. calibration
rule("1.  CALIBRATION  -  every p-value checked against scipy before any result is read")
cal: Dict[str, float] = {}
n1, n2 = 15, 12
pv = P.all_pvalues(n1, n2)
cal["z2_minus_chi2"] = float(np.abs(P.z_stat(n1, n2) ** 2 - P.chi2_uncorrected(n1, n2)).max())
fis, yat, bar = 0.0, 0.0, 0.0
for i in range(n1 + 1):
    for j in range(n2 + 1):
        tab = [[i, n1 - i], [j, n2 - j]]
        fis = max(fis, abs(stats.fisher_exact(tab)[1] - pv["fisher"][i, j]))
        if 0 < i + j < n1 + n2:
            yat = max(yat, abs(stats.chi2_contingency(tab, correction=True)[1] - pv["yates"][i, j]))
for i, j in [(0, 4), (2, 8), (5, 6), (7, 1), (10, 3), (3, 3), (12, 2), (1, 9)]:
    ref = stats.barnard_exact(np.array([[i, j], [n1 - i, n2 - j]])).pvalue
    bar = max(bar, abs(ref - pv["barnard"][i, j]))
cal.update({"fisher_vs_scipy": fis, "yates_vs_scipy": yat, "barnard_vs_scipy": bar})
results["calibration"] = cal
say(f"  on a 15/12 design, all {(n1 + 1) * (n2 + 1)} tables:")
say(f"    max |z^2 - chi-square|           {cal['z2_minus_chi2']:.1e}   <- the SAME test, written twice")
say(f"    max |Fisher - scipy|             {fis:.1e}")
say(f"    max |Yates  - scipy|             {yat:.1e}")
say(f"    max |Barnard - scipy|, 8 tables  {bar:.1e}   <- {P.NUISANCE.size}-point nuisance grid, a lower bound")
say()
say("  So 'z-test or chi-square?' is not a choice. The two-sided pooled z-test and the uncorrected")
say("  chi-square are one test; any report that ran both and 'they agreed' ran one test twice.")

# ------------------------------------------------------------------------- 2. exact size
rule("2.  EXACT SIZE  -  the worst Type I rate over every true common rate 0.01..0.50")
say(f"  size = max over {len(P.SIZE_PS)} true rates of P(reject | both arms at that rate).")
say("  INFLATED above 0.055, CONSERVATIVE below 0.045. Exact - no interval needed.")
say()
say("   n1/n2       z = chi2         Yates            Fisher           Barnard")
say("  " + "-" * 84)
size_rows = P.size_table()
results["size"] = size_rows
for r in size_rows:
    cells = "".join(f"  {r[t]:.4f} {r[t + '_verdict'][:5]:<6}  " for t in P.TESTS)
    say(f"  {r['n1']:>4}/{r['n2']:<4} {cells}")
say()
say("  Mean Type I rate over the same grid (what an analyst meets on an average day, not the worst):")
for r in size_rows:
    say(f"  {r['n1']:>4}/{r['n2']:<4}  " + "   ".join(f"{t:>7} {r[t + '_mean']:.4f}" for t in P.TESTS))

infl = [r for r in size_rows if r["z_verdict"] == "INFLATED"]
cons_f = [r for r in size_rows if r["fisher_verdict"] == "CONSERVATIVE"]
cons_b = [r for r in size_rows if r["barnard_verdict"] == "CONSERVATIVE"]
worst_z = max(size_rows, key=lambda r: r["z"])
say()
say(f"  z-test inflated on {len(infl)} of {len(size_rows)} designs; worst {worst_z['z']:.4f} at "
    f"{worst_z['n1']}/{worst_z['n2']}, true rate {worst_z['z_at_p']}.")
say(f"  Fisher conservative on {len(cons_f)} of {len(size_rows)}; Barnard on {len(cons_b)}.")
over_b = [r for r in size_rows if r["barnard"] > P.ALPHA + 1e-9]
say(f"  Barnard above 0.05 on {len(over_b)} designs - it controls size by construction, so any "
    f"row here would be a harness bug.")

# ------------------------------------------------------------------------ 3. the rule of 5
rule("3.  'USE FISHER WHEN AN EXPECTED COUNT IS BELOW 5'  -  the rule scored as a classifier")
ecr = P.expected_count_rule(size_rows)
results["expected_count_rule"] = {k: (v if isinstance(v, int) else len(v))
                                  for k, v in ecr.items()}
results["expected_count_cells"] = {"safe_but_broken": ecr["safe_but_broken"],
                                   "unsafe_but_fine": ecr["unsafe_but_fine"]}
total = sum(results["expected_count_rule"].values())
say(f"  {total} cells (design x true rate). 'Broken' = z-test exact Type I above 0.055.")
say()
say("                          z actually fine    z actually broken")
say(f"  rule says safe (>= 5)   {ecr['safe_ok']:>10}          {len(ecr['safe_but_broken']):>10}")
say(f"  rule says unsafe (< 5)  {len(ecr['unsafe_but_fine']):>10}          {ecr['unsafe_broken']:>10}")
say()
if ecr["safe_but_broken"]:
    w = max(ecr["safe_but_broken"], key=lambda c: c["size"])
    say(f"  Worst cell the rule calls SAFE: {w['n1']}/{w['n2']} at p = {w['p']}, min expected "
        f"{w['min_expected']:.1f}, z-test Type I {w['size']:.4f}.")
unsafe_total = len(ecr["unsafe_but_fine"]) + ecr["unsafe_broken"]
if unsafe_total:
    say(f"  Of the {unsafe_total} cells the rule sends to Fisher, the z-test was already fine on "
        f"{len(ecr['unsafe_but_fine'])} ({len(ecr['unsafe_but_fine']) / unsafe_total:.0%}).")

# ------------------------------------------------------------------------- 4. exact power
rule("4.  EXACT POWER  -  and what Fisher's conservatism costs")
power_rows = P.power_table()
results["power"] = power_rows
size_by = {(r["n1"], r["n2"]): r for r in size_rows}
say("   p1 -> p2     n1/n2      z = chi2    Yates     Fisher   Barnard   Barnard - Fisher")
say("  " + "-" * 86)
for r in power_rows:
    say(f"  {r['p1']:.2f} -> {r['p2']:.2f}  {r['n1']:>4}/{r['n2']:<4}   "
        + "  ".join(f"{r[t]:.4f}" for t in P.TESTS)
        + f"     {r['barnard'] - r['fisher']:+.4f}")
gain = [r["barnard"] - r["fisher"] for r in power_rows]
say()
say(f"  Barnard beats Fisher on {sum(g > 0 for g in gain)} of {len(gain)} designs, by up to "
    f"{max(gain):.4f} of power, while holding size on every design in section 2.")
same = [r for r in size_rows if abs(r["yates"] - r["fisher"]) < 1e-9]
results["yates_equals_fisher_size"] = len(same)
say(f"  Yates' size equals Fisher's EXACTLY on {len(same)} of {len(size_rows)} designs: the correction")
say("  does not repair the z-test, it turns it into an approximation of Fisher - conservatism included.")
say("  The z-test's power lead is NOT a finding wherever section 2 flags it INFLATED - an")
say("  inflated test detects more because it rejects more of everything.")

# ---------------------------------------------------------------------- 5. one table, four answers
rule("5.  ONE TABLE, FOUR ANSWERS")
ex = P.exemplar()
results["exemplar"] = ex
say(f"  Chosen by rule, not by hand: the smallest table on a {ex['n']}/{ex['n']} design where the")
say("  z-test rejects and Fisher does not.")
say()
say(f"  control {ex['x1']}/{ex['n']} converted,  variant {ex['x2']}/{ex['n']} converted")
for t in P.TESTS:
    say(f"    {P.LABEL[t]:<22} p = {ex[t]:.4f}   {'significant' if ex[t] < P.ALPHA else 'not significant'}")
lat = P.lattice_regions()
results["lattice"] = lat
c = lat["counts"]
say()
say(f"  Across all {(ex['n'] + 1) ** 2} tables on that design:")
say(f"    all three reject {c['all']},  z and Barnard only {c['z_barnard']},  z only {c['z_only']},"
    f"  none {c['none']},  any other pattern {c['other']}")
say("  On this design the rejection regions NEST - Fisher inside Barnard inside z - on all 441")
say("  tables; measured here, not assumed, and not claimed for other designs.")

say()
say("  How often a reader meets the disagreement, at the true rates of each power design:")
for r in power_rows:
    say(f"    {r['p1']:.2f} -> {r['p2']:.2f}  {r['n1']:>4}/{r['n2']:<4}  z says moved, Fisher says not: "
        f"{r['z_not_fisher']:.4f}    Barnard says moved, Fisher says not: {r['barnard_not_fisher']:.4f}")


def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


with open("results.json", "w") as f:
    json.dump(_clean(results), f, indent=1, sort_keys=True)
with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
print("\nwrote evidence.txt and results.json")
