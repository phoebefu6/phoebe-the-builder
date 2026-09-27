"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). The study numbers are exact (noncentral t
and a 2-D quadrature); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import paired as P

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
    "alpha": P.ALPHA, "delta_sd": P.DELTA, "rhos": list(P.RHOS), "ns": list(P.NS), "gl_nodes": P.GL_NODES,
    "mc_reps": P.MC_REPS, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

rule("1.  CALIBRATION  -  the quadrature and the vectorised tests checked three ways")
cal = P.calibrate()
results["calibration"] = cal
say(f"  quadrature at rho = 0 vs Student's noncentral t (15 designs)   max gap {cal['max_gap_vs_student_nct_at_rho0']:.1e}")
say(f"  {P.GL_NODES} vs 320 Gauss-Legendre nodes at n=20, rho=0.8        gap {cal['gap_160_vs_320_nodes']:.1e}")
say(f"  vectorised decisions vs scipy ttest_rel / ttest_ind            {cal['scipy_decision_mismatches']} "
    f"mismatches in {cal['decisions_compared']}")
mc = P.monte_carlo()
results["monte_carlo"] = mc
say(f"\n  raw bivariate-normal Monte Carlo, {P.MC_REPS:,} reps a design; exact inside the 99% Wilson interval?")
for r in mc:
    say(f"    n={r['n']:>2} rho={r['rho']:>5} delta={r['delta']:.1f}   "
        + "   ".join(f"{k}: exact {r[k]['exact']:.4f} MC {r[k]['mc']:.4f} {'inside' if r[k]['inside_99'] else 'OUTSIDE'}"
                     for k in ("paired", "independent")))
    for k in ("paired", "independent"):
        if "recheck" in r[k]:
            rc = r[k]["recheck"]
            say(f"      -> {k} re-run at {rc['reps']:,} reps, fresh seed: MC {rc['mc']:.4f} "
                f"{'inside' if rc['inside_99'] else 'OUTSIDE'} (the first miss is reported, not replaced)")

rows = P.grid()
results["grid"] = rows


def cell(n: int, rho: float) -> Dict:
    return next(r for r in rows if r["n"] == n and r["rho"] == rho)


rule(f"2.  SIZE  -  how often each analysis rejects a TRUE null (nominal {P.ALPHA})")
say("  n pairs   paired  |  independent analysis at rho =" + "".join(f"{r:>7}" for r in P.RHOS))
for n in P.NS:
    say(f"  {n:>7}   {P.ALPHA:.3f}  |  " + " " * 29 + "".join(f"{cell(n, r)['size_independent']:>7.4f}" for r in P.RHOS))
say("\n  rho > 0: conservative, so it looks 'safe'.  rho < 0: anti-conservative - the wrong analysis over-rejects.")

rule(f"3.  POWER at a true shift of {P.DELTA} SD  -  paired / independent, same data")
say("  n pairs" + "".join(f"{'rho=' + str(r):>15}" for r in P.RHOS))
for n in P.NS:
    say(f"  {n:>7}" + "".join(f"{cell(n, r)['power_paired']:>7.3f}/{cell(n, r)['power_independent']:<7.3f}" for r in P.RHOS))

rule("4.  PAIRS NEEDED FOR 80% POWER  -  the sample the wrong analysis wastes")
need = []
for rho in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95):
    npair = P.n_for_power(0.8, rho, "paired")
    nind = P.n_for_power(0.8, rho, "independent")
    need.append({"rho": rho, "n_paired": npair, "n_independent_analysis": nind, "wasted": nind - npair,
                 "ratio": nind / npair})
    say(f"  rho = {rho:>4}   paired {npair:>3}   analysed as independent {nind:>3}   wasted {nind - npair:>3} pairs"
        f"  ({nind / npair:.1f}x)")
results["n_needed"] = need
say("\n  The independent analysis never drops below ~31 pairs: it prices mean(D) at variance 2/n whatever rho is.")

rule("5.  NEGATIVE RESULT  -  below this rho, pairing COSTS power (the paired test gives up n-1 df)")
be = []
for n in (3, 5, 10, 20, 40, 80):
    r = P.break_even_rho(n)
    be.append({"n": n, "break_even_rho": r})
    say(f"  n = {n:>3} pairs   break-even rho = {r:.3f}")
results["break_even"] = be
lo = cell(5, 0.0)
say(f"\n  worst case in the grid: n=5, rho=0  paired {lo['power_paired']:.3f} vs independent {lo['power_independent']:.3f}")

rule("6.  ONE REPORT  -  12 stores, weekly sales $k, SD 4, store-to-store rho 0.85, true lift $2k")
ex = P.exemplar()
results["exemplar"] = ex
say(f"  seed {ex['seed']}: observed lift ${ex['delta']:.2f}k, sample rho {ex['rho_hat']:.3f}")
say(f"  paired t p = {ex['p_paired']:.4f}    two-sample t p = {ex['p_independent']:.4f}")
say(f"  how often the two disagree at this design (MC): only paired rejects {ex['p_only_paired']:.4f}, "
    f"only independent rejects {ex['p_only_independent']:.4f}")
say(f"  exact power here: paired {P.power_paired(12, 0.85):.3f}, independent {P.power_independent(12, 0.85):.3f}")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
