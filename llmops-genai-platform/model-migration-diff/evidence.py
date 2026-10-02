"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every gate number is exact (2D convolution over
cases); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import migrate as M
import numpy as np

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 100)
    say(title)
    say("=" * 100)


pa, intent, kind = M.eval_set()
results: Dict[str, Any] = {"config": {
    "n": M.N, "intents": dict(M.INTENTS), "flaky_share": M.FLAKY_SHARE, "fail_share": M.FAIL_SHARE,
    "stable_p": M.STABLE_P, "n_swap": M.N_SWAP, "concentrate_in": M.CONCENTRATE_IN, "alpha": M.ALPHA,
    "margin": M.MARGIN, "mc_reps": M.MC_REPS, "kinds": np.bincount(kind).tolist(),
    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

rule("0.  THE EVAL SET  -  synthetic, declared")
say(f"  {M.N} cases in {len(M.INTENTS)} intents: " + ", ".join(f"{n} {s}" for n, s in M.INTENTS))
say(f"  per intent: {M.FLAKY_SHARE:.0%} flaky (pass prob ~ Beta(2, 2)), {M.FAIL_SHARE:.0%} stable fail, rest stable pass "
    f"(pi = {M.STABLE_P})  ->  {np.bincount(kind).tolist()} pass/fail/flaky")
say(f"  migrations: break {M.N_SWAP} stable passes (all in '{M.CONCENTRATE_IN}', or spread) and fix {M.N_SWAP} stable fails"
    " elsewhere; or break 20 spread with no fixes")

rule("1.  CALIBRATION  -  exact gate rates vs raw simulated runs")
cal = M.calibrate()
results["calibration"] = cal
say(f"  2D convolution vs scipy binom (closed-form margin): max gap {cal['conv_vs_binom_gap']:.1e}; "
    f"mass off the fixed = 0 row {cal['fixed_mass_off_zero']:.1e}")
for r in cal["rates"]:
    say(f"  {r['scenario']:<24} {r['gate']:<15} exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")

ev = M.evaluate()
results.update(ev)
rule("2.  FOUR MIGRATIONS  -  what the score says, what changed, and which gate notices")
say(f"  noise churn (A vs A run again): {ev['noise_churn']:.2f} cases expected")
for r in ev["scenarios"]:
    g = r["gates"]
    say(f"\n  {r['scenario'].upper()}   cases really changed: {r['cases_changed']}")
    say(f"    score {r['score_a']:.1%} -> {r['score_b']:.1%}   expected broke {r['expected_broke']:.2f}  fixed "
        f"{r['expected_fixed']:.2f}  churn {r['expected_churn']:.2f}")
    worst = min(r["intent_scores"].items(), key=lambda kv: kv[1][1] - kv[1][0])
    say(f"    worst intent: {worst[0]} {worst[1][0]:.1%} -> {worst[1][1]:.1%}")
    say("    " + "   ".join(f"{k} {g[k]:.3f}" for k in M.GATES) + "   (tost = P(certified equivalent))")

rule("3.  SWEEP  -  a net-zero spread migration of k breaks + k fixes")
sw = M.sweep()
results["sweep"] = sw
say(f"  {'k':>3} {'changed':>8} " + " ".join(f"{g:>15}" for g in M.GATES))
for r in sw:
    say(f"  {r['swap']:>3} {r['churn_cases']:>8} " + " ".join(f"{r[g]:>15.3f}" for g in M.GATES))
say("  McNemar gets QUIETER as the swap grows: it compares broke with fixed, and more of both makes a")
say("  balanced split more likely, not less.")

rule("4.  NEGATIVE RESULT  -  slicing by intent does not detect a spread regression")
pr = next(r for r in ev["scenarios"] if r["scenario"] == "plain regression")["gates"]
say(f"  plain regression, 20 breaks spread over 8 intents: whole-set McNemar {pr['mcnemar']:.3f}, per-intent "
    f"Bonferroni {pr['per-intent']:.3f}")
say("  Each intent sees 2-3 breaks; an exact test at 0.05/8 needs about 8. Slice to diagnose, not to detect.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
