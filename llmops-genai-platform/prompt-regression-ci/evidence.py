"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every gate number is exact (per-case binomials,
Poisson-binomial counts); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import gate as G
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


pi_a = G.held_out_set()
results: Dict[str, Any] = {"config": {
    "n": G.N, "stable_pass": G.N_STABLE_PASS, "stable_fail": G.N_STABLE_FAIL, "flaky": G.N_FLAKY,
    "stable_p": G.STABLE_P, "q_runs": G.Q_RUNS, "n_break": G.N_BREAK, "alpha_gate": G.ALPHA_GATE,
    "prs_per_month": G.PRS_PER_MONTH, "mc_reps": G.MC_REPS, "flaky_pi": pi_a[-G.N_FLAKY:].tolist(),
    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

rule("0.  THE HELD-OUT SET  -  synthetic, declared")
say(f"  {G.N} cases: {G.N_STABLE_PASS} stable pass (pi = {G.STABLE_P}), {G.N_STABLE_FAIL} stable fail, "
    f"{G.N_FLAKY} flaky (pi ~ Beta(2, 2), mean {pi_a[-G.N_FLAKY:].mean():.3f})")
say(f"  scenarios: a no-op PR; a side effect that breaks {G.N_BREAK} stable-pass cases hard (-> {1 - G.STABLE_P}),")
say(f"  intermittently (-> 0.5), or breaks {G.N_BREAK} of the flaky cases hard")

rule("1.  CALIBRATION  -  exact red rates vs raw simulated runs")
cal = G.calibrate()
results["calibration"] = cal
say(f"  Poisson-binomial convolution vs scipy binom on 50 equal cases: max gap {cal['poisson_binomial_vs_binom_gap']:.1e}")
for r in cal["red_rates"]:
    say(f"  {r['rule']:<27} {r['scenario']:<20} T={r['T']:>2}   exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")

ev = G.evaluate(pi_a)
results["rules"] = ev

rule("2.  A NO-OP PR  -  nothing changed, what does CI say?")
say(f"  {'rule':<27} {'phantom breaks':>14} {'P(red), zero tolerance':>23} {'calibrated T':>12} {'P(red) at T':>11}"
    f" {'months with a false red at T':>29}")
for r in ev:
    say(f"  {r['rule']:<27} {r['noop_expected_flags']:>14.2f} {r['noop_red_at_T1']:>23.3f} {r['T']:>12} "
        f"{r['noop_red_at_T']:>11.3f} {r['month_any_false_red_T']:>29.3f}")
say(f"\n  'zero tolerance' = red on any broken case. At {G.PRS_PER_MONTH} PRs a month, a gate that is red 4% of the time on")
say("  a no-op still goes red for nothing in most months. That is the calibrated version.")

for sc in G.SCENARIOS[1:]:
    rule(f"3.  {sc.upper()}  -  {G.N_BREAK} cases really broken; gate at its calibrated T")
    say(f"  {'rule':<27} {'P(red)':>7} {'recall per case':>16} {'report precision':>17} {'cases listed':>13} {'calls/PR':>9}")
    for r in ev:
        x = r[sc]
        say(f"  {r['rule']:<27} {x['red_at_T']:>7.3f} {x['recall']:>16.3f} {x['precision']:>17.3f} "
            f"{x['expected_flags']:>13.2f} {r['calls_per_pr']:>9,}")

rule("4.  NEGATIVE RESULT  -  quarantine buys a quiet gate by going blind on the flaky cases")
qn = next(r for r in ev if r["rule"] == "quarantine + majority k=3")
m3 = next(r for r in ev if r["rule"] == "majority k=3")
say(f"  quarantine drops {G.N - qn['gated_cases']:.1f} of {G.N} cases on average (the flaky ones, plus stable ones that fail 1 run in 200)")
say(f"  hard break on stable cases: P(red) {m3['hard break']['red_at_T']:.3f} -> {qn['hard break']['red_at_T']:.3f}")
say(f"  hard break on FLAKY cases:  P(red) {m3['break on flaky cases']['red_at_T']:.3f} -> {qn['break on flaky cases']['red_at_T']:.3f}")
say("  Whatever the quarantined cases test, the gate no longer tests it - and they are the cases most likely to move.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
