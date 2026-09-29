"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). The estimation numbers are exact (binomial
variances over fixed allocations); simple random sampling with reweighting, and regression power, are
Monte Carlo and say so.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import golden as G
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


pct = lambda x: f"{100 * x:+.1f}"  # noqa: E731
results: Dict[str, Any] = {"config": {
    "m": G.M, "months": G.MONTHS, "intents": G.INTENTS, "w0": G.W0.tolist(), "w12": G.W12.tolist(), "q": G.Q.tolist(),
    "flaky": G.FLAKY, "reg_intent": G.INTENTS[G.REG_INTENT], "reg_flip": G.REG_FLIP, "mc_reps": G.MC_REPS,
    "numpy": np.__version__, "scipy": __import__("scipy").__version__,
}}

rule("0.  THE TRAFFIC  -  synthetic, declared: 12 intents, two launch after the set is built")
say(f"  {'intent':<17} {'share m0':>8} {'share m12':>9} {'pass rate':>9}")
for i, name in enumerate(G.INTENTS):
    say(f"  {name:<17} {G.W0[i]:>8.3f} {G.W12[i]:>9.3f} {G.Q[i]:>9.2f}")
say(f"\n  true production pass rate: month 0 {G.truth(0):.4f}   month 12 {G.truth(12):.4f}")

rule("1.  CALIBRATION  -  exact moments vs raw simulation of drawing and labelling the set")
cal = G.calibrate()
results["calibration"] = cal
for r in cal["moments"]:
    say(f"  {r['design']:<12} month {r['month']:>2} {'reweighted' if r['reweight'] else 'raw':<10}  bias exact {r['exact_bias']:+.5f} "
        f"MC {r['mc_bias']:+.5f} (z {r['bias_z']:+.2f})   sd exact {r['exact_sd']:.5f} MC {r['mc_sd']:.5f} (ratio {r['sd_ratio']:.3f})")
say(f"  one-sided exact McNemar vs scipy.stats.binomtest: {cal['mcnemar_vs_binomtest_mismatches']} mismatches in {cal['mcnemar_cases']}")

rule(f"2.  THE HEADLINE  -  error of the golden-set pass rate vs production, m = {G.M}, in percentage points")
say(f"  {'design':<13} {'allocation':<42} {'m0 raw':>14} {'m12 raw':>14} {'m12 reweighted':>16}")
head = []
for d in G.DESIGNS:
    alloc = "multinomial on month-0 traffic" if d == "random" else " ".join(map(str, G.allocate(d)[:G.EXISTING]))
    row = {"design": d, "allocation": None if d == "random" else G.allocate(d).tolist()}
    for key, (t, rw) in {"m0_raw": (0, False), "m0_rew": (0, True), "m12_raw": (12, False), "m12_rew": (12, True)}.items():
        row[key] = G.headline(d, t, rw)
    head.append(row)
    say(f"  {d:<13} {alloc:<42} " + " ".join(
        f"{pct(row[k]['bias']):>6} +/-{100 * row[k]['sd']:.1f}" for k in ("m0_raw", "m12_raw")) +
        f"   {pct(row['m12_rew']['bias']):>6} +/-{100 * row['m12_rew']['sd']:.1f}")
results["headline"] = head
say("  (bias +/- SD. random + reweighted is Monte Carlo; every other cell is exact)")
eq0 = next(r for r in head if r["design"] == "equal")
say(f"\n  equal allocation, month 0: raw {pct(eq0['m0_raw']['bias'])} pts, reweighted by the traffic mix "
    f"{pct(eq0['m0_rew']['bias'])} pts, RMSE {100 * eq0['m0_rew']['rmse']:.2f} vs proportional "
    f"{100 * next(r for r in head if r['design'] == 'proportional')['m0_raw']['rmse']:.2f}")

rule("3.  STALENESS  -  a frozen proportional set, month by month")
n_prop = G.allocate("proportional")
stale = []
say(f"  {'month':>5}  {'truth':>7}  {'raw bias':>9}  {'reweighted bias':>15}  {'traffic with no cases':>21}")
for t in range(0, G.MONTHS + 1, 2):
    raw, rew, gap = G.moments(n_prop, t, False), G.moments(n_prop, t, True), G.coverage_gap(t)
    stale.append({"month": t, "truth": G.truth(t), "raw_bias": raw["bias"], "rew_bias": rew["bias"], **gap})
    say(f"  {t:>5}  {G.truth(t):>7.4f}  {pct(raw['bias']):>9}  {pct(rew['bias']):>15}  {gap['uncovered_share']:>21.1%}")
results["staleness"] = stale
s12 = stale[-1]
say(f"\n  month 12: the dashboard reads {G.truth(12) + s12['raw_bias']:.3f}, production is {G.truth(12):.3f}. "
    f"Reweighting (free) removes {pct(s12['raw_bias'] - s12['rew_bias'])} of {pct(s12['raw_bias'])} pts;")
say(f"  the other {pct(s12['rew_bias'])} is the {s12['uncovered_share']:.0%} of traffic from intents that did not exist at month 0.")

rule("4.  NEGATIVE RESULT  -  a bigger frozen set does not fix staleness")
big = []
for m in (60, 240, 960, 2400):
    a, b = G.headline("proportional", 12, False, m), G.headline("proportional", 12, True, m)
    big.append({"m": m, "raw": a, "rew": b})
    say(f"  m = {m:>5}   raw bias {pct(a['bias'])} (SD {100 * a['sd']:.2f})   reweighted bias {pct(b['bias'])} (SD {100 * b['sd']:.2f})")
results["bigger_set"] = big
say("  The SD shrinks with m. The bias does not move: 10x the labels buys a more precise wrong number.")

rule("5.  THE REFRESH  -  add cases for the new intents only, and reweight")
ref = []
for add in (0, 5, 10, 20, 40):
    n = G.refresh(n_prop, add)
    a, b = G.moments(n, 12, False), G.moments(n, 12, True)
    ref.append({"add_per_new_intent": add, "labels_added": 2 * add, "raw": a, "rew": b})
    say(f"  +{add:>2} per new intent ({2 * add:>2} labels)   raw bias {pct(a['bias'])}   reweighted bias {pct(b['bias'])}"
        f"  RMSE {100 * b['rmse']:.2f}")
results["refresh"] = ref
say(f"  Month-0 RMSE was {100 * G.headline('proportional', 0, False)['rmse']:.2f}. Raw WITHOUT reweighting overshoots once the"
    " new intents are over-represented: the refresh needs the reweighting too.")

rule(f"6.  CATCHING A REGRESSION  -  v2 breaks {G.REG_FLIP:.0%} of the passing '{G.INTENTS[G.REG_INTENT]}' cases "
     f"({G.Q[G.REG_INTENT]:.2f} -> {G.Q[G.REG_INTENT] * (1 - G.REG_FLIP):.2f})")
say(f"  {G.FLAKY:.0%} of cases are flaky (coin flip every run). Monte Carlo, {G.MC_REPS:,} reps. Production pass rate "
    f"falls {100 * G.W12[G.REG_INTENT] * G.Q[G.REG_INTENT] * G.REG_FLIP:.1f} pts at month 12.")
say(f"  {'design':<13} {'cases there':>11}   {'aggregate McNemar':>17}   {'per-intent (Bonferroni)':>23}   {'false alarm agg / per':>21}")
reg = []
for d in ("proportional", "sqrt", "equal"):
    n = G.allocate(d)
    p, f = G.regression_power(n), G.regression_power(n, flip=0)
    reg.append({"design": d, **p, "false_aggregate": f["aggregate"], "false_per_intent": f["per_intent"]})
    say(f"  {d:<13} {p['cases_in_regressed']:>11}   {p['aggregate']:>17.3f}   {p['per_intent']:>23.3f}   "
        f"{f['aggregate']:>10.3f} / {f['per_intent']:.3f}")
results["regression"] = reg
say("\n  The set built to estimate the average (proportional) misses a halved intent almost 9 times in 10.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
