"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every scorer number is exact (enumeration of every
trajectory the declared agent can produce); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import trajectory as T

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 100)
    say(title)
    say("=" * 100)


results: Dict[str, Any] = {"config": {"p_elig": T.P_ELIG, "agents": T.AGENTS, "plans": T.PLANS,
                                      "scorers": T.SCORERS, "mc_reps": T.MC_REPS, "numpy": np.__version__}}

rule("0.  THE TASK  -  synthetic, declared")
say(f"  refund request, eligible with p = {T.P_ELIG}. reference path: lookup_order > check_policy > "
    "issue_refund (or decline)")
say("  invariants of a valid path: " + ", ".join(T.INVARIANTS))
for name, a in T.AGENTS.items():
    runs = T.enumerate_runs(a)
    say(f"  {name:<15} plan {dict(zip(T.PLANS, a['plan']))}  kb {a['kb']}  rash {a['rash']}  dup {a['dup']}  "
        f"badarg {a['badarg']}  ({len(runs)} enumerated runs, mass {sum(r['p'] for r in runs):.12f})")

rule("1.  CALIBRATION  -  exact enumeration vs raw step-by-step simulation")
cal = T.calibrate()
results["calibration"] = cal
for r in cal:
    say(f"  {r['agent']:<15} {r['quantity']:<17} exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")
say(f"  {sum(r['inside_99'] for r in cal)} of {len(cal)} inside their 99% Wilson interval")
a = T.AGENTS["v1 careful"]
strict = T.summarise(T.enumerate_runs(a))["scorers"]["strict"]["reported"]
# closed form: the clean path, plus a rash agent that guessed "decline" on an ineligible task - it read
# afterwards and issued nothing, so its trace is the reference path exactly. No invariant can see that.
clean = a["plan"][0] * (1 - a["kb"]) * (1 - a["rash"]) * (1 - a["misread"]) * (T.P_ELIG * (1 - a["dup"]) + 1 - T.P_ELIG)
hidden_rash = a["plan"][0] * (1 - a["kb"]) * a["rash"] * (1 - T.P_ELIG) * (1 - T.P_ELIG)
results["closed_form_gap"] = abs(strict - clean - hidden_rash)
say(f"  v1 strict, closed form {clean + hidden_rash:.9f} vs enumeration {strict:.9f}: gap {results['closed_form_gap']:.1e}")
se = (strict * (1 - strict) / T.MC_REPS) ** 0.5
zs = [(T.simulate(a, T.MC_REPS, seed=s)["strict"] - strict) / se for s in (1, 2, 3, 4)]
results["reseed_z"] = zs
say("  the one OUTSIDE cell above, reseeded 4 times: z = " + ", ".join(f"{z:+.2f}" for z in zs)
    + "  (12 cells at 99% expect 0.12 misses; seed 189 drew one)")

ev = T.evaluate()
results["agents"] = ev
rule("2.  WHAT EACH SCORER REPORTS, AND HOW OFTEN IT IS WRONG")
for name, s in ev.items():
    say(f"\n  {name.upper()}   truly valid runs {s['p_good']:.1%}   answer right {s['p_correct']:.1%}   "
        f"tool calls {s['calls']:.2f}")
    say(f"    {'scorer':<17} {'reports':>8} {'passes bad':>11} {'fails good':>11} {'bad among passes':>17}")
    for k, v in s["scorers"].items():
        say(f"    {k:<17} {v['reported']:>8.1%} {v['false_pass']:>11.1%} {v['false_fail']:>11.1%} "
            f"{v['bad_among_passes']:>17.1%}")
    say("    right answer, broken path: " + ", ".join(f"{k} {v:.1%}" for k, v in s["hidden_by_outcome"].items()))

rule("3.  THE PROMPT CHANGE  -  v1 -> v2 as each scorer sees it")
v1, v2 = ev["v1 careful"], ev["v2 fewer calls"]
deltas = {"truly valid": v2["p_good"] - v1["p_good"]}
deltas.update({k: v2["scorers"][k]["reported"] - v1["scorers"][k]["reported"] for k in T.SCORERS})
results["deltas"] = deltas
for k, d in deltas.items():
    say(f"  {k:<17} {d * 100:+6.1f} pts")
say(f"  tool calls {v1['calls']:.2f} -> {v2['calls']:.2f}. The strict match barely moves: v2 also dropped the")
say("  harmless variation (extra kb search, reads in the other order) that strict was failing all along.")

with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
