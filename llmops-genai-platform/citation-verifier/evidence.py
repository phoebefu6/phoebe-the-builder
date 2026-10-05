"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every checker rate is exact (enumeration of every claim
the declared generator can emit); the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import citation as T
import numpy as np

OUT: List[str] = []
TAUS = [round(0.3 + 0.05 * k, 2) for k in range(15)]


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 100)
    say(title)
    say("=" * 100)


t0 = time.time()
results: Dict[str, Any] = {"config": {"tau": T.TAU, "models": T.MODELS, "edits": T.EDITS, "cites": T.CITES,
                                      "checkers": T.CHECKERS, "mc_reps": T.MC_REPS, "numpy": np.__version__}}

rule("0.  THE CORPUS AND THE GENERATOR  -  synthetic, declared")
for i, p in enumerate(T.PASSAGES):
    say(f"  [{i + 1:>2}] ({T.DOC_OF[i]:<8}) {p}   neighbour -> [{T.sibling(i) + 1}]")
say("  a claim = one true fact, one edit, one citation choice, verbatim or paraphrased")
say("  SUPPORTED = the cited passage states the same (entity, attribute, value, scope, polarity)")
for name, m in T.MODELS.items():
    rows = T.enumerate_claims(m)
    say(f"  {name:<20} edit {dict(zip(T.EDITS, m['edit']))}")
    say(f"  {'':<20} cite {dict(zip(T.CITES, m['cite']))}  paraphrase {m['para']}  "
        f"({len(rows)} enumerated claims, mass {sum(r['p'] for r in rows):.12f})")
say(f"  lexical threshold tau = {T.TAU} (content-word coverage of the claim in the passage); the sweep below varies it")

rule("1.  CALIBRATION  -  exact enumeration vs raw simulation")
cal = T.calibrate()
results["calibration"] = cal
for r in cal:
    say(f"  {r['model']:<20} {r['quantity']:<32} exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")
say(f"  {sum(r['inside_99'] for r in cal)} of {len(cal)} inside their 99% Wilson interval at seed {T.SEED}")

ev = T.evaluate()
results["models"] = ev
rule("2.  WHAT EACH CHECKER REPORTS, AND HOW OFTEN IT IS WRONG")
for name, s in ev.items():
    say(f"\n  {name.upper()}   supported by its citation {s['p_supported']:.1%}   claim true {s['p_true']:.1%}   "
        f"carries a citation {s['p_cited']:.1%}")
    say(f"    {'checker':<32} {'reports':>8} {'passes bad':>11} {'fails good':>11} {'bad among passes':>17}")
    for k, v in s["scorers"].items():
        say(f"    {k:<32} {v['reported']:>8.1%} {v['false_pass']:>11.1%} {v['false_fail']:>11.1%} "
            f"{v['bad_among_passes']:>17.1%}")
    say("    what the sharpest checker still passes: " + ", ".join(
        f"{c} {p:.1%}" for c, p in s["scorers"]["cited_lexical+numbers+negation"]["passed_bad_by_class"].items()))

rule("3.  THE INSTRUCTION CHANGE  -  'always cite your sources', as each checker sees it")
v1, v2 = ev["v1 cites when sure"], ev["v2 always cite"]
deltas = {"truly supported": v2["p_supported"] - v1["p_supported"]}
deltas.update({k: v2["scorers"][k]["reported"] - v1["scorers"][k]["reported"] for k in T.CHECKERS})
results["deltas"] = deltas
for k, d in deltas.items():
    say(f"  {k:<32} {d * 100:+6.1f} pts")
say(f"  citations {v1['p_cited']:.0%} -> {v2['p_cited']:.0%}; true claims unchanged at {v1['p_true']:.1%}. The new citations")
say("  point at the neighbouring passage (same attribute, other plan), which shares most of the claim's words.")

rule("4.  THE THRESHOLD CANNOT SEPARATE A POLARITY FLIP FROM AN HONEST PARAPHRASE")
sweep = T.threshold_sweep(T.MODELS["v1 cites when sure"], TAUS)
order = T.flip_vs_paraphrase_order(T.MODELS["v1 cites when sure"])
results["sweep"] = sweep
results["order"] = order
say(f"  {'tau':>5} {'fails a supported paraphrase':>30} {'passes a polarity flip':>24}")
for r in sweep:
    say(f"  {r['tau']:>5.2f} {r['fails_supported_paraphrase']:>30.1%} {r['passes_polarity_flip']:>24.1%}")
say(f"  a verbatim flip ('does not include' for 'includes') has coverage {order['verbatim_flip_coverage_min']:.2f}: "
    "the stopword list drops 'not', so it is lexically identical to its source.")
say(f"  across every (flip, supported paraphrase) pair: flip scores above {order['flip_above']:.1%}, level "
    f"{order['tied']:.1%}, below {order['flip_below']:.1%}.")

results["elapsed_s"] = time.time() - t0
with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
