"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every rate is closed-form under the declared model;
the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np
import staleness as T

OUT: List[str] = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line)


def rule(title: str) -> None:
    say()
    say("=" * 100)
    say(title)
    say("=" * 100)


t0 = time.time()
results: Dict[str, Any] = {"config": {"classes": T.CLASSES, "policies": T.POLICIES, "mc_reps": T.MC_REPS,
                                      "seed": T.SEED, "numpy": np.__version__}}

rule("0.  THE CORPUS AND THE MODEL  -  synthetic, declared")
for c in T.CLASSES:
    lam, mu = T.rates(c)
    say(f"  {c['name']:<5} {c['docs']:>4} docs  edited every {c['change_every']:>5.0f} d  fact-bearing {c['fact_share']:.0%}  "
        f"query share {c['query_share']:.0%}   lambda {lam:.5f}/d  mu {mu:.5f}/d   ({c['what']})")
say("  stale(mu, T) = 1 - (1 - exp(-mu T)) / (mu T): a doc reindexed every T days, served at a uniform phase")
say("  hash_freshness: doc-weighted, any edit.  fact_freshness: doc-weighted, fact edits.  answer_staleness: query-weighted")

rule("1.  CALIBRATION  -  closed form vs raw simulation")
cal = T.calibrate()
results["calibration"] = cal
for r in cal:
    say(f"  {r['policy']:<18} {r['quantity']:<20} exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")
say(f"  {sum(r['inside_99'] for r in cal)} of {len(cal)} inside their 99% Wilson interval at seed {T.SEED}, "
    f"{T.MC_REPS:,} reps per cell")

rule("2.  THREE POLICIES, THREE NUMBERS  -  the dashboard ranks them one way, the user the other")
ev = {k: T.evaluate(p) for k, p in T.POLICIES.items()}
results["policies"] = ev
say(f"  {'policy':<18} {'hash fresh':>10} {'fact fresh':>10} {'answers stale':>13} {'reindex/day':>11}")
for k, e in ev.items():
    say(f"  {k:<18} {e['hash_freshness']:>10.1%} {e['fact_freshness']:>10.1%} {e['answer_staleness']:>13.1%} "
        f"{e['reindex_per_day']:>11.1f}")
for k, e in ev.items():
    say(f"  {k:<18} per class fact-stale: " + ", ".join(f"{c} {v['fact_stale']:.1%} (T={v['period']:.0f})"
                                                        for c, v in e["per_class"].items()))
v1, wk, ti = ev["v1 monthly full"], ev["v2a weekly full"], ev["v2b tiered 1/7/90"]
deltas = {
    "weekly: hash freshness": wk["hash_freshness"] - v1["hash_freshness"],
    "weekly: answer staleness": wk["answer_staleness"] - v1["answer_staleness"],
    "tiered: hash freshness": ti["hash_freshness"] - v1["hash_freshness"],
    "tiered: answer staleness": ti["answer_staleness"] - v1["answer_staleness"],
}
results["deltas"] = deltas
results["ratios"] = {"stale_answers_weekly_over_tiered": wk["answer_staleness"] / ti["answer_staleness"],
                     "cost_weekly_over_tiered": wk["reindex_per_day"] / ti["reindex_per_day"],
                     "v1_answer_over_hash_stale": v1["answer_staleness"] / (1 - v1["hash_freshness"])}
for k, d in deltas.items():
    say(f"  {k:<28} {d * 100:+6.1f} pts")
say(f"  weekly full serves {results['ratios']['stale_answers_weekly_over_tiered']:.1f}x the stale answers of tiered at "
    f"{results['ratios']['cost_weekly_over_tiered']:.1f}x the reindex cost, and scores higher on index freshness")

rule("3.  INSIDE THE CYCLE  -  the average is not what the user on day 29 gets")
cyc = T.cycle(T.POLICIES["v1 monthly full"])
results["cycle"] = cyc
for d in (0, 1, 7, 14, 21, 29, 30):
    say(f"  day {d:>2}: {cyc[d]['answer_staleness']:.1%} of answers stale")
say(f"  phase-averaged (what a monthly metric reports): {v1['answer_staleness']:.1%}")

rule("4.  ADDING DOCUMENTS NOBODY ASKS ABOUT  -  the metric improves, the answers do not")
arch = {}
for extra in (0, 300, 1000):
    cls, pol = T.with_archive(extra, T.POLICIES["v1 monthly full"])
    e = T.evaluate(pol, cls)
    arch[extra] = {"hash_freshness": e["hash_freshness"], "answer_staleness": e["answer_staleness"], "docs": e["docs"]}
    say(f"  +{extra:>4} archive docs ({e['docs']:>4} total): hash freshness {e['hash_freshness']:.1%}   "
        f"answers stale {e['answer_staleness']:.1%}")
results["archive"] = arch

rule("5.  THE AUDIT SAMPLE  -  two unbiased estimators of two different numbers")
aud = {s: T.audit_sample(T.POLICIES["v1 monthly full"], s) for s in ("docs", "queries")}
results["audit"] = aud
say(f"  sample documents uniformly:        expected stale share {aud['docs']:.1%}  (= 1 - fact freshness)")
say(f"  sample from the query log:         expected stale share {aud['queries']:.1%}  (= answer staleness)")

rule("6.  NEGATIVE RESULT  -  retrieval similarity cannot see staleness")
sg = T.similarity_gap()
results["similarity"] = sg
for r in sg["rows"][:4]:
    say(f"  {r['query']:<52} stale {r['stale_sim']:.3f}  fresh {r['fresh_sim']:.3f}  gap {r['gap']:+.3f}")
say(f"  ... {sg['n']} facts; stale chunk at least as similar as the fresh one in {sg['stale_at_least_as_similar']} of "
    f"{sg['n']}; max |gap| {sg['max_abs_gap']:.3f}")
say(f"  a similarity gate that keeps every fresh chunk passes {sg['stale_passed_by_gate_keeping_all_fresh']:.0%} of "
    "stale chunks: the query asks about the attribute and never carries the value")

results["elapsed_s"] = time.time() - t0
with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
