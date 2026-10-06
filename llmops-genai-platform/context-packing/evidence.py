"""Regenerate every number the README, the chart, the app and the notebook quote.

Writes evidence.txt (human) and results.json (machine). Every rate is an exact expectation over the reading
model given the declared, seeded query log; the Monte Carlo section only checks them.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np
import packing as P

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
LOG = P.make_log()
results: Dict[str, Any] = {"config": {
    "Q": P.Q, "N_CAND": P.N_CAND, "LEN_RANGE": P.LEN_RANGE, "FACTS_NEEDED": P.FACTS_NEEDED, "RANK_DECAY": P.RANK_DECAY,
    "MISS_RATE": P.MISS_RATE, "INSTRUCTION_TOKENS": P.INSTRUCTION_TOKENS, "QUESTION_TOKENS": P.QUESTION_TOKENS,
    "L0": P.L0, "LIU_POINTS": P.LIU_POINTS, "LENGTH_SLOPE": P.LENGTH_SLOPE, "COMPLY": [P.COMPLY_AT_ZERO, P.COMPLY_SLOPE,
    P.COMPLY_FLOOR], "policies": P.POLICIES, "mc_reps": P.MC_REPS, "seed": P.SEED, "numpy": np.__version__}}

rule("0.  THE LOG AND THE READING MODEL  -  synthetic, declared")
ks = sorted(P.FACTS_NEEDED)
say(f"  {P.Q:,} queries, {P.N_CAND} candidate chunks each of {P.LEN_RANGE[0]}-{P.LEN_RANGE[1]} tokens; the answer needs "
    + ", ".join(f"{k} fact{'s' if k > 1 else ''} {P.FACTS_NEEDED[k]:.0%}" for k in ks))
say(f"  a needed fact's chunk sits at retriever rank r with P(r) ~ {P.RANK_DECAY}^r, and is missed entirely {P.MISS_RATE:.0%}")
hit = {}
pr = P.RANK_DECAY ** np.arange(1, P.N_CAND + 1)
pr = pr / pr.sum()
for k in (1, 3, 5, 10, 30):
    hit[k] = (1 - P.MISS_RATE) * float(pr[:k].sum())
    say(f"    hit@{k:<2} {hit[k]:.3f}")
results["retriever_hit_at"] = hit
say(f"  read(u, L): first / middle / last = {P.LIU_POINTS} at L = {P.L0:,} tokens, minus {P.LENGTH_SLOPE} per 1,000 tokens "
    f"beyond; comply(d) = {P.COMPLY_AT_ZERO} - {P.COMPLY_SLOPE} per 1,000 tokens between the instruction and the question")
for L in (2_000, 4_000, 8_000, 12_000):
    say(f"    L = {L:>6,}: read first {P.read(0, L):.3f}  middle {P.read(0.5, L):.3f}  last {P.read(1, L):.3f}   "
        f"comply (instruction at top, d ~ L) {P.comply(L - P.INSTRUCTION_TOKENS - P.QUESTION_TOKENS):.3f}")

rule("1.  CALIBRATION  -  exact expectation vs raw Bernoulli simulation")
cal = P.calibrate(LOG)
results["calibration"] = cal
for r in cal:
    say(f"  {r['policy']:<24} {r['quantity']:<17} exact {r['exact']:.4f}   MC {r['mc']:.4f}   "
        f"{'inside' if r['inside_99'] else 'OUTSIDE'}")
say(f"  {sum(r['inside_99'] for r in cal)} of {len(cal)} inside their 99% Wilson interval at seed {P.SEED}, "
    f"{P.MC_REPS} replicates per query")

rule("2.  FOUR POLICIES, THREE NUMBERS  -  the dashboard ranks the big window first, the user ranks it last")
ev = {k: P.evaluate(p, LOG) for k, p in P.POLICIES.items()}
results["policies"] = ev
say(f"  {'policy':<24} {'context recall':>14} {'effective recall':>16} {'answer rate':>11} {'comply':>7} {'tokens':>7} {'chunks':>6}")
for k, e in ev.items():
    say(f"  {k:<24} {e['context_recall']:>14.1%} {e['effective_recall']:>16.1%} {e['answer_rate']:>11.1%} "
        f"{e['compliance']:>7.1%} {e['tokens_per_query']:>7.0f} {e['chunks_per_query']:>6.1f}")
v1, v2, v3, v4 = (ev[k] for k in P.POLICIES)
deltas = {
    "12k window: context recall": v2["context_recall"] - v1["context_recall"],
    "12k window: answer rate": v2["answer_rate"] - v1["answer_rate"],
    "reorder: context recall": v3["context_recall"] - v1["context_recall"],
    "reorder: answer rate": v3["answer_rate"] - v1["answer_rate"],
    "reorder+repeat: context recall": v4["context_recall"] - v1["context_recall"],
    "reorder+repeat: answer rate": v4["answer_rate"] - v1["answer_rate"],
}
results["deltas"] = deltas
for k, d in deltas.items():
    say(f"  {k:<32} {d * 100:+6.1f} pts")
say("  by context recall the order is: " + " > ".join(sorted(ev, key=lambda k: -ev[k]["context_recall"])))
say("  by answer rate the order is:    " + " > ".join(sorted(ev, key=lambda k: -ev[k]["answer_rate"])))

rule("3.  WHERE THE ANSWERS WENT  -  three failure classes, and the dashboard sees one of them")
say(f"  {'policy':<24} {'retrieval miss':>14} {'present, unread':>16} {'instruction ignored':>19}   (sum = 1 - answer rate)")
for k, e in ev.items():
    say(f"  {k:<24} {e['fail_missing']:>14.1%} {e['fail_unread']:>16.1%} {e['fail_ignored']:>19.1%}")
say(f"  v1: the largest failure class is a fact that WAS in the window ({v1['fail_unread']:.1%} of queries); "
    f"context recall counts it as a success")
say(f"  v2: the big window turns {v1['fail_missing'] - v2['fail_missing']:.1%} of retrieval misses into "
    f"{v2['fail_unread'] - v1['fail_unread']:.1%} more unread and {v2['fail_ignored'] - v1['fail_ignored']:.1%} more ignored")

rule("4.  THE BUDGET SWEEP  -  context recall is monotone in the budget; the answer rate is not")
budgets = [1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 6_000, 8_000, 10_000, 12_000]
sw = {"rank, top": P.sweep(budgets, LOG, "rank", "top"), "reorder, both": P.sweep(budgets, LOG, "reorder", "both")}
results["sweep"] = sw
for name, rows in sw.items():
    say(f"  {name}")
    for r in rows:
        say(f"    budget {r['budget']:>6,}: context recall {r['context_recall']:.1%}  answer rate {r['answer_rate']:.1%}  "
            f"comply {r['compliance']:.1%}  tokens {r['tokens_per_query']:.0f}")
    best = max(rows, key=lambda r: r["answer_rate"])
    say(f"    answer rate peaks at {best['budget']:,} tokens ({best['answer_rate']:.1%}); context recall never falls")
results["sweep_peak"] = {name: max(rows, key=lambda r: r["answer_rate"])["budget"] for name, rows in sw.items()}
say(f"  every candidate fits by ~{sw['rank, top'][-1]['tokens_per_query']:.0f} tokens, so 12k and larger are the same window")

rule("5.  SAME CHUNKS, DIFFERENT ORDER  -  the retrieval log cannot tell v1 from v3")
sd = P.same_set_different_order(LOG, P.POLICIES["v1 rank, 4k"], P.POLICIES["v3 reorder, 4k"])
results["same_set"] = sd
say(f"  identical kept sets on all {P.Q:,} queries: {sd['identical_sets']};  context recall gap {sd['context_recall_gap']:+.4f};  "
    f"tokens gap {sd['tokens_gap']:+.1f}")
say(f"  gold chunks in the middle third: {sd['gold_in_middle_a']:.1%} (rank order) -> {sd['gold_in_middle_b']:.1%} (reorder)")
say(f"  answer rate gap {sd['answer_rate_gap'] * 100:+.1f} pts - the published mitigation is nearly a wash here, and the "
    "next section says why")

rule("6.  BY RETRIEVER RANK  -  reorder helps rank 3+, and sends rank 2 to the far end")
br = {k: P.by_rank(P.POLICIES[k], LOG) for k in ("v1 rank, 4k", "v3 reorder, 4k", "v2 rank, 12k")}
results["by_rank"] = br
say(f"  {'rank':>4} {'n':>5}   {'v1 read':>8} {'v3 read':>8} {'v2 read':>8}   (P(kept and read) for a needed fact at this rank)")
for a, b, c in zip(*br.values()):
    say(f"  {a['rank']:>4} {a['n']:>5}   {a['read']:>8.3f} {b['read']:>8.3f} {c['read']:>8.3f}")
r2 = next(r for r in br["v3 reorder, 4k"] if r["rank"] == 2)
r2a = next(r for r in br["v1 rank, 4k"] if r["rank"] == 2)
say(f"  rank 2 is read {r2a['read']:.3f} second from the top under v1 and {r2['read']:.3f} at the far end under v3; "
    f"{sum(r['n'] for r in br['v1 rank, 4k'][:2]) / sum(r['n'] for r in br['v1 rank, 4k']):.0%} of needed facts sit at rank 1-2")

rule("7.  ORDER ALONE AT 4K  -  three orders, instruction at the top")
orders = {}
for o in ("rank", "reorder", "reverse"):
    e = P.evaluate({"budget": 4_000, "order": o, "instruction": "top"}, LOG)
    orders[o] = {"answer_rate": e["answer_rate"], "effective_recall": e["effective_recall"], "gold_in_middle_third": e["gold_in_middle_third"]}
    say(f"  {o:<8} effective recall {e['effective_recall']:.1%}  answer rate {e['answer_rate']:.1%}  gold in middle third {e['gold_in_middle_third']:.1%}")
results["orders"] = orders

results["elapsed_s"] = time.time() - t0
with open("evidence.txt", "w") as f:
    f.write("\n".join(OUT) + "\n")
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, sort_keys=True)
