"""Generate demo.ipynb.

Plumbing copied from `refusal-rate-monitor` (Day 193), content written fresh. The engine is EXTRACTED from
latency.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Every percentile is exact on the grid, so the notebook reproduces evidence.txt to the digit. Only the Monte Carlo
calibration runs at NB_REPS instead of the study's 200,000 - stated in the notebook, and a test asserts the note.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/latency-budget"
NB_REPS = 20_000


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("latency.py")

EXPORTS = [
    "SEED", "MC_REPS", "GRID_MS", "SHARDS", "SLO_MS", "HOPS", "BASE", "FIXES", "lognormal_pmf", "shift", "conv",
    "shard_pmf", "hop_pmfs", "chain", "percentile", "summary", "tail_attribution", "study", "fix_table",
    "simulate", "wilson", "calibrate", "parse_params",
]

HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
"""

cells: List[Dict[str, Any]] = []
_N = 0


def _lines(src: str) -> List[str]:
    """nbformat wants each source line to KEEP its trailing newline."""
    return src.strip("\n").splitlines(keepends=True)


def _nid() -> str:
    global _N
    _N += 1
    return f"c{_N:02d}"


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _lines(src)})


def code(src: str) -> None:
    cells.append({"cell_type": "code", "id": _nid(), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _lines(src)})


# ---------------------------------------------------------------------------

md(f"""
# It is slow and nobody knows where

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A RAG request walks five hops: gateway, query embedding, vector search fanned out to 8 shards, reranker, LLM. Each
hop has a dashboard with p50, p95 and p99. This notebook declares each hop's latency (a lognormal body plus the one
tail mechanism it really has), computes the end-to-end distribution EXACTLY on a 1 ms grid, and asks three
questions the per-hop dashboards answer wrongly: is the chain over its SLO, which hop owns the slow requests, and
which fix to fund.

1. The engine
2. Calibration - exact vs simulation
3. Percentiles do not add
4. Fan-out: every shard is healthy, the step is not
5. Who owns the average vs who owns the tail
6. Five fixes, ranked two ways
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `latency.py` at build time. Sequential hops convolve; a fan-out step is the max of 8 shards, so
its CDF is the shard CDF to the 8th power; a hedged call is a min, so survivals multiply; a timeout-and-retry is
a mixture of the body and (timeout + a fresh body).
""")

code(HEADER + "\n" + ENGINE)

md(f"""
## 2. Calibration - exact vs simulation

`simulate` draws the same model continuously and rounds each hop to the nearest ms (the grid's own convention).
Each exact tail probability P(T > t) should sit inside the 99% Wilson interval of the simulated share.

**Resolution of the instrument.** The study in `evidence.txt` uses {200_000:,} simulated requests; this notebook
uses {NB_REPS:,} to run in seconds, so its intervals are about 3x wider and it can flag a different cell than the
study does. The exact numbers below do not depend on this.
""")

code(f"""cal = calibrate(n={NB_REPS}, seed=SEED)
for r in cal:
    print(f"{{r['chain']:<30}} t={{r['t']:>5}}  exact {{r['exact']:.5f}}  mc {{r['mc']:.5f}}  {{'ok' if r['inside'] else 'OUTSIDE'}}")
print(f"{{sum(r['inside'] for r in cal)}}/{{len(cal)}} inside")""")

md("""
## 3. Percentiles do not add

The tempting budget: each hop owns a p99 budget, and the budgets add to the SLO. But a request that is slow in
the reranker is almost never also slow in the embedder, so the end-to-end p99 is far below the sum of hop p99s.
""")

code("""s = study()
print(f"{'hop':<10} {'p50':>6} {'p95':>6} {'p99':>6} {'mean':>8} {'p99/p50':>8}")
for h in HOPS:
    r = s["hops"][h]
    print(f"{h:<10} {r['p50']:>6} {r['p95']:>6} {r['p99']:>6} {r['mean']:>8.1f} {r['p99'] / r['p50']:>8.1f}")
e = s["e2e"]
print(f"{'END-TO-END':<10} {e['p50']:>6} {e['p95']:>6} {e['p99']:>6} {e['mean']:>8.1f}")
print(f"sum of per-hop p99 = {s['sum_of_hop_p99']} ms vs true p99 {e['p99']} ms  (overstates by {s['sum_of_hop_p99'] - e['p99']} ms)")
print(f"SLO p99 <= {SLO_MS} ms: per-hop sum says {s['sum_of_hop_p99'] - SLO_MS} ms over; truth {e['p99'] - SLO_MS} ms over, P(T > SLO) = {s['p_over_slo']:.2%}")""")

md("""
## 4. Fan-out: every shard is healthy, the step is not

One shard pauses for GC 0.8% of the time, below its own p99, so every shard dashboard is clean. The search step
waits for the slowest of 8, so it meets a pause on 1 - 0.992^8 of requests.
""")

code("""print(f"one shard p99 = {s['shard']['p99']} ms")
print(f"P(some shard pauses) = {s['p_any_shard_gc']:.1%}, search step p95 = {s['hops']['search']['p95']} ms")""")

md("""
## 5. Who owns the average vs who owns the tail

Mean share is what a latency breakdown chart shows. Tail excess is E[hop | request slower than p99] - E[hop]:
how much MORE of each hop a slow request carries. It is exact (one convolution per hop) and the five excesses sum
to E[T | slow] - E[T]. The hop with the worst p99/p50 ratio (embedding, 13x) is a third answer.
""")

code("""print(f"{'hop':<10} {'mean share':>11} {'tail excess ms':>15} {'tail share':>11}")
for h in HOPS:
    print(f"{h:<10} {s['mean_share'][h]:>11.1%} {s['tail_excess_ms'][h]:>15.1f} {s['tail_share'][h]:>11.1%}")""")

md("""
## 6. Five fixes, ranked two ways

A smaller LLM, a warm embedding pool, hedged shard calls, a shorter reranker timeout, and the three config
changes together. The mean ranks the LLM first by 5.6x; the p99 and the SLO rank the three config changes first.
""")

code("""fixes = fix_table()
print(f"{'fix':<30} {'d mean':>8} {'d p50':>6} {'d p99':>6} {'p99':>6} {'P(T>SLO)':>9}")
for r in fixes:
    print(f"{r['fix']:<30} {r['d_mean']:>+8.1f} {r['d_p50']:>+6} {r['d_p99']:>+6} {r['p99']:>6} {r['p_over_slo']:>9.2%}")""")

md("""
## 7. The chart

The full audit figure is `latency_audit.png` (built by `make_chart.py` from `results.json`). Here, the
end-to-end distribution before and after the two candidate investments.
""")

code("""fig, ax = plt.subplots(figsize=(10, 4.2))
for label, prm, col in [("today", BASE, "#1f2733"),
                        ("smaller LLM", {**BASE, **FIXES["smaller LLM (-15% median)"]}, "#2e8b57"),
                        ("3 config fixes", {**BASE, **FIXES["all three config fixes"]}, "#c8562b")]:
    p = chain(hop_pmfs(prm))
    surv = 1 - np.cumsum(p)
    ax.semilogy(np.arange(len(p)), np.clip(surv, 1e-6, 1), color=col, lw=2, label=f"{label} (p99 {percentile(p, 0.99)} ms)")
ax.axvline(SLO_MS, color="#c8562b", ls="--", lw=1)
ax.axhline(0.01, color="#8a94a3", ls=":", lw=1)
ax.set_xlim(800, 2000)
ax.set_ylim(1e-4, 1)
ax.set_xlabel("end-to-end latency (ms)")
ax.set_ylabel("P(slower than x)")
ax.set_title("The smaller LLM moves the whole curve; the config fixes cut the tail", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Adding per-hop p99s says the chain is 792 ms over a 1,500 ms SLO. It is 51 ms over.
- Every shard's p99 is 82 ms; the search step's p95 is 272 ms, because 8 shards meet a 0.8% pause 6.2% of the time.
- The LLM is 83% of the average and 26% of the tail excess; the reranker's hang-and-retry is 7% and 52%.
- Three config changes save 24 ms of mean and 190 ms of p99. A 15% smaller LLM saves 136 ms of mean and 154 ms
  of p99. A mean dashboard funds the LLM; the SLO wants the config changes (0.07% vs 0.33% of requests over).
- Negative result: on its own, no single config fix beats the smaller LLM on p99. The win needs all three.
""")

md("## 8. Try your own")

code("""# Override any BASE parameter, e.g. your reranker never hangs but your LLM is spikier:
# mine = parse_params("rerank_hang_p=0.0, llm_sigma=0.3")
# print(study(mine)["e2e"], study(mine)["tail_share"])
# for r in fix_table(mine): print(r["fix"], round(r["d_mean"]), r["d_p99"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your
own hop parameters: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`llm-router`](../llm-router) and [`semantic-cache`](../semantic-cache) are serving-path tools
that change latency; this build says where a latency budget is actually spent.
[`context-packing`](../context-packing) is an earlier audit of a metric that ranked the fixes backwards.
""")


def write() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open("demo.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote demo.ipynb with {len(cells)} cells ({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
