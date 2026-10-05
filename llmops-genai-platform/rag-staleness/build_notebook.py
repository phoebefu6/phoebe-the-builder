"""Generate demo.ipynb.

Plumbing copied from `citation-verifier` (Day 190), content written fresh. The engine is EXTRACTED from
staleness.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Every policy number is closed-form, so the notebook reproduces evidence.txt to the digit. Only the Monte Carlo
calibration runs at NB_REPS instead of the study's 200,000 - stated in the notebook (the Day 179 resolution
note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/rag-staleness"
NB_REPS = 20_000


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER.

    Tuple targets are walked, because an extractor reading only `targets[0].id` drops them
    silently - the Day 178 defect.
    """
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("staleness.py")

EXPORTS = [
    "SEED", "MC_REPS", "CLASSES", "POLICIES", "FACTS", "stale", "stale_at", "rates", "evaluate", "cycle",
    "with_archive", "audit_sample", "tokens", "jaccard", "chunk", "query", "similarity_gap", "wilson", "simulate",
    "calibrate", "parse_classes", "parse_policy",
]

HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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
# It answered from last quarter's document

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A RAG index is a snapshot of documents that keep changing. The dashboard reports index freshness: the share of
documents whose indexed bytes still match the source. The user hits a different number: the share of *served
answers* whose chunk carries a fact that has since changed. This notebook declares a small corpus (three document
classes with their own edit rates and query shares), derives both numbers in closed form for three reindex
policies, checks the closed form against a raw simulation, and shows the two numbers ranking the policies in
opposite orders.

1. The model: a snapshot, a Poisson source, a uniform phase
2. Calibration - closed form vs raw simulation
3. Three policies, three numbers
4. Inside the cycle, and the archive that improves the metric
5. The retrieval score that cannot see staleness
6. Try your own
""")

md("""
## The engine

Extracted from `staleness.py` at build time, character for character. A document class has `docs`, a mean
number of days between edits, the share of edits that change a fact a chunk carries, and the share of served
answers that retrieve from it. A policy is a reindex period per class. `stale(mu, T)` is the probability that a
document reindexed every `T` days, served at a uniform phase, carries a changed fact.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every policy number here is closed-form and matches `evidence.txt` to the
digit. Only the calibration's raw simulation runs at {NB_REPS:,} replicates instead of the study's 200,000.
""")

md("## 1. The model")

code("""
for c in CLASSES:
    lam, mu = rates(c)
    print(f"{c['name']:<5} {c['docs']:>4} docs  edited every {c['change_every']:>4.0f} d  fact-bearing {c['fact_share']:.0%}  "
          f"query share {c['query_share']:.0%}   ({c['what']})")
print()
print("stale(mu, T) = 1 - (1 - exp(-mu T)) / (mu T)")
for T in (1, 7, 30, 90):
    print(f"  hot doc reindexed every {T:>2} days: served stale {stale(rates(CLASSES[0])[1], T):.1%} of the time")
""")

md("""
The hot class is 5% of the documents and half of the answers. Monthly reindexing leaves a hot document
serving a changed fact 44% of the time.

## 2. Calibration
""")

code(f"""
cal = calibrate({NB_REPS})
print(f"inside their 99% interval: {{sum(r['inside_99'] for r in cal)}} of {{len(cal)}}")
""")

md("## 3. Three policies, three numbers")

code("""
ev = {k: evaluate(p) for k, p in POLICIES.items()}
print(f"{'policy':<18} {'hash fresh':>10} {'fact fresh':>10} {'answers stale':>13} {'reindex/day':>11}")
for k, e in ev.items():
    print(f"{k:<18} {e['hash_freshness']:>10.1%} {e['fact_freshness']:>10.1%} {e['answer_staleness']:>13.1%} "
          f"{e['reindex_per_day']:>11.1f}")
v1, wk, ti = ev["v1 monthly full"], ev["v2a weekly full"], ev["v2b tiered 1/7/90"]
deltas = {"weekly: hash freshness": wk["hash_freshness"] - v1["hash_freshness"],
          "weekly: answer staleness": wk["answer_staleness"] - v1["answer_staleness"],
          "tiered: hash freshness": ti["hash_freshness"] - v1["hash_freshness"],
          "tiered: answer staleness": ti["answer_staleness"] - v1["answer_staleness"]}
print()
for k, d in deltas.items():
    print(f"{k:<28} {d * 100:+6.1f} pts")
print(f"weekly full serves {wk['answer_staleness'] / ti['answer_staleness']:.1f}x the stale answers of tiered at "
      f"{wk['reindex_per_day'] / ti['reindex_per_day']:.1f}x the reindex cost")
""")

code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
names = list(ev)
ys = np.arange(len(names))[::-1]
ax.barh(ys + 0.19, [(1 - ev[n]["hash_freshness"]) * 100 for n in names], 0.36, color="#8a94a3", label="docs not fresh (dashboard)")
ax.barh(ys - 0.19, [ev[n]["answer_staleness"] * 100 for n in names], 0.36, color="#7a4fb5", label="answers from a changed fact")
ax.set_yticks(ys); ax.set_yticklabels(names); ax.set_xlabel("stale share (%)"); ax.legend(loc="lower right")
ax.set_title("The dashboard ranks weekly first; the answers rank it last", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
Weekly full reindexing moves the dashboard 8 points and the answers 19. The tiered schedule moves the
dashboard 1 point and the answers 24, at 60% of the cost. Sorted by index freshness, weekly wins; sorted by
what the user receives, tiered wins by 2.8x.

## 4. Inside the cycle, and the archive
""")

code("""
cyc = cycle(POLICIES["v1 monthly full"])
for d in (1, 7, 14, 29):
    print(f"day {d:>2} of the monthly cycle: {cyc[d]['answer_staleness']:.1%} of answers stale")
print(f"phase average, what a monthly metric reports: {v1['answer_staleness']:.1%}")
print()
for extra in (0, 300, 1000):
    cls, pol = with_archive(extra, POLICIES["v1 monthly full"])
    e = evaluate(pol, cls)
    print(f"+{extra:>4} archive docs nobody asks about: index freshness {e['hash_freshness']:.1%}, answers stale {e['answer_staleness']:.1%}")
print()
print(f"audit by sampling documents uniformly estimates {audit_sample(POLICIES['v1 monthly full'], 'docs'):.1%}; "
      f"by sampling the query log, {audit_sample(POLICIES['v1 monthly full'], 'queries'):.1%}")
""")

md("""
The user on day 29 gets 46% stale answers against a reported 27%. Adding a thousand archive documents raises
index freshness six points and changes no answer. A uniform document sample is an unbiased estimate of the
dashboard's number; only a sample drawn from the query log estimates the user's.

## 5. The retrieval score
""")

code("""
sg = similarity_gap()
for r in sg["rows"][:4]:
    print(f"{r['query']:<44} stale {r['stale_sim']:.3f}  fresh {r['fresh_sim']:.3f}  gap {r['gap']:+.3f}")
print(f"... {sg['n']} facts, max |gap| {sg['max_abs_gap']:.3f}; a gate that keeps every fresh chunk passes "
      f"{sg['stale_passed_by_gate_keeping_all_fresh']:.0%} of stale ones")
""")

md("""
The query names the attribute and never the value, so swapping the value changes nothing the score can see.
Confidence is not a staleness signal, by construction.

## Summary

| claim | what the numbers say |
|---|---|
| "the index is 89% fresh" | and 27% of answers carry a changed fact: the queries land on the 5% of docs that move |
| "weekly reindexing fixed it" | index +8 pts, answers -19; the tiered schedule is +1 and -24 at 60% of the cost |
| "the monthly number is 27%" | it is 2.5% on day 1 and 46% on day 29 |
| "freshness went up this quarter" | 1,000 archive docs did that; no answer changed |
| "the retriever was confident" | identical similarity for the stale and the fresh chunk, 12 of 12 facts |
""")

md("## 6. Try your own")

code("""
# cls = parse_classes("pricing, 5, 10, 0.7, 6\\nguides, 60, 60, 0.5, 3\\nlegal, 200, 400, 0.2, 1")
# for name, text in (("daily full", "1"), ("tiered", "pricing=1, guides=14, legal=120")):
#     e = evaluate(parse_policy(text, cls), cls)
#     print(name, f"index fresh {e['hash_freshness']:.1%}  answers stale {e['answer_staleness']:.1%}  "
#           f"reindex/day {e['reindex_per_day']:.1f}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you describe your own corpus classes and reindex policies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`data-freshness-monitor`](../../data-infra-toolkit/data-freshness-monitor) asks whether a table
is late; this build asks whether the answers served from an index are out of date, which is a different
weighting of a different quantity. [`citation-verifier`](../citation-verifier) is the previous day's audit of
a metric that moved against the truth.
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
    print(f"wrote demo.ipynb with {len(cells)} cells "
          f"({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
