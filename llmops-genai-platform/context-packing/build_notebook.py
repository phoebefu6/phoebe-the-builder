"""Generate demo.ipynb.

Plumbing copied from `rag-staleness` (Day 191), content written fresh. The engine is EXTRACTED from packing.py by
AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank
one (the Day 177 defect).

Every policy number is an exact expectation over the declared, seeded log, so the notebook reproduces
evidence.txt to the digit. Only the Monte Carlo calibration runs at NB_REPS replicates per query instead of the
study's 200 - stated in the notebook (the Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/context-packing"
NB_REPS = 20


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


ENGINE = extract("packing.py")

EXPORTS = [
    "SEED", "MC_REPS", "Q", "N_CAND", "POLICIES", "LIU_POINTS", "make_log", "curve", "read", "comply", "select",
    "arrange", "window", "score_query", "evaluate", "sweep", "same_set_different_order", "by_rank", "wilson",
    "simulate", "evaluate_present", "calibrate", "parse_policy",
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
# The fact was in the context, and the model did not read it

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A RAG answer is generated from a window: the instruction, the retrieved chunks a packer kept under a token
budget in the order it chose, then the question. The eval column that describes the window is *context recall*,
the share of needed facts physically present. The model does not read every position equally well: Liu et al.
(2023, "Lost in the Middle") measured a U over position that sinks as the context grows, and an instruction at
the top is obeyed less the further it sits from the question. This notebook declares a seeded query log and a
reading model, scores four packing policies exactly, checks the exact numbers against a raw simulation, and
shows the dashboard ranking the policies in the opposite order from the answers.

1. The log and the reading model
2. Calibration - exact expectation vs raw simulation
3. Four policies, three numbers
4. Where the answers went - three failure classes
5. The budget sweep
6. Same chunks, different order
7. Try your own
""")

md("""
## The engine

Extracted from `packing.py` at build time, character for character. `make_log` is the declared query log;
`read(u, L)` and `comply(d)` are the reading model; `select`, `arrange` and `window` are the packer;
`score_query` and `evaluate` are the exact rates; `simulate` and `calibrate` are the raw check.
""")

code(HEADER + "\n" + ENGINE + "\n\nLOG = make_log()\nprint(f'engine loaded, {len(LOG):,} queries in the log')")

md(f"""
**Resolution of the instrument.** Every policy number here is an exact expectation over the declared log and
matches `evidence.txt` to the digit. Only the calibration's raw simulation runs at {NB_REPS} replicates per
query instead of the study's 200.
""")

md("## 1. The log and the reading model")

code("""
print("read(u, L): first / middle / last position, then comply with the instruction at the top")
for L in (2_000, 4_000, 8_000, 12_000):
    print(f"  L = {L:>6,}: read first {read(0, L):.3f}  middle {read(0.5, L):.3f}  last {read(1, L):.3f}   "
          f"comply {comply(L - INSTRUCTION_TOKENS - QUESTION_TOKENS):.3f}")
print()
e = LOG[0]
w = window(e["lens"], POLICIES["v1 rank, 4k"])
print(f"query 0 needs facts at ranks {e['gold']}; v1 keeps ranks {w['kept']} in order {w['layout']}, {w['total']} tokens")
""")

md("## 2. Calibration")

code(f"""
cal = calibrate(LOG, reps={NB_REPS})
for r in cal:
    print(f"{{r['policy']:<24}} {{r['quantity']:<17}} exact {{r['exact']:.4f}}   MC {{r['mc']:.4f}}   "
          f"{{'inside' if r['inside_99'] else 'OUTSIDE'}}")
print(f"{{sum(r['inside_99'] for r in cal)}} of {{len(cal)}} inside their 99% Wilson interval")
""")

md("""
Context recall has no randomness in it (a fact is in the window or it is not), so the simulation asserts the
count rather than interval-checking it. The first version of this build averaged recall per query while the
simulation pooled facts, and the deterministic column read as OUTSIDE its own interval - two readers of one
number. One definition now.

## 3. Four policies, three numbers
""")

code("""
ev = {k: evaluate(p, LOG) for k, p in POLICIES.items()}
print(f"{'policy':<24} {'context recall':>14} {'effective recall':>16} {'answer rate':>11} {'comply':>7} {'tokens':>7} {'chunks':>6}")
for k, e in ev.items():
    print(f"{k:<24} {e['context_recall']:>14.1%} {e['effective_recall']:>16.1%} {e['answer_rate']:>11.1%} "
          f"{e['compliance']:>7.1%} {e['tokens_per_query']:>7.0f} {e['chunks_per_query']:>6.1f}")
v1, v2, v3, v4 = (ev[k] for k in POLICIES)
deltas = {
    "12k window: context recall": v2["context_recall"] - v1["context_recall"],
    "12k window: answer rate": v2["answer_rate"] - v1["answer_rate"],
    "reorder: context recall": v3["context_recall"] - v1["context_recall"],
    "reorder: answer rate": v3["answer_rate"] - v1["answer_rate"],
    "reorder+repeat: context recall": v4["context_recall"] - v1["context_recall"],
    "reorder+repeat: answer rate": v4["answer_rate"] - v1["answer_rate"],
}
print()
for k, d in deltas.items():
    print(f"{k:<32} {d * 100:+6.1f} pts")
print()
print("by context recall: " + " > ".join(sorted(ev, key=lambda k: -ev[k]["context_recall"])))
print("by answer rate:    " + " > ".join(sorted(ev, key=lambda k: -ev[k]["answer_rate"])))
""")

code("""
fig, ax = plt.subplots(figsize=(8, 3.8))
names = list(ev)
ys = np.arange(len(names))[::-1]
ax.barh(ys + 0.19, [ev[n]["context_recall"] * 100 for n in names], 0.36, color="#8a94a3", label="context recall (dashboard)")
ax.barh(ys - 0.19, [ev[n]["answer_rate"] * 100 for n in names], 0.36, color="#2e8b57", label="answer rate (user)")
ax.set_yticks(ys); ax.set_yticklabels(names); ax.set_xlabel("share (%)"); ax.legend(loc="lower right")
ax.set_title("The dashboard ranks the 12k window first; the answers rank it last", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
Tripling the budget puts 10 more points of needed facts into the window and takes 11 points off the answer rate:
every position reads worse in a longer window, and the instruction at the top is obeyed 69% of the time instead
of 87%. Repeating the instruction before the question costs one chunk (context recall falls 1.4 points) and
gains 4.5 points of answers. The dashboard scores the first change up and the second down.

## 4. Where the answers went
""")

code("""
print(f"{'policy':<24} {'retrieval miss':>14} {'present, unread':>16} {'instruction ignored':>19}")
for k, e in ev.items():
    print(f"{k:<24} {e['fail_missing']:>14.1%} {e['fail_unread']:>16.1%} {e['fail_ignored']:>19.1%}")
print()
print(f"v1: the largest failure class is a fact that WAS in the window ({v1['fail_unread']:.1%} of queries); "
      "context recall counts it as a success")
""")

md("""
The three classes sum to one minus the answer rate. Context recall measures the first; the other two are
invisible to it, and the larger of them is the one it cannot see.

## 5. The budget sweep
""")

code("""
budgets = [1_000, 1_500, 2_000, 3_000, 4_000, 5_000, 6_000, 8_000, 10_000, 12_000]
for name, (order, instr) in {"rank, top": ("rank", "top"), "reorder, both": ("reorder", "both")}.items():
    rows = sweep(budgets, LOG, order, instr)
    best = max(rows, key=lambda r: r["answer_rate"])
    print(f"{name}: context recall {rows[0]['context_recall']:.1%} -> {rows[-1]['context_recall']:.1%}, never falls; "
          f"answer rate peaks at {best['budget']:,} tokens ({best['answer_rate']:.1%}) and is {rows[-1]['answer_rate']:.1%} at 12k")
""")

md("""
Context recall is monotone in the budget, so a dashboard built on it can only ever recommend a bigger window.
The answer rate has an interior maximum.

## 6. Same chunks, different order
""")

code("""
sd = same_set_different_order(LOG, POLICIES["v1 rank, 4k"], POLICIES["v3 reorder, 4k"])
print(f"identical kept sets on all {Q:,} queries: {sd['identical_sets']};  context recall gap {sd['context_recall_gap']:+.4f};  "
      f"tokens gap {sd['tokens_gap']:+.1f}")
print(f"gold chunks in the middle third: {sd['gold_in_middle_a']:.1%} -> {sd['gold_in_middle_b']:.1%};  "
      f"answer rate gap {sd['answer_rate_gap'] * 100:+.1f} pts")
print()
v1r = {r["rank"]: r for r in by_rank(POLICIES["v1 rank, 4k"], LOG, 5)}
v3r = {r["rank"]: r for r in by_rank(POLICIES["v3 reorder, 4k"], LOG, 5)}
for r in range(1, 6):
    print(f"rank {r}: read {v1r[r]['read']:.3f} under rank order, {v3r[r]['read']:.3f} under reorder")
""")

md("""
Nothing the retrieval log records distinguishes the two policies, and the answer rate moves. It moves very
little: the published mitigation halves the share of needed chunks in the middle third, but it also sends the
rank-2 chunk from second-from-top to the far end, where it reads worse, and 39% of needed facts sit at rank 1 or 2.

## Summary

| claim | what the numbers say |
|---|---|
| "context recall is 83%" | and 34% of queries lose a fact that was in the window; the answer rate is 37% |
| "a bigger window will fix it" | 12k: context recall +9.6 pts, answers -10.6 pts; the dashboard ranks it first |
| "reorder for lost-in-the-middle" | same chunks, same recall, +0.7 pts: rank 2 moves to the far end |
| "repeat the instruction" | context recall -1.4 pts, answers +4.5 pts; the dashboard scores it down |
| "more context recall is better" | it is monotone in the budget; the answer rate peaks at 4-5k |
""")

md("## 7. Try your own")

code("""
# for text in ("3000, rank, top", "3000, reorder, both", "8000, reverse, both"):
#     e = evaluate(parse_policy(text), LOG)
#     print(f"{text:<22} context recall {e['context_recall']:.1%}  answer rate {e['answer_rate']:.1%}  "
#           f"unread {e['fail_unread']:.1%}  tokens {e['tokens_per_query']:.0f}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you describe your own packing policies:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`chunk-optimizer`](../chunk-optimizer) chooses how to cut documents
into chunks; this build asks what happens to the chunks once they are in the window. [`rag-eval`](../rag-eval)
reports context recall; this is the number it leaves out. [`rag-staleness`](../rag-staleness) is the previous
day's audit of a metric that ranked the fixes backwards.
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
