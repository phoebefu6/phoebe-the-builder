"""Generate demo.ipynb.

Plumbing copied from `model-migration-diff` (Day 188), content written fresh. The engine is EXTRACTED from
trajectory.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The scorer numbers are exact (every trajectory enumerated), so the notebook reproduces them to the digit.
Only the Monte Carlo calibration runs at NB_REPS instead of the study's 200,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/agent-trajectory-eval"
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


ENGINE = extract("trajectory.py")

EXPORTS = [
    "P_ELIG", "ORDER", "WRONG", "PLANS", "PLAN_STEPS", "AGENTS", "SCORERS", "INVARIANTS", "MC_REPS", "SEED",
    "reference", "build", "violations", "score", "enumerate_runs", "summarise", "evaluate", "simulate", "wilson",
    "calibrate", "parse_steps", "audit",
]

HEADER = """from __future__ import annotations

import itertools
from collections import Counter
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
# The answer was right. Was the path?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A refund agent is scored on its final answer. This notebook enumerates every trajectory a declared agent
can produce, and measures - exactly - how often a right answer sits on a broken path, and what the usual
reference-match modes (strict, unordered, superset) see instead.

1. The task and the spec
2. Calibration - exact enumeration vs raw simulation
3. Five scorers, two agent versions
4. The prompt change, as each scorer reports it
5. Try your own
""")

md("""
## The engine

Extracted from `trajectory.py` at build time, character for character. The agent makes a few discrete
choices (which reads, in what order, an extra search, act before reading, retry the refund, mangle the
order id). `enumerate_runs` lists all 512 combinations with their probabilities; every scorer then runs on
the real step list of each one.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every scorer number here is exact and matches `evidence.txt` to the digit.
Only the calibration's raw simulation runs at {NB_REPS:,} replicates instead of the study's 200,000.
""")

md("## 1. The task and the spec")

code("""
print("reference path, eligible:", reference(True))
print("invariants:", ", ".join(INVARIANTS))
bad = [("lookup_order", ORDER), ("check_policy", None), ("issue_refund", WRONG)]
print("refund to the wrong order -> violations", violations(bad, True), " scores", score(bad, True, True))
""")

md("""
The last line is the whole problem in one row: every name-based scorer passes a refund sent to the wrong
order, because the tool names and their order are exactly the reference.

## 2. Calibration
""")

code(f"""
cal = calibrate({NB_REPS})
print(f"inside their 99% interval: {{sum(r['inside_99'] for r in cal)}} of {{len(cal)}}")
""")

md("## 3. Five scorers, two agent versions")

code("""
ev = evaluate()
for name, s in ev.items():
    print(f"{name}: truly valid {s['p_good']:.1%}, answer right {s['p_correct']:.1%}, calls {s['calls']:.2f}")
    for k, v in s["scorers"].items():
        print(f"   {k:<17} reports {v['reported']:6.1%}  passes broken {v['false_pass']:6.1%}  "
              f"fails valid {v['false_fail']:6.1%}  broken among passes {v['bad_among_passes']:6.1%}")
""")

md("""
Outcome-only scoring never fails a valid run and passes 58% of broken ones. Strict reference match is the
mirror image: it fails 57% of v1's valid runs (another order, an extra search) and still passes the wrong
order id. Superset is the best name matcher and still passes a third of v1's broken runs - duplicates and
refunds-before-reading. No name matcher gets both errors low; the spec does, because the spec is what
"valid" means.

## 4. The prompt change
""")

code("""
v1, v2 = ev["v1 careful"], ev["v2 fewer calls"]
names = ["truly valid"] + list(SCORERS)
d = [v2["p_good"] - v1["p_good"]] + [v2["scorers"][k]["reported"] - v1["scorers"][k]["reported"] for k in SCORERS]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(names[::-1], [x * 100 for x in d][::-1], color=["#7a4fb5", "#e3b23c", "#8a6d3b", "#8a94a3", "#2f6fdb", "#1f2733"])
ax.set_xlabel("change in reported pass rate, v1 -> v2 (points)")
ax.set_title("A 'fewer tool calls' prompt change", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
for n, x in zip(names, d):
    print(f"{n:<17} {x * 100:+6.1f} pts")
""")

md("""
The truly valid rate fell 31 points. Outcome scoring saw 13 of them. Strict reference matching saw 1 -
v2 also stopped doing the harmless things (extra search, reads in the other order) that strict was
failing all along, so the two moves cancel.

## Summary

| claim | what the numbers say |
|---|---|
| "the answer is right, so the run passed" | 35% of v2's right-answer runs broke the path |
| "match the reference trajectory" | strict fails 57% of valid v1 runs and reports -1 pt for a -31 pt change |
| "superset match plus the outcome" | best of the matchers, still passes a quarter of v1's broken runs |
| "write invariants" | the only scorer that encodes order, count and argument; its cost is writing them |
""")

md("## 5. Try your own")

code("""
# runs = [(True, True, "lookup_order:A17;check_policy;issue_refund:A17"),
#         (True, True, "issue_refund:A17;lookup_order:A17;check_policy")]   # eligible, answer right, steps
# res = audit(runs)
# print(res["good"], res["scorers"]["outcome"], dict(res["hidden"]))
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own runs:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`model-migration-diff`](../model-migration-diff) shows an unchanged score hiding churn;
[`agent-eval-dashboard`](../../ai-agent-workshop/agent-eval-dashboard) is the outcome-pass-rate dashboard this
build audits.
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
