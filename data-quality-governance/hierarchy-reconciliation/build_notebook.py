"""Generate demo.ipynb.

Plumbing copied from `reference-data-versioning` (Day 209), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Random(210)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed; every cost centre is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/hierarchy-reconciliation"


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start : node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("explore.py")
EVIDENCE = extract("evidence.py")

EXPORTS = ["make_world", "rollup", "breakdown", "defects", "one_parent", "study", "seeds", "lines", "METHODS", "REGIONS"]

HEADER = """from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


class _E:  # evidence.py refers to the engine as E
    pass
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
    cells.append(
        {
            "cell_type": "code",
            "id": _nid(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(src),
        }
    )


# ---------------------------------------------------------------------------

md(f"""
# The regions do not sum to the total: rolling up a hierarchy extracted with defects

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

The cost report by region does not tie to the ledger, or it ties and the regions are still wrong. The cause is
usually the hierarchy, not the postings: a parent-child extract where moved cost centres kept their old row, a
retired area left orphans, and one mis-keyed parent field made a loop. This notebook builds a fictional hierarchy
with those defects, rolls a year of postings up five ways, and scores every posted dollar against the truth.

1. The engine
2. What a validator finds
3. Five roll-ups, per dollar
4. The plug
5. Which parent row to keep
6. What no roll-up rule can place
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Random(210)`), every cost centre fictional, and
every number below matches `evidence.txt` exactly. The extract is a list of (child, parent, load) rows; the truth is
a single-parent map the simulation keeps aside.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
out = lines(r, s)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. What a validator finds

Four checks run on the extract before any roll-up: a child with more than one parent row, a parent that is not in
the extract, a node whose parent chain loops, and a posting to a code the extract does not hold. One mis-keyed parent
field on a single area cut 9 nodes and 7.2% of dollars off the tree.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Five roll-ups, per dollar

The check every report runs, "do the regions sum to the total?", passes for all five: it compares the report with
itself. A recursive walk along every edge (what a recursive CTE does) reaches 92.5% of the ledger, counting 8.2% of
dollars twice and placing 15.6% nowhere. Keeping one parent row per child removes the double count, but the first row
in the file is the old parent for some of the moves.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. The plug

Finance makes the report tie by adding an 'Other' line: ledger minus the regions. Here the plug is 7.5% of the
ledger, and underneath it 23.8% of dollars are in the wrong place, because double counts and drops cancel in the net.
The plug measures the net; the error is the gross.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. Which parent row to keep

When a moved cost centre has two parent rows, the file order is not a rule. Keeping the latest load files every move
correctly here, but only because this simulation always loads the new row later; in a real extract, check that the
load stamp means "when this parent became true" before trusting it.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. What no roll-up rule can place

This is the negative result. After validation the total ties and every unplaced dollar is named, but 13.4% of the
ledger is still unplaced and no region is within 0.5% of the truth. The extract does not say where an orphan's
retired area went or what the looping area's parent should have been. Across 30 seeds that share runs from 5.8% to
14.1%. Only a fix in the source hierarchy places it.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The full figure, with the three defects drawn, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(r["methods"])[::-1]
segs = [("right", "#2e8b57"), ("misfiled", "#e0a23a"), ("dropped", "#c8562b"), ("double", "#7b5ea7")]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, k in enumerate(names):
    x = 0.0
    for key, col in segs:
        v = 100 * r["methods"][k][key]
        ax.barh(i, v, left=x, color=col, label=key if i == 0 else None)
        x += v
    ax.text(x + 1, i, f"{100 * r['methods'][k]['wrong']:.1f}% wrong", va="center")
ax.set_yticks(range(len(names)), names)
ax.set_xlim(0, 135)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Where each posted dollar ends up", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- "Regions sum to the total" passes for every method: it is a consistency check, not a correctness check.
- A recursive walk counts moved cost centres twice and drops orphans, cycles, unknown codes and area-level postings.
- A plug ties the total and hides a gross error several times its size.
- Validate the extract first (multi-parent, orphan, cycle, unknown code) and name what cannot be placed.
- The unplaced share is fixed in the source hierarchy, not in the roll-up.
""")

md("## 8. Try your own")

code("""# w = make_world(seed=1)                    # another fictional hierarchy
# print(defects(w))
# print(rollup(w, "validated"))
# print(breakdown(w, "first_parent"))""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version rolls the
hierarchy up with any of the five methods next to the truth: `pip install -r requirements.txt && streamlit run app.py`.
""")

def write() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open("demo.ipynb", "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote demo.ipynb with {len(cells)} cells ({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
