"""Generate demo.ipynb.

Plumbing copied from `forecast-reconciliation` (Day 211), content written fresh. The engine is EXTRACTED from
explore.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook with a
blank one (the Day 177 defect).

Everything in this build is seeded (Generator(212)), so the notebook reproduces evidence.txt to the character.
Every customer, product and order is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/fanout-join-audit"


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

EXPORTS = ["make_world", "build", "checks", "run", "study", "seeds", "lines", "SCENARIOS", "CHECKS"]

HEADER = """from __future__ import annotations

import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# Revenue went up after the join: which checks catch fan-out before the aggregate

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A report of revenue by customer tier joins orders to a customer table, a product table and a promo-code table. If
any of those has more than one row per key, every matching order is counted twice, and the total goes up without an
error anywhere. This notebook builds a fictional shop, breaks the join six common ways, and runs six checks that a
team could put in front of the aggregate.

1. The engine
2. What each broken join does to the report
3. Which check fires
4. The declared test passes on the biggest defect
5. Why the row count understates the revenue error
6. Two defects that cancel
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Generator(212)`), everything fictional, and
every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
out = lines(r, s)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. What each broken join does to the report

Forgetting `WHERE is_current` on the customer history adds 8.9% rows and 14.5% revenue, and the gold tier reads
26.5% high, because each upgraded customer's orders are counted once under the old tier and once under the new one.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Which check fires

Six checks, all run on the joined rows or the dimensions before anything is summed. Only `pk_preserved` (every source
order appears exactly once in the joined rows) catches all six broken joins with no false alarm. The verdicts are
structural: across 30 more worlds every check gives the same verdict on every join.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. The declared test passes on the biggest defect

A type-2 history table is unique on (key, valid_from), not on the key, so that is the test a team declares. It
passes. The join uses the key alone. The test checks the table; the defect is in how the query uses it.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. Why the row count understates the revenue error

When the duplicated rows are random (a double load, a promo bridge), revenue inflation is about the row inflation.
When they belong to big spenders (upgraded customers), revenue moves 1.71x as far as rows in the median world. A
1% row tolerance is not a 1% revenue tolerance.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. Two defects that cancel

The negative result for the two most common checks. A missing filter (rows up) plus an inner join that drops
late-arriving customers (rows down) can cancel in the row count, or in the revenue total. These two cases are tuned
to cancel exactly; how often real defects cancel is not measured here. The point is that a total can be right while
every cell under it is wrong.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The two-panel figure is `audit.png` (built by `make_chart.py`). This is the check matrix on its own.
""")

code("""sc = r["scenarios"]
m = np.array([[1 if sc[k]["fires"][c] else 0 for c in CHECKS] for k in SCENARIOS])
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.imshow(m, cmap="Blues", vmin=-0.3, vmax=1.3, aspect="auto")
for i in range(len(SCENARIOS)):
    for j in range(len(CHECKS)):
        ax.text(j, i, "alarm" if m[i, j] else "pass", ha="center", va="center", fontsize=8,
                color="white" if m[i, j] else "#1f2933")
ax.set_xticks(range(len(CHECKS)), [c.replace("_", "\\n") for c in CHECKS], fontsize=8)
ax.set_yticks(range(len(SCENARIOS)), SCENARIOS, fontsize=8)
ax.set_title("Which pre-aggregate check fires on which join", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- A dimension with more than one row per key silently multiplies every order that matches it.
- The test a type-2 table declares, unique on (key, valid_from), passes while the join on the key fans out.
- Non-random duplicates move revenue further than rows: a row tolerance is not a revenue tolerance.
- Row count and revenue total can each be fooled by two defects that cancel.
- Assert the grain before you sum: every source key exactly once in the joined rows. It says something broke, not which join; a referenced-key uniqueness check names the dimension.
""")

md("## 8. Try your own")

code("""# w = make_world(seed=7)                           # another fictional shop
# b = build(w, "promo_bridge")                     # or any of SCENARIOS
# print(checks(w, b))
# j = b["joined"]; print(j.order_id.is_unique, len(j), len(w["orders"]))""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version lets you pick a
broken join and see which checks pass it: `pip install -r requirements.txt && streamlit run app.py`.
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
