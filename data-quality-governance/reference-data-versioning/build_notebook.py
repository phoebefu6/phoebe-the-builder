"""Generate demo.ipynb.

Plumbing copied from `golden-record` (Day 208), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Random(209)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed; every territory is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/reference-data-versioning"


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

EXPORTS = ["make_world", "rows_known", "lookup", "report", "restate", "study", "lines", "STRATEGIES", "REGIONS"]

HEADER = """from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
# The country list changed: can last year's regional report be rebuilt today?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Finance asks for last year's revenue by region, and the number no longer matches the report that went out. The
territory list changed underneath it: territories moved region, codes were renamed, split, merged, and one code was
retired and reused. This notebook builds 40 fictional territories over 36 months, publishes a regional report every
month from the table as it stood that day, then tries to rebuild those reports four ways from what is stored today.

1. The engine
2. Rebuilding the published reports
3. The reused code
4. Backdated corrections
5. As-was vs as-is
6. The split nobody can undo
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Random(209)`), every territory fictional, and
every number below matches `evidence.txt` exactly. The reference table is SCD2 (`valid_from`, `valid_to` per
version); each change also carries the month it was RECORDED, which is later than the month it took effect for 3
region moves. Sales facts carry the code that was valid on the day of the sale.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Rebuilding the published reports

Joining to today's table only (`overwrite`) reproduces 10 of 36 months and drops 6.3% of revenue: sales under codes
that were renamed, split or merged away find no row. Joining the full history on the code alone (`code_only`)
drops nothing and reports 115.7% of revenue, because every code that ever changed matches several versions and the
sale is counted once per version - it reproduces no month at all. The as-of join (`scd2_asof`) gets every dollar
and 29 of 36 months. Only the bitemporal join reproduces all 36.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. The reused code

A retired code that comes back for a different territory is the quiet one: nothing errors. Today's table knows
only the new owner, so `overwrite` files the old territory's whole history under the new territory's region.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Backdated corrections

An SCD2 table records when a change was valid, not when anybody knew it. Three region moves were keyed in 2-4
months after they took effect, so the reports for those months went out with the old region. The as-of join on
today's table quietly rewrites them: 7 of 36 months (15 cells) no longer match what was sent. A second time axis
(when the row was recorded) fixes it at the cost of a handful of rows; storing a monthly snapshot of the whole table
does too, at 25 times the rows.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. As-was vs as-is

"Last year by region" has two right answers. As-was is what was reported. As-is moves every past sale to the
region its territory sits in today, so a trend line compares like with like. They differ: the as-was rebuild sits
within 1% of the true as-is restatement in only 61 of 180 cells. The failure is not picking one; it is not knowing
which one a number is.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The split nobody can undo

To restate a split territory to today's regions, each old sale must be given to one of the two halves, and the sale
never recorded which. A 50/50 rule puts 10.2% of that revenue in the wrong region; allocating by each half's share
after the split does worse, 20.5%, because the mix drifted in the years before the split (by construction here, and
commonly in practice: the split usually happens because one half grew). This is the negative result: no rule over
the reference data recovers it. Only capturing the finer location on the sale itself does.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The full figure, with the reused-code mechanism drawn, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(r["asw"])
fig, ax = plt.subplots(figsize=(8, 3.6))
months = [r["asw"][k]["months"] for k in names]
ax.barh(names[::-1], months[::-1], color=["#2e8b57", "#e0a23a", "#c8562b", "#c8562b"])
for i, k in enumerate(names[::-1]):
    a = r["asw"][k]
    ax.text(a["months"] + 0.5, i, f"{a['months']} of 36  ({100 * a['reported']:.1f}% of revenue)", va="center")
ax.set_xlim(0, 52)
ax.set_title("Published months each rebuild reproduces", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Today's table alone reproduces 10 of 36 months and drops 6.3% of revenue; a code-only history join reports 115.7%.
- A reused code files the old territory's history under the new one, with no error.
- Valid time alone misses every month whose change was recorded late; add a recorded-time axis.
- As-was and as-is are both right answers to different questions: label which one every report is.
- A split cannot be restated from reference data: capture the finer location on the sale.
""")

md("## 8. Try your own")

code("""# w = make_world(seed=1)                       # another fictional territory history
# for e in w["events"]: print(e)
# rep, lost = report(w, "scd2_asof")
# print(dict(rep[20]), lost)""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version rebuilds any
month four ways next to what was published: `pip install -r requirements.txt && streamlit run app.py`.
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
