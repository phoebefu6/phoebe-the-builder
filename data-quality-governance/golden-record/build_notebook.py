"""Generate demo.ipynb.

Plumbing copied from `table-checksum` (Day 207), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Random(208)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed; every person is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/golden-record"


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

EXPORTS = ["make_world", "RULES", "GENERIC", "FIELDS", "golden", "study", "lines", "most_recent_field_ts"]

HEADER = """from __future__ import annotations

import json
import random
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

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
# Which customer record is real? Six survivorship rule-sets scored against the truth

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

The same customer sits in CRM, billing and the web shop, and the three copies disagree. A survivorship rule picks
one value per field to make the golden record. `dedup-pipeline` (Day 67) implements those rules; this notebook asks
the question nobody can ask on real data: is the kept value the TRUE one? It builds 364 fictional customers whose
details change over time, lets three systems capture them the way real systems do, and scores six rule-sets against
each customer's details today.

1. The engine
2. Field accuracy, rule by rule
3. How often the rule-sets disagree
4. Which timestamp 'most recent' reads
5. The ceiling
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Random(208)`), every person fictional, and
every number below matches `evidence.txt` exactly. The systems: CRM (keyed in by sales, early; later touched for an
opt-in change that refreshes no field), web (typed by the customer, recent; nicknames, swapped day and month,
skipped dates of birth), billing (address verified by a payment, but in 30% of cases overwritten by a nightly CRM
sync; company switchboard phones and invoice mailboxes). The six rule-sets: `most_recent`, `source_priority`
(crm > billing > web), `most_complete`, `longest`, `majority`, and `per_field` (a declared rule per field: trusted
source for name, dob and email, the field's own capture day for phone and address).
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Field accuracy, rule by rule

No generic rule-set wins every field. `source_priority` is best on name (98.1%: CRM has the legal name) and worst
on address (54.4%: CRM's copy is the oldest). `most_recent` is the reverse: best generic on address, worst on name
(76.6%), because the most recent record is the web form with the nickname. The declared `per_field` rule gets all
five fields right for 42.0% of customers against 35.2% for the best generic one - and still loses email to plain
`majority` (79.1% vs 85.4%), because trusting CRM for a field that changes keeps the old address book.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. How often the rule-sets disagree

The five generic rule-sets keep the same golden record for only 119 of 364 customers. For 80 they produce three or
four different golden records from the same rows. Picking a rule-set is not a technical detail: it decides a third
of the email and phone fields on its own.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Which timestamp 'most recent' reads

`updated_at` is a ROW timestamp: an opt-in change bumps it without refreshing the address. Reading the field's own
capture day helps a little (70.9% -> 72.5%). The rest is the sync: when CRM's old address is copied into billing and
stamped with the sync day, it looks like the freshest address in the cluster and wins two systems to one in a vote.
All 20 of the 20 customers in that exact shape got the stale address from `majority`. Carrying the original capture
day through the sync (lineage) brings address to 73.9%, the ceiling.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. The ceiling

For 163 of 364 customers (44.8%) at least one true value is held by no system at all: they moved, or changed phone,
after every capture. No survivorship rule can fix those fields; a perfect chooser would still get only 55.2% of
customers fully right. The fix for that half is a fresh capture - ask the customer, verify against a payment - not
a better rule.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The chart

The full figure, with the sync drawn, is `audit.png` (built by `make_chart.py`).
""")

code("""rules = list(r["acc"])
grid = [[r["acc"][k][f] for f in FIELDS] + [r["whole"][k]] for k in rules]
grid.append([r["ceiling"][f] for f in FIELDS] + [r["oracle_whole"]])
fig, ax = plt.subplots(figsize=(9, 5))
ax.imshow(grid, cmap="RdYlGn", vmin=0.3, vmax=1, aspect="auto")
ax.set_xticks(range(len(FIELDS) + 1), FIELDS + ["all five"])
ax.set_yticks(range(len(rules) + 1), rules + ["ceiling"])
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=9)
ax.axhline(len(rules) - 0.5, color="black", lw=1.5)
ax.set_title("Share of customers whose kept value is the true one", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- No generic rule-set wins every field: priority wins name, recency wins address, majority wins email.
- The five generic rule-sets agree on the whole golden record for 119 of 364 customers.
- A declared rule per field gets 42.0% of customers fully right; the best generic one 35.2%.
- `updated_at` is a row timestamp; a sync that re-stamps a copied value makes it look fresh and wins the vote.
- 44.8% of customers have a true value no system holds: that half needs a fresh capture, not a rule.
""")

md("## 7. Try your own")

code("""# world = make_world(seed=1)            # another fictional population
# c = world[0]
# for k, rule in RULES.items():
#     print(k, golden(c["records"], rule))
# print("truth", c["truth"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version resolves your
own cluster pasted as CSV: `pip install -r requirements.txt && streamlit run app.py`.
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
