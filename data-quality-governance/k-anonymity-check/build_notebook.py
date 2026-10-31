"""Generate demo.ipynb.

Plumbing copied from `dedupe-key-choice` (Day 205), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Random(206)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/k-anonymity-check"


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

EXPORTS = ["population", "sample", "key", "profile", "ladder", "sampling_sweep", "linkage", "study", "lines", "FULL", "K"]

HEADER = """from __future__ import annotations

import random
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import combinations, product
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
# The anonymised export was re-identified: how few columns it takes, counted on one town

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

The names were removed. Postcode, date of birth and sex were left in, because a clinic's analysts need them. This
notebook builds a fictional town of 60,000 residents, draws a 3,000-row clinic export from it, and COUNTS how many
rows those three columns point at one person - in the export, and in the whole town - then walks the
generalisation ladder (date to month to year to decade; postcode to sector) to see what each step buys, and runs a
linkage attack with a public register that is complete and one that is missing a fifth of the town. It is not a
redactor (see `pii-redactor`, Day 124); it measures what is left after the redactor is done.

1. The engine
2. How few columns make a row unique
3. The generalisation ladder
4. Unique in the export is not unique in the town
5. The linkage attack, and the register that is wrong
6. k does not protect the value
7. The ceiling
8. The chart
9. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. The town is seeded (`Random(206)`), every resident is
fictional, and every number below matches `evidence.txt` exactly. The study takes a few seconds.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. How few columns make a row unique

Two. The postcode alone makes no resident unique in the town (the smallest postcode holds 15 people), and sex alone
makes none; the date of birth alone already makes 498 of the 3,000 exported residents the only person in town born
that day. Postcode + date of birth make 2,932 of 3,000 unique in the whole town, and adding sex lifts that to
2,964. The third column added 32 rows; the second added 2,434.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. The generalisation ladder

Forty ways to coarsen the three columns. Year of birth with the full postcode still leaves 2,487 export rows unique
and 1,097 unique in the town; the decade still leaves 111 town-unique rows. Only 6 of the 40 generalisations reach
k = 5 in the export without suppressing a row, and the finest of them is sector + sex, 40 classes, with the date of
birth gone entirely. Allowing 5% of rows to be suppressed, the finest is sector + decade with sex dropped: 199
classes, 51 rows suppressed.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Unique in the export is not unique in the town

At sector + year + sex, 40.1% of the 3,000 export rows are unique in the export and 0.03% - one row - is unique in
the town. The export's own uniqueness is an upper bound, and it is loose until the export is most of the town. Which
bound matters depends on who is looking: an insider who knows the person is a patient needs only the export, and
re-identifies 1,203 people; a stranger needs the town, and re-identifies 1. At the full key both numbers are the
same, 98.8%, because the key is finer than the town.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. The linkage attack, and the register that is wrong

An attacker holds a public register with names and the same three columns. Complete, it re-identifies exactly the
town-unique rows - 2,964 - and is never wrong. Missing 20% of the town, at the full key it loses 599 rows to
"not in register" and is wrong once; at year of birth it is right 977 times and wrong 101 times, each of those a
confident match to the wrong resident, because the right one is not in the register and someone else shares the
key. At sector + year it is wrong more often than right.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. k does not protect the value

k is about the quasi-identifiers; the diagnosis is a fourth column that k never looks at. In the 199-class release
no class is homogeneous, but 31 rows sit in classes where one diagnosis is at least 80% of the class. Ten of the 40
generalisations produce a k = 5 release in which an attacker who places a person in their class is at least 50%
sure of a sensitive diagnosis (baseline 11.7%).
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The ceiling

The export's k is not the town's k, and the town is not in the file. At sector + year + sex the export has 2,740
rows below k = 5 and the town has 52; neither number can be read off the other. And a six-digit postcode that
holds 15 residents cannot hide anyone, whatever is done to the other columns.
""")

code("""print("\\n".join(section(6)))""")

md("""
## 8. The chart

The full figure, with one row walked down the ladder, is `audit.png` (built by `make_chart.py`).
""")

code("""sw = [s for s in r["sweep"] if tuple(s["levels"]) == ("sector (2 digits)", "year", "kept")]
xs = [100 * s["frac"] for s in sw]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(xs, [s["unique_export_pct"] for s in sw], "-o", color="#2f6fdb", label="unique in the export")
ax.plot(xs, [s["unique_pop_pct"] for s in sw], "-s", color="#c8562b", label="unique in the town")
ax.set_xscale("log")
ax.set_xticks(xs, [f"{x:g}%" for x in xs])
ax.set_xlabel("share of the town in the export")
ax.set_ylabel("rows unique on sector + year + sex (%)")
ax.set_title("Unique in the export is an upper bound, loose until the export is the town", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Two columns do it: postcode + date of birth make 97.7% of the exported residents unique in a town of 60,000.
- Of 40 generalisations, 6 reach k = 5 without suppression and all of them drop the date of birth or the postcode
  to its sector; with 5% suppression, sector + decade.
- "Unique in the export" is what an insider needs and an upper bound on what a stranger needs; the two differ by a
  factor of 1,000 at a coarse key and not at all at a fine one.
- A linkage register that is incomplete does not fail quietly: it names the wrong person, confidently, 101 times.
- k says nothing about the diagnosis column; count the share of each class's commonest value before releasing.
""")

md("## 9. Try your own")

code("""# pop = population()
# export = sample(pop, 0.05)
# for levels in [("6 digits", "year", "kept"), ("sector (2 digits)", "decade", "kept"), ("4 digits", "dropped", "kept")]:
#     p = profile(export, pop, levels)
#     print(levels, "classes", p["classes"], "unique export", p["unique_export"], "unique town", p["unique_pop"],
#           "below k", p["below_k"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version measures your
own export's columns at any generalisation: `pip install -r requirements.txt && streamlit run app.py`.
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
