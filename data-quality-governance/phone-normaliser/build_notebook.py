"""Generate demo.ipynb.

Plumbing copied from `name-splitter` (Day 203), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

The only randomness is the seeded sweep (random.Random(204)), so the notebook reproduces evidence.txt to the
character.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/phone-normaliser"


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

EXPORTS = ["CORPUS", "NORMALISERS", "KEY_ONLY", "readings", "score", "sweep", "ceiling", "study", "lines"]

HEADER = """from __future__ import annotations

# !pip install phonenumbers==9.0.41   # uncomment on Colab / Binder
import random
import re
from collections import Counter
from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import phonenumbers as pn


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
# Two formats, one number, two customers: six phone normalisers against one CRM table

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

One customer types `020 7946 0958`, the same customer's second record says `+44 20 7946 0958`, and the CRM now has
two customers. A Spanish customer types `612 345 678` on a record with a French address, and the CRM files them
under a French customer's number. This notebook runs six ways of turning a typed phone number into a match key over
40 CRM rows belonging to 15 real customers, each row hand-labelled with its true E.164 number.

1. The engine
2. The scoreboard: splits and merges
3. Two customers made one
4. Valid, and the wrong number
5. How many numbers one string can be
6. The ceiling
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Every number is a fictional or documentation range
(Ofcom drama numbers, NANP 555-01xx, ACMA 0491 570 1xx, libphonenumber's own examples). The only randomness is the
seeded sweep in section 5, so every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. The scoreboard

A **false split** is one customer turned into two (same true number, different keys); a **false merge** is two
customers turned into one. `digits only` splits 37 of the 48 true duplicate pairs. `last 9 digits`, the usual fix,
cuts splits to 4 and makes 11 false merges. The library with the record's country field gets 33/40 right and still
merges 5 pairs, because the country field is the shipping address, not the phone's country. Only parsing with no
region at all makes zero errors, and it does that by refusing 28 of 40 rows.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Two customers made one

France and Spain share the nine digits `612345678` once the French trunk 0 is dropped, and Singapore and Hong Kong
share the eight digits `9123 4567`. Every key that ignores the country code collides on them.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Valid, and the wrong number

The library does not return an error on these rows: it returns a valid number that belongs to someone else. Under a
Singapore default region, `0033612345678` becomes a Hungarian number, because `00` is not the international prefix
everywhere.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. How many numbers one string can be

Parse each string under all 245 regions libphonenumber knows and count the distinct valid numbers. The digits the
Spanish customer typed are valid in 30 ways. In a seeded sweep of random strings typed without a country code, none
had a single reading: the median trunk-0 string is 21 valid numbers.
""")

code("""print("\\n".join(section(4)))
print()
print("\\n".join(section(5)))""")

md("""
## 6. The ceiling

Only 13/40 rows carry enough in the string itself to say which number they are. Adding the country field raises the
bound to 39/40, and no further: `9123 4567` on a Singapore-address record is a Singapore customer in one row and a
Hong Kong customer in another.
""")

code("""print("\\n".join(section(6)))""")

md("""
## 7. The chart

The full figure, with one string fanned out to its readings, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(NORMALISERS)
sp = [len(r["per"][n]["false_splits"]) for n in names]
mg = [len(r["per"][n]["false_merges"]) for n in names]
fig, ax = plt.subplots(figsize=(11, 4.4))
y = list(range(len(names)))
ax.barh([v - 0.2 for v in y], sp, height=0.38, color="#2f6fdb", label="false split")
ax.barh([v + 0.2 for v in y], mg, height=0.38, color="#c8562b", label="false merge")
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel(f"row pairs (of {r['per'][names[0]]['true_pairs']} true duplicate pairs)")
ax.set_title("Every normaliser errs; refusing is the only zero", loc="left")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- A phone string without a country code does not contain a number; a region hint assigns one.
- Keys that drop the country code merge customers across countries; keys that keep formatting split one customer.
- The record's country field is usually the shipping address: it fixes most rows and misfiles the rest silently.
- The fix is upstream: store E.164 with a + at capture (a country picker beside the field), and route the rest to a
  human instead of a default region.
""")

md("## 8. Try your own")

code("""# for s, c in [("0412 345 678", "AU"), ("0412 345 678", "FR"), ("(02) 9876 5432", "")]:
#     print(s, c, len(readings(s)), {n: f(s, c, "SG") for n, f in NORMALISERS.items()})""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version normalises your
own rows under your own default region: `pip install -r requirements.txt && streamlit run app.py`.
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
