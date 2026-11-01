"""Generate demo.ipynb.

Plumbing copied from `k-anonymity-check` (Day 206), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Random(207)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/table-checksum"


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

EXPORTS = ["make_table", "SCHEMES", "BENIGN", "MUTATIONS", "canonical", "money", "study", "lines", "tally"]

HEADER = """from __future__ import annotations

import csv
import hashlib
import io
import random
from collections import Counter
from decimal import ROUND_HALF_EVEN, Decimal
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
# Are these two tables the same? Five whole-table fingerprints against what happens in transit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

The table was copied to another system and you cannot ship both copies to one place to diff them. You can ship one
number from each side. This notebook builds a fictional 1,000-row keyless order-event export (20 exact duplicate
pairs, as event logs have), then asks five fingerprints two questions: after a benign transform (same data), do they
MATCH; after a real mutation (different data), do they DIFFER. It is not a row differ (see `data-diff`, Day 100)
or a reconciliation report (`reconciliation-checker`, Day 99); it scores the one number you compare.

1. The engine
2. Same data: which fingerprints raise a false alarm
3. Different data: which fingerprints miss it
4. The score
5. The ceiling: a half cent
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Random(207)`), every row fictional, and every
number below matches `evidence.txt` exactly. The five fingerprints: `csv_md5` (md5 of the dump), `count_sums` (row
count + money column sums, the reconciliation habit), `xor_rows` and `sum_rows` (XOR or sum mod 2^64 of a 64-bit
hash of each row joined with `|`), and `canonical` (the sum, over a typed row: columns by name, money quantised to the
cent, NULL a distinct token, strings length-prefixed, -0 folded to 0). The study takes several seconds.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
r["tally"] = tally(r)
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Same data: which fingerprints raise a false alarm

The dump hash fails all 7: shuffling rows alone breaks it. The naive row hashes survive a shuffle and nothing else:
a column reorder, a CSV round trip, a 15-digit export, -0.0 and 5.0-stored-as-5 each change the string that gets
hashed. Column sums and canonical survive all of those. Every fingerprint fails the re-add, for a reason section 5
explains.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Different data: which fingerprints miss it

Column sums miss 4 of 6: two amounts swapped between rows, two offsetting cents (caught in 1 trial of 100), a NULL
that became an empty string, and a `|` moved across a column boundary. XOR misses the duplicate swap - a dedup step
dropped both copies of one retried event and a retry wrote another twice, and a pair XORs to zero. The `|`-joined
row hash cannot tell `Ada Lim|` + `gift` from `Ada Lim` + `|gift`. The dump hash misses NULL-to-empty, because the
CSV writer writes both as nothing.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. The score

Canonical: 1 false alarm of 7, 0 misses of 6. Column sums: 1 false alarm, 4 misses. Every other fingerprint false-alarms
on at least 6 of 7 benign transforms, which in practice means somebody stops looking at it.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. The ceiling: a half cent

The target re-adds the same line items in the opposite order. 259 of 1,000 float amounts differ in the bits. Rounding
to the cent folds all of them back but one: 13 true amounts sit exactly on a half cent, and for event 10459 the
forward sum lands on 821.225 (rounds to 821.22) while the reverse sum lands one ulp above (rounds to 821.23). Any
fingerprint that rounds at compare time has a boundary, and a value either side of it splits. The fix is not a better
hash: store money at its declared scale (NUMERIC(12,2)) at load, so both sides round once, in the same place.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The chart

The full figure, with the half-cent drawn, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(r["scores"]["benign"]) + list(r["scores"]["mutations"])
grid = [[(v["match"][k] if g == "benign" else 1 - v["match"][k]) for k in SCHEMES]
        for g in ("benign", "mutations") for v in r["scores"][g].values()]
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SCHEMES)), list(SCHEMES))
ax.set_yticks(range(len(names)), names)
ax.axhline(len(r["scores"]["benign"]) - 0.5, color="black", lw=1.5)
ax.set_title("Share of trials each fingerprint answered right (top: same data, bottom: different)", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- A dump hash is a row-order test, not a data test: 7 false alarms of 7.
- Column sums are quiet and blind: 4 of 6 real changes pass, including two amounts swapped between rows.
- XOR cancels pairs, and a keyless table has pairs; a `|`-join cannot see a `|` in the data.
- Canonical, typed, summed row hashes got 12 of 13 right, and needed the schema declared to do it.
- The thirteenth is a half cent: round money once at load, not at compare time.
""")

md("## 7. Try your own")

code("""# t, items = make_table()
# mine = (list(t[0]), [dict(r) for r in t[1]])
# mine[1][0]["note"] = "changed"
# for k, f in SCHEMES.items():
#     print(k, "match" if f(mine) == f(t) else "differ")""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version compares your
own two tables pasted as CSV: `pip install -r requirements.txt && streamlit run app.py`.
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
