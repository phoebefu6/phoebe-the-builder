"""Generate demo.ipynb.

Plumbing copied from `fanout-join-audit` (Day 212), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one.

Everything in this build is seeded (Generator(213)), so the notebook reproduces evidence.txt to the character.
Every value is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/float-roundtrip"


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

EXPORTS = ["make_values", "WRITERS", "WHAT", "compare", "bits", "run", "ledger", "study", "seeds", "lines", "GROUPS", "EDGE"]

HEADER = """from __future__ import annotations

import csv
import io
import json
import math
import pickle
import sqlite3
import struct
from typing import Callable, Dict, List, Optional

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
# The number changed after a save: which writers give a float64 back bit for bit

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A float64 has 17 significant digits. Many ways of saving it write fewer, and some readers misread a file that was
written correctly. Nothing errors: the value comes back a few bits different, and every join or dedupe key on that
column stops matching. This notebook sends four groups of fictional values through 15 writers and compares what
comes back bit for bit.

1. The engine
2. Exact, changed, lost
3. Why a test fixture misses it
4. The total does not move; the keys do
5. Edge values
6. pandas: the file is exact, the reader is not
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Generator(213)`), every value fictional, and
every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
out = lines(r, s)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Exact, changed, lost

Seven of 15 writers give every value back bit for bit: repr, %.17g, pickle, np.savetxt, json, the csv module and
pandas with `float_precision='round_trip'`. Writing 15 significant digits changes 1318 of 2000 computed values, and
SQLite's text cast does exactly the same thing.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Why a test fixture misses it

Typed amounts with two decimals have a short exact decimal form, so 14 of 15 writers give them back. A fixture of
typed amounts passes every writer except float32. The same writers fail on the amounts a pipeline actually holds:
after an FX rate and a split.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. The total does not move; the keys do

The ledger total over 2000 computed amounts is unchanged to the last bit after a 15-digit round trip, because the
rounding errors are far below a cent and cancel. A reconciliation on the total passes. Two-thirds of the values no
longer equal their source, so an exact join or a dedupe key on that column misses them.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. Edge values

SQLite stores NaN as NULL and -0.0 as 0.0, even in a REAL column. Strict JSON refuses NaN and infinity, which is
correct and still loses the value if the caller swallows the error. Fifteen digits turn the largest double into
infinity.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. pandas: the file is exact, the reader is not

`to_csv` writes the shortest exact string. The default `read_csv` parser then misreads some of them, by up to
thousands of ULP on small values. `float_precision='round_trip'` reads every one back exactly.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The two-panel figure is `roundtrip.png` (built by `make_chart.py`). This is the computed column on its own.
""")

code("""wr = r["writers"]
names = list(WRITERS)[::-1]
vals = [100 * (wr[k]["computed"]["changed"] + wr[k]["computed"]["lost"]) / N for k in names]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(range(len(names)), vals, color=["#1f4e79" if v else "#a9c8e4" for v in vals])
ax.set_yticks(range(len(names)), names, fontsize=8)
for i, v in enumerate(vals):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlim(0, 112)
ax.set_xlabel("% of computed values that came back with different bits", fontsize=9)
ax.set_title("After an FX rate and a split, 6 of 15 writers change the bits", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Write floats with repr, %.17g or a binary format. Fifteen significant digits is a display format, not a storage one.
- SQLite's `CAST(v AS TEXT)` is a 15-digit format; a REAL column also drops NaN and the sign of zero.
- pandas writes CSV exactly; read it with `float_precision='round_trip'`.
- Test with computed values, not typed ones: a two-decimal fixture passes every writer but float32.
- A reconciliation on the total passes while the join keys break. Check equality on the column, not the sum.
""")

md("## 8. Try your own")

code("""# x = 1234.5678 * 0.731 / 3                     # any value you compute in a pipeline
# for name, fn in WRITERS.items():
#     y = fn([x])[0]
#     print(f"{name:<18} {y!r:<26} {'exact' if compare([x], [y])['exact'] else 'CHANGED'}")""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version lets you type a
value and pick a writer: `pip install -r requirements.txt && streamlit run app.py`.
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
