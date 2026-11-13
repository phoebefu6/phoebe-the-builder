"""Generate demo.ipynb.

Plumbing copied from `float-roundtrip` (Day 213), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one.

Everything in this build is seeded (Generator(214)), so the notebook reproduces evidence.txt to the character.
Every value is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/json-number-precision"


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

EXPORTS = ["make_ids", "ndjson", "READERS", "FIXES", "CONTEXTS", "parse", "study", "seeds", "lines", "GROUPS"]

HEADER = """from __future__ import annotations

import io
import json
import os
import tempfile
import warnings
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


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
# The ID lost its last digits: which JSON readers give a 64-bit ID back exactly

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A float64 holds every integer up to 2**53 and then starts skipping. Database keys, snowflake IDs and hashes are
64-bit, so any JSON reader that turns them into a double hands back a different number with no error. A single ID
usually survives; what breaks it is the column, because readers pick one type per column and one odd row decides
it. This notebook sends three groups of fictional IDs through 9 readers in five column contexts.

1. The engine
2. Which reader, which column
3. The column type still says integer
4. What a wrong ID looks like
5. Where each reader starts rounding
6. The documented fix
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Generator(214)`), every ID fictional, and
every number below matches `evidence.txt` exactly. Section 6 of the evidence file (real Node and jq binaries) is not
rerun here; the `JSON.parse (double)` reader is the Python equivalent, and the evidence file shows the real Node
JSON.parse and jq 1.6 give identical values.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
out = lines(r, s, {})
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Which reader, which column

Only 3 of 9 readers give every ID back in every context: json, ujson and pydantic. Small IDs below 2**31 are never
wrong, so a fixture of ordinary database keys passes every reader. A null in the column turns pandas to float64; a
single `123.0` does the same to pandas, pyarrow and duckdb; one ID above 2**63 does it to pyarrow. 14 of 45
reader-contexts change IDs with no error and no warning.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. The column type still says integer

pandas reads the column through a double and casts it back, so `df.dtypes` reports int64 while 1,960 of 2,000
snowflake IDs are wrong. Asking for `dtype={'id': 'Int64'}` - the usual advice - casts after the damage and fixes
nothing. A schema check on the dtype passes.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. What a wrong ID looks like

Past 2**53 every other integer is skipped, so on the counter crossing 2**53, 499 IDs merge with a neighbour: two
customers become one key. Snowflake IDs sit near 2**60, where doubles are 256 apart, so almost every ID is off by up
to 63 but only a few collide - the join misses instead of merging.
""")

code("""print("\\n".join(section(3)))
print()
print("\\n".join(section(7)))""")

md("""
## 5. Where each reader starts rounding

A clean column only breaks for JSON.parse (at 2**53 + 1), pyarrow (2**63 + 1) and duckdb (2**64 + 1). pandas refuses
beyond 2**64 instead. Testing the reader on a clean file tells you almost nothing about production files.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The documented fix

`dtype=str` in pandas is worse than nothing: with a null in the column it hands back strings like
`2.0640592815436145e+18`. pyarrow's explicit int64 schema is correct and loud - it refuses the file. duckdb with
`columns={'id': 'HUGEINT'}` and parsing with `json` before building the DataFrame are exact in every context.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The two-panel figure is `precision.png` (built by `make_chart.py`). This is the boundary group on its own: which
IDs a double keeps.
""")

code("""ids = make_ids()["boundary"]
off = [int(float(i)) - i for i in ids]
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.scatter([i - 2**53 for i in ids], off, s=6, color=["#c0392b" if o else "#7fb38f" for o in off])
ax.axvline(0, color="#6b7684", lw=0.8)
ax.set_xlabel("ID minus 2**53", fontsize=9)
ax.set_ylabel("double minus ID", fontsize=9)
ax.set_title("Past 2**53 every odd ID comes back one away (red)", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- One row decides the column type. Test the reader with a null, a `123.0` and an oversized value in the column.
- pandas `read_json` can report int64 while the values are wrong; `dtype='Int64'` and `dtype=str` do not fix it.
- Exact everywhere: `json`, `ujson`, pydantic, duckdb with an explicit HUGEINT column, `json` then a DataFrame.
- In JavaScript, send IDs as strings or use a reviver that reads `context.source` into BigInt.
""")

md("## 8. Try your own")

code("""# my_id = 1801234567890123457                  # paste one of your IDs
# for name in READERS:
#     d = parse(name, [my_id], "one null")         # or "clean", "one 123.0", "one > 2**63", "one > 2**64"
#     print(f"{name:<21} {'raised' if d['raised'] else 'exact' if not d['wrong'] else 'WRONG'}")""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version lets you type an
ID and pick the odd row: `pip install -r requirements.txt && streamlit run app.py`.
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
