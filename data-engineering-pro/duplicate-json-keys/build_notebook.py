"""Generate demo.ipynb.

Plumbing copied from `json-number-precision` (Day 214), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one.

Everything in this build is seeded (Generator(215)), so the notebook reproduces evidence.txt to the character.
Every value is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-engineering-pro/duplicate-json-keys"


def extract(path: str) -> str:
    """Every top-level def, assignment, if-block and call, in source order (docstring dropped). Imports are left to HEADER."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.If, ast.Expr)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start : node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("explore.py")
EVIDENCE = extract("evidence.py")

EXPORTS = ["READERS", "FIXTURES", "classify", "matrix", "make_feed", "feed_study", "detect", "study", "seeds",
           "pairs", "lines"]

HEADER = """from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from itertools import combinations
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
# Which value wins? One JSON object with a repeated key, through 14 readers

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

`{{"a":1,"a":2}}` is legal JSON. RFC 8259 only says names SHOULD be unique, and that a reader of such an object may
keep the last value or report an error. So every reader here follows the spec, and they disagree. That matters the
moment two systems read the same payload: a validator that sees the first value and a consumer that acts on the
last one.

1. The engine
2. Which value each reader returns
3. A refund feed patched by string concatenation
4. Catching it before it loads
5. One engine, three entry points
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Generator(215)`), every value fictional.
Node and jq join only when their binaries exist on the machine running this; the evidence file was made with both.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
out = lines(r, s)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Which value each reader returns

Three camps on the plain case: last wins (json, ujson, pydantic, PyYAML, Node, jq, pandas), first wins (polars,
duckdb's `json_extract`, duckdb's `read_json` with `ignore_errors=true`), refuse (json with a strict
`object_pairs_hook`, ruamel.yaml, pyarrow, duckdb's `read_json`). polars is not even consistent with itself: it keeps
the first value at the top level, refuses a nested duplicate, and turns `1` then `"x"` into the string `'1'`. The last
column is not a duplicate at all - `A` and `a` are different names - and duckdb still returns the wrong one.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. A refund feed patched by string concatenation

A refund job appends `,"status":"refunded"` to the order line instead of rewriting it. Last-wins readers see every
refund. First-wins readers see none of them and say nothing. Strict readers refuse all 2,000 rows because of 46.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Catching it before it loads

A grep for a repeated quoted key is the usual pre-flight check. It cannot see `"st\\u0061tus"`, which every reader
decodes to `status`. Parsing with a strict pairs hook sees the decoded keys and catches every one.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. One engine, three entry points

The same duckdb process refuses the file, keeps the first value with `ignore_errors=true`, and keeps the first value
again through `json_extract`. Turning on `ignore_errors` to get a load through is how a refusal becomes a silent
first-wins.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The chart

The two-panel figure is `dupkeys.png` (built by `make_chart.py`). This is the feed alone: refunds each reader saw.
""")

code("""fd = r["feed"]
names = [k for k in fd if k != "_truth"]
vals = [fd[k]["refunds"] for k in names]
col = ["#f2c94c" if fd[k]["raised"] else "#7fb38f" if v == fd["_truth"]["refunds"] else "#c0392b"
       for k, v in zip(names, vals)]
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh(names, vals, color=col)
ax.invert_yaxis()
ax.set_xlabel("refunds seen (yellow = refused the file)", fontsize=9)
ax.set_title(f"{fd['_truth']['refunds']} refunds in the feed: first-wins readers report none", loc="left",
             fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Duplicate keys are legal JSON; last-wins, first-wins and refusal are all spec-compliant, and they disagree.
- If any system in the path uses a different camp, a validator and a consumer can act on different values.
- Detect with a parser (`object_pairs_hook` that raises), never with a regex: escaped keys decode to the same name.
- Do not turn on `ignore_errors` to get a refused file through - in duckdb it switches to keeping the first value.
""")

md("## 7. Try your own")

code("""# my_doc = '{"user":"ana","role":"viewer","role":"admin"}'
# for name, fn in READERS.items():
#     try:
#         print(f"{name:<22} {fn([my_doc], ['role'])[0]!r}")
#     except Exception as e:
#         print(f"{name:<22} refused ({type(e).__name__})")""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version lets you paste
a document and see every reader's answer: `pip install -r requirements.txt && streamlit run app.py`.
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
