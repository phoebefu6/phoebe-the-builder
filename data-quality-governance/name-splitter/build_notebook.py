"""Generate demo.ipynb.

Plumbing copied from `email-validator` (Day 202), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

There is no randomness in this build, so the notebook reproduces evidence.txt to the character.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/name-splitter"


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

EXPORTS = ["CORPUS", "SHARE", "SPLITTERS", "shape", "roles", "ceiling", "score", "study", "lines"]

HEADER = """from __future__ import annotations

# !pip install nameparser==2.5.0   # uncomment on Colab / Binder
from collections import Counter
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
from nameparser import HumanName


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
# First name came out as van: six name splitters against twelve naming conventions

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A CRM export writes `van den Broek, Pieter`, the import script splits on the first space, and the welcome email
opens "Dear van". This notebook runs six ways of splitting a full name into given + family over 51 fictional names
in 12 naming conventions, each hand-labelled from the person's own convention.

1. The engine
2. The scoreboard
3. Which conventions each splitter breaks
4. Dear Dr., Dear van: greetings that came out as a title or particle
5. The ceiling: what no splitter can fix
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Truth is two fields per row: **given** (the name you
greet them by) and **family** (the surname as filed, particles included; empty for mononyms and Malay bin/binti
patronymics). Nothing is random, so every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. The scoreboard

`both right` means given AND family exactly right. `greeted wrong` weights each convention by a **declared** share
of a customer table (51% plain, 12% middle names, 7% comma-inverted exports, and so on; edit `SHARE`). The library,
nameparser 2.5, ties for the most rows right (34 of 51) and greets the fewest wrong, 9.15%. `rsplit(" ", 1)`, "the
surname is the last word", greets 38.50% wrong.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Which conventions each splitter breaks

Every splitter gets plain, hyphenated and mononym names right. They part ways on everything else. Adding a list of
East Asian surnames fixes family-first names (83%) and breaks the names that were already turned round into Western
order (33%): it trades one group for another.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Dear Dr., Dear van

The cheap splitters greet a titled customer as "Dr." and an inverted export as "van"; `rsplit` greets Pieter as
"Pieter van den". Six names defeat every splitter here: two Iberian double given names, a Hungarian family-first
name with no list entry, and three Malay patronymics, where `bin Ahmad` is the father, not a surname.
""")

code("""print("\\n".join(section(3)))
print()
print("\\n".join(section(4)))""")

md("""
## 5. The ceiling

Two bounds on anything that reads only the string. **By string**: `Chen Lin` is in the corpus twice, once
family-first and once given-first, so 50/51 is the maximum. **By shape**: a rule over word positions cannot tell
`Thomas Edward Barlow` (middle name) from `Carlos Ruiz Delgado` or `Rupert Ashby Lowe` (two-word surname); 41/51 is
the most such a rule can get. The best splitter here reaches 34.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 6. The chart

The full figure, with the same-shape names drawn word by word, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(SPLITTERS)
both = [r["per"][n]["both_right"] for n in names]
greet = [r["per"][n]["greeted_wrong"] * 100 for n in names]
fig, ax = plt.subplots(figsize=(11, 4.4))
y = range(len(names))
ax.barh(list(y), both, color="#2f6fdb", height=0.55, label=f"names split right (of {r['n']})")
ax.axvline(r["ceiling"]["by_shape"], color="#c8562b", ls="--", label="shape ceiling")
ax.axvline(r["ceiling"]["by_string"], color="#c8562b", ls=":", label="string ceiling")
ax.set_yticks(list(y), [f"{n}  ({g:.1f}% greeted wrong)" for n, g in zip(names, greet)])
ax.invert_yaxis()
ax.set_xlim(0, 55)
ax.set_title("No splitter reaches what the string allows", loc="left")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- A full-name field does not contain a split; a convention assigns one, and the string does not say which.
- The library (nameparser) beats every one-liner, and still files 16 of 51 surnames wrong.
- Rule patches trade groups: a surname list fixes family-first names and breaks Western-ordered ones.
- The fix is upstream: capture given and family in separate fields, plus "what should we call you".
""")

md("## 7. Try your own")

code("""# for s in ["Maria de Lourdes Souza", "Okonkwo, Chidi", "Le Thi Hoa"]:
#     print(s, {n: f(s) for n, f in SPLITTERS.items()})""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version splits your own
names and your own customer mix: `pip install -r requirements.txt && streamlit run app.py`.
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
