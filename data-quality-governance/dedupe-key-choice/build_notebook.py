"""Generate demo.ipynb.

Plumbing copied from `phone-normaliser` (Day 204), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

There is no randomness anywhere in this build, so the notebook reproduces evidence.txt to the character.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/dedupe-key-choice"


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

EXPORTS = ["CORPUS", "ROW_KEYS", "PAIR_KEYS", "NAMES", "clusters", "score", "frontier", "ceiling", "study", "lines"]

HEADER = """from __future__ import annotations

# !pip install jellyfish==1.0.1 rapidfuzz==3.14.5   # uncomment on Colab / Binder
import re
import unicodedata
from itertools import combinations
from typing import Callable, Dict, List, Set, Tuple

import jellyfish as jf
import matplotlib.pyplot as plt
from rapidfuzz import fuzz


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
# Same customer, three rows: six match keys against one customer table

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

`Jonathan Smith`, `Smith, Jonathan` and `Jonathon Smith` are one customer. `Johnathan Smith` and the two
`John Smith` rows are three other people. Every dedupe script starts by choosing a KEY that decides which of those
rows are the same, and this notebook is not a deduper (see `dedup-pipeline`, Day 67, for merging and survivorship):
it runs six keys - exact, normalised, Soundex, Metaphone, Jaro-Winkler at 0.9, token-set ratio at 85 - over 45 CRM
rows belonging to 23 real customers, each row hand-labelled with who it is, and reports which pairs each key merges
that the others keep apart.

1. The engine
2. The scoreboard: splits and merges
3. What each key merges that the others do not
4. Two customers made one
5. Chained: merged without ever matching
6. The frontier
7. The ceiling
8. The chart
9. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Every name is fictional. There is no randomness, so
every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. The scoreboard

A **false split** is one customer turned into two (same person, different keys); a **false merge** is two customers
turned into one. The exact string splits all 37 true pairs and still merges one: the two different `John Smith`s.
Normalising (case, accents, punctuation, title, token order) closes 10 of those pairs and leaves 36 customers for
23 people. The phonetic keys and the fuzzy keys close more and open merges: Jaro-Winkler at 0.9 leaves 19 customers,
fewer than the truth, with 19 pairs of different people joined.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. What each key merges that the others do not

Row key merges, column key keeps apart. Exact is the finest key: nothing it joins is kept apart by any other, so
its row is all zeros. Soundex and Metaphone disagree on 3 pairs between them; the token-set ratio joins 45 pairs the
exact string keeps apart.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Two customers made one

`Jon Smith` is one customer's short form, and Soundex codes `jon` and `john` both as J500, so every phonetic and
fuzzy key files him with both `John Smith`s. `Anna Müller`, a different Anna, is joined to `Anna Müller-Schmidt`
by both fuzzy keys.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. Chained: merged without ever matching

A fuzzy key is pairwise, and the table is then clustered by connected components, which every fuzzy dedupe script
does whether it says so or not. `Anna Schmidt` and `Anna Müller` score 53 on the token-set ratio, far below 85, and
end up in one cluster because `Anna Müller-Schmidt` matched each of them. Judged pair by pair the token-set key
makes 11 splits and 15 merges; clustered it makes 9 and 18.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. The frontier

Every threshold the similarities take, judged pair by pair. The best single Jaro-Winkler threshold is 0.969 with
22 splits and 5 merges; the best token-set threshold is 96.552 with 20 and 3. No threshold reaches zero of both,
because a one-letter typo in one person's name scores the same as a one-letter difference between two people.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The ceiling

The two `John Smith`s are the same string, so every key merges them. And 7 of the 37 true pairs are joined by no key
here at all: `William Jones` and `Bill Jones`, `Mohammed Al-Sayed` and `Mohamed Alsayed`. A nickname and a
transliteration are not spelling variants; nothing computed from the string recovers them.
""")

code("""print("\\n".join(section(6)))""")

md("""
## 8. The chart

The full figure, with one name fanned out to its six keys, is `audit.png` (built by `make_chart.py`).
""")

code("""names = list(NAMES)
sp = [len(r["per"][n]["false_splits"]) for n in names]
mg = [len(r["per"][n]["false_merges"]) for n in names]
fig, ax = plt.subplots(figsize=(11, 4.4))
y = list(range(len(names)))
ax.barh([v - 0.2 for v in y], sp, height=0.38, color="#2f6fdb", label="false split")
ax.barh([v + 0.2 for v in y], mg, height=0.38, color="#c8562b", label="false merge")
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel(f"row pairs (of {r['ceiling']['true_pairs']} true duplicate pairs)")
ax.set_title("Every key errs; the fuzzy ones trade splits for merges", loc="left")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- The key is not a preprocessing step: it decides how many customers the table has (44 to 19 here, truth 23).
- Exact and normalised keys only split; phonetic and fuzzy keys buy fewer splits with merges of different people.
- A fuzzy threshold is judged pair by pair, but the table is clustered, and clusters join rows that never matched.
- No threshold reaches zero errors, and 7 of 37 true pairs are beyond every key: label a sample, count both errors,
  and route the ambiguous band to a person, as `dedup-pipeline` does after this choice is made.
""")

md("## 9. Try your own")

code("""# for a, b in [("Jonathan Smith", "Jon Smith"), ("Catherine Lee", "Kathryn Lee"), ("Sam Ng", "Sam Ngo")]:
#     print(a, "|", b, {n: f(a) == f(b) for n, f in ROW_KEYS.items()},
#           {n: round(sim(a, b), 3) for n, (sim, t) in PAIR_KEYS.items()})""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version keys your own
names under all six schemes: `pip install -r requirements.txt && streamlit run app.py`.
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
