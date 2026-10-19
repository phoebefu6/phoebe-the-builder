"""Generate demo.ipynb.

Plumbing copied from `exploration-budget` (Day 201), content written fresh. The engine is EXTRACTED from explore.py
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
PATH = "data-quality-governance/email-validator"


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

EXPORTS = ["reference", "CORPUS", "SHARE", "VALIDATORS", "score", "chain", "study", "lines"]

HEADER = """from __future__ import annotations

# !pip install email-validator==2.3.0   # uncomment on Colab / Binder
import ipaddress
import re
import string
from email.utils import parseaddr
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
from email_validator import EmailNotValidError, validate_email


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
# The valid address was rejected: ten email validators against the RFC

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A customer called O'Brien cannot sign up, and a support ticket says the form called a real address invalid. This
notebook runs ten validators people actually paste into code against a reference parser for RFC 5321 (SMTP) and
RFC 6531 (international addresses), over 68 hand-labelled addresses.

1. The engine
2. The scoreboard
3. Which real customers each one turns away
4. The overlap: what every validator gets wrong together
5. The chart
6. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. `reference` sorts an address into **ok** (deliverable),
**grey** (valid by the grammar but local-only: `user@localhost`, an IP literal, a numeric TLD) or **bad**. Every
corpus row also carries a hand label, and the two agree on all 68. Regexes are applied the way they are pasted:
`re.match` on a `^...$` pattern. Nothing is random, so every number below matches `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
out = lines(r)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. The scoreboard

FR counts deliverable addresses rejected (of 36), FA invalid addresses accepted (of 27). `sign-ups lost` weights
each deliverable group by a **declared** share of real sign-ups (88% plain, 3% plus-tags, 1.04% international, and
so on; edit `SHARE`). The most-copied tutorial regex turns away 1.18%; OWASP's 2.16%, because it caps the TLD at
seven letters; an old `{2,4}` TLD regex 24.64%. The library, email-validator 2.3, is closest: 3 FR (all quoted local
parts, by policy) and 1 FA.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Which real customers each one turns away

The groups each validator refuses. Apostrophes (O'Brien, D'Angelo) and international addresses are the expensive
ones: real people, legal since 2012 (RFC 6531), and refused by five of ten.
""")

code("""print("\\n".join(section(2)))
print()
print("\\n".join(section(3)))""")

md("""
## 4. The overlap

Zero validators are error-free. A 65-octet local part passes all ten: the library only enforces RFC 5321's 64-octet
limit with `strict=True`. `john@example.com\\n` passes 9 of 10, because Python's `$` matches before a final newline:
a header-injection risk, not a typo. Stacking the browser's HTML5 check, the tutorial regex and the library rejects 15
deliverable addresses, 1.18% of sign-ups: a stack is as strict as its strictest member and as loose as nothing.
""")

code("""print("\\n".join(section(4)))
print()
print("\\n".join(section(5)))""")

md("""
## 5. The chart

The full figure, with the address taken apart, is `audit.png` (built by `make_chart.py`). Here, the two error counts.
""")

code("""names = list(VALIDATORS)
fr = [len(r["per"][n]["false_reject"]) for n in names]
fa = [len(r["per"][n]["false_accept"]) for n in names]
fig, ax = plt.subplots(figsize=(9, 4.6))
y = range(len(names))
ax.barh([i + 0.2 for i in y], fr, height=0.4, color="#c8562b", label="deliverable rejected (of 36)")
ax.barh([i - 0.2 for i in y], fa, height=0.4, color="#2f6fdb", label="invalid accepted (of 27)")
ax.set_yticks(list(y), names)
ax.invert_yaxis()
ax.set_title("Every validator errs; they differ in which way", loc="left")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- The copy-pasted regexes refuse real customers: apostrophes, international addresses, long TLDs, plus-tags.
- The permissive checks (`"@" in s`, `parseaddr`) refuse nobody and accept almost anything, including a display name.
- 9 of 10 accept a trailing newline. Strip and reject control characters before any regex.
- Validate syntax with a maintained library, then confirm the address by sending to it: syntax cannot prove a
  mailbox exists.
""")

md("## 6. Try your own")

code("""# for a in ["first.last@sub.example.travel", "\\"weird\\"@example.com", "x@[IPv6:::1]"]:
#     print(a, reference(a), {n: f(a) for n, f in VALIDATORS.items()})""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version checks your own
addresses and your own sign-up mix: `pip install -r requirements.txt && streamlit run app.py`.
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
