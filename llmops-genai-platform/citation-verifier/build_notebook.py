"""Generate demo.ipynb.

Plumbing copied from `agent-trajectory-eval` (Day 189), content written fresh. The engine is EXTRACTED from
citation.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Every checker number is exact (every claim the generator can emit is enumerated), so the notebook reproduces
them to the digit. Only the Monte Carlo calibration runs at NB_REPS instead of the study's 200,000 - stated in
the notebook (the Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "llmops-genai-platform/citation-verifier"
NB_REPS = 20_000


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER.

    Tuple targets are walked, because an extractor reading only `targets[0].id` drops them
    silently - the Day 178 defect.
    """
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("citation.py")

EXPORTS = [
    "SEED", "MC_REPS", "TAU", "FACTS", "EDITS", "CITES", "CHECKERS", "MODELS", "STOP", "PASSAGES", "TUPLES",
    "render", "sibling", "edit_claim", "cited_index", "coverage", "numbers", "negations", "check", "build",
    "enumerate_claims", "summarise", "evaluate", "threshold_sweep", "flip_vs_paraphrase_order", "simulate",
    "wilson", "calibrate", "parse_claims", "audit",
]

HEADER = """from __future__ import annotations

import functools
import itertools
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
    cells.append({"cell_type": "code", "id": _nid(), "execution_count": None,
                  "metadata": {}, "outputs": [], "source": _lines(src)})


# ---------------------------------------------------------------------------

md(f"""
# The citation does not say that

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A RAG answer cites its sources with [n]. This notebook declares a small corpus and a claim generator, enumerates
every cited claim the generator can emit, and measures - exactly - how often the cited passage does not say
what the claim says, and what five text-only checkers (citation present, context-level lexical, cited-passage
lexical, plus number and negation checks) report instead.

1. The corpus and the generator
2. Calibration - exact enumeration vs raw simulation
3. Five checkers, two model versions
4. The "always cite your sources" instruction, as each checker reports it
5. The threshold that cannot separate a polarity flip from a paraphrase
6. Try your own
""")

md("""
## The engine

Extracted from `citation.py` at build time, character for character. A claim is one true fact, one of six edits
(none, polarity, scope, a neighbour's number, a novel number, the other plan's entity), one of four citation
choices (the source, the neighbouring passage, an unrelated one, none), verbatim or paraphrased. SUPPORTED means
the cited passage states the same entity, attribute, value, scope and polarity.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every checker number here is exact and matches `evidence.txt` to the digit.
Only the calibration's raw simulation runs at {NB_REPS:,} replicates instead of the study's 200,000.
""")

md("## 1. The corpus and the generator")

code("""
for i, p in enumerate(PASSAGES):
    print(f"[{i + 1:>2}] {p}")
r = build(5, "polarity", "source", False)
print()
print(r["claim"], "->", "supported" if r["supported"] else "NOT supported", "| coverage", round(r["cov"], 2))
print("passes:", [k for k, v in r["checks"].items() if v])
""")

md("""
The last lines are the whole problem in one row: a polarity flip has coverage 1.00 against its source, because
the stopword list drops "not". Only the negation count sees it.

## 2. Calibration
""")

code(f"""
cal = calibrate({NB_REPS})
print(f"inside their 99% interval: {{sum(r['inside_99'] for r in cal)}} of {{len(cal)}}")
""")

md("## 3. Five checkers, two model versions")

code("""
ev = evaluate()
for name, s in ev.items():
    print(f"{name}: supported by its citation {s['p_supported']:.1%}, claim true {s['p_true']:.1%}, "
          f"carries a citation {s['p_cited']:.1%}")
    for k, v in s["scorers"].items():
        print(f"   {k:<32} reports {v['reported']:6.1%}  passes bad {v['false_pass']:6.1%}  "
              f"fails good {v['false_fail']:6.1%}  bad among passes {v['bad_among_passes']:6.1%}")
""")

md("""
Half of v1's claims are supported by the passage they cite, though 81% of them are true: the rest are true and
uncited, or true and pointing at the wrong passage. The context-level check (is the vocabulary anywhere in the
retrieved context) passes 98% - every edit here reuses context vocabulary, including the neighbour's number.
The sharpest lexical rule still passes a widened scope and the other plan's fact, because those share every
content word with the passage.

## 4. The instruction change
""")

code("""
v1, v2 = ev["v1 cites when sure"], ev["v2 always cite"]
names = ["truly supported"] + list(CHECKERS)
d = [v2["p_supported"] - v1["p_supported"]] + [v2["scorers"][k]["reported"] - v1["scorers"][k]["reported"] for k in CHECKERS]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(names[::-1], [x * 100 for x in d][::-1], color=["#7a4fb5", "#2f6fdb", "#e3b23c", "#8a6d3b", "#8a94a3", "#1f2733"])
ax.set_xlabel("change in reported pass rate, v1 -> v2 (points)")
ax.set_title("'Always cite your sources'", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
for n, x in zip(names, d):
    print(f"{n:<32} {x * 100:+6.1f} pts")
""")

md("""
Citations went from 74% to 98% of claims and support fell 8 points: the new citations point at the neighbouring
passage (same attribute, other plan). Citation coverage reports +24. The cited-passage lexical check reports
+11, because the neighbour shares most of the claim's words. Only the rules that check numbers see the sign.

## 5. The threshold
""")

code("""
m = MODELS["v1 cites when sure"]
for r in threshold_sweep(m, [0.3, 0.5, 0.6, 0.8, 0.9, 1.0]):
    print(f"tau {r['tau']:.2f}  fails a supported paraphrase {r['fails_supported_paraphrase']:6.1%}   "
          f"passes a polarity flip {r['passes_polarity_flip']:6.1%}")
o = flip_vs_paraphrase_order(m)
print(f"flip scores above the paraphrase in {o['flip_above']:.1%} of pairs, level in {o['tied']:.1%}")
""")

md("""
Raising the threshold fails honest paraphrases before it fails flips. The flip is a closer lexical match than the
paraphrase in 79% of pairs, so no threshold on this score separates them.

## Summary

| claim | what the numbers say |
|---|---|
| "every sentence has a citation now" | coverage +24 points, support -8: the new citations name the neighbour |
| "the cited passage shares the words" | +11 points for a -8 change; passes 66% of v2's unsupported claims |
| "check the numbers too" | tracks the sign; still passes a widened scope and the other plan's fact |
| "raise the lexical threshold" | fails paraphrases first - a flip is lexically closer than a paraphrase |
| "it is in the context" | a different question: 98% of claims, including the misattributed ones |
""")

md("## 6. Try your own")

code("""
# passages, claims = parse_claims(
#     "[1] Pro plan includes 5 seats.\\n[2] Basic plan includes 2 seats.",
#     "Pro plan comes with 5 seats. [1]\\nPro plan includes 2 seats. [1]\\nBasic plan does not include 2 seats. [2]")
# for r in audit(passages, claims):
#     print(r["claim"], [k for k, v in r["checks"].items() if v], r["notes"])
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own passages and claims:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`hallucination-checker`](../hallucination-checker) asks whether a claim is anywhere in the
retrieved context - the `context_lexical` baseline here; [`agent-trajectory-eval`](../agent-trajectory-eval) is the
previous day's audit of a scorer that sat still while quality moved.
""")

def write() -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    with open("demo.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote demo.ipynb with {len(cells)} cells "
          f"({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
