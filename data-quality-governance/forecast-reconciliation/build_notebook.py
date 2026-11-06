"""Generate demo.ipynb.

Plumbing copied from `hierarchy-reconciliation` (Day 210), content written fresh. The engine is EXTRACTED from explore.py
by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Everything in this build is seeded (Generator(211)), so the notebook reproduces evidence.txt to the character.
There is no randomness beyond that seed; every store is fictional.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/forecast-reconciliation"


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

EXPORTS = ["make_world", "base_forecasts", "reconcile", "run", "study", "seeds", "shock_study", "lines", "METHODS", "LEVELS"]

HEADER = """from __future__ import annotations

import json
from typing import Dict, List

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
# The regions do not sum to the forecast: seven ways to make them add up, scored

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Finance has a national forecast, each region has its own, each store has its own, and they do not add up. The usual
fix is to pick one level and force the others to match it. This notebook builds a fictional retailer (4 regions,
12 stores), forecasts every series on its own, makes the numbers add up seven ways, and scores each way at every
level against the 12 months that actually happened.

1. The engine
2. Why the numbers do not add up
3. Seven ways to make them add up
4. Which method wins where
5. The check that passes for the worst method
6. What no reconciliation can fix
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` and `evidence.py` at build time. Seeded (`Generator(211)`), every store fictional, and
every number below matches `evidence.txt` exactly. `S` is the summing matrix: it turns 12 store numbers into all 17.
Reconciliation is one line, `S @ G @ yhat`; the methods differ only in `G`.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nE = _E()\nE.__dict__.update(globals())\n\n" + EVIDENCE)

code("""r = study()
s = seeds()
sh = shock_study()
out = lines(r, s, sh)
section = lambda n: [l for l in "\\n".join(out).split(f"\\n{n}. ")[1].split("\\n\\n")[0].splitlines()]
print(out[0])""")

md("""
## 2. Why the numbers do not add up

A linear model with the same inputs on every series adds up by construction: sums pass straight through it. The gap
appears as soon as any level is modelled differently. Here every series is fitted on the log scale (growth is
multiplicative), and the regions miss the national forecast by up to 1.75%, the stores by 2.63%.
""")

code("""print("\\n".join(section(1)))""")

md("""
## 3. Seven ways to make them add up

Every reconciled method adds up exactly. Accuracy is another matter. Splitting the national number by historical
shares makes the region forecasts +77.6% worse on average, because the store mix moved. Least-squares methods
(WLS, MinT) gain most at the top, a few percent, and give back a little at the stores. No method beats the
unreconciled forecasts at all three levels in more than 2 of 30 worlds.
""")

code("""print("\\n".join(section(2)))""")

md("""
## 4. Which method wins where

The textbook answer is MinT. In these worlds bottom-up has the lowest national error in 23 of 30, because the
national model is the weaker forecast: stores grow at different rates, their sum is not one exponential, and the
national log model under-forecasts. The best method depends on which level your forecasts are good at, and that is
an empirical question for your own backtest.
""")

code("""print("\\n".join(section(3)))""")

md("""
## 5. The check that passes for the worst method

"Do the regions sum to the forecast?" is the check every planning pack runs. It passes for every reconciled method,
including the one that doubles the region error in this world. It tests arithmetic, not accuracy.
""")

code("""print("\\n".join(section(4)))""")

md("""
## 6. What no reconciliation can fix

This is the negative result. One region loses 15% of its sales in the scored year, a cut no history contains. Every
method over-forecasts it by roughly the same amount, because reconciliation only redistributes the forecasts it is
given. The numbers add up and they are all wrong.
""")

code("""print("\\n".join(section(5)))""")

md("""
## 7. The chart

The full figure, with the three reconciliation directions drawn, is `audit.png` (built by `make_chart.py`).
""")

code("""names = [m for m in METHODS if m != "base"][::-1]
cols = {"total": "#1f4e79", "region": "#4f8fc0", "store": "#a9c8e4"}
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, m in enumerate(names):
    for j, lv in enumerate(LEVELS):
        v = 100 * s["mean"][m][lv]
        ax.barh(i + 0.25 - 0.25 * j, max(min(v, 25), -25), height=0.22, color=cols[lv], label=lv if i == 0 else None)
ax.axvline(0, color="#6b7684", lw=1)
ax.set_yticks(range(len(names)), names)
ax.set_xlim(-25, 25)
ax.set_xlabel("RMSE change vs unreconciled, %, mean of 30 worlds (capped at 25)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Every method adds up; few are more accurate", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Forecasts made level by level stop adding up as soon as any level is modelled differently (here: log scale).
- Splitting the top number by historical shares adds up and is the least accurate method when the mix drifts.
- Least-squares reconciliation gains a few percent at the top; no method improves all three levels reliably.
- Which level to trust is a backtest question: here the stores were the better forecasts, so bottom-up won the top.
- Reconciliation cannot add information: a shock no level's history contains stays a miss under every method.
""")

md("## 8. Try your own")

code("""# w = make_world(seed=7, shock=0.25)          # another fictional retailer, one region cut 25%
# yhat, res = base_forecasts(w)
# print(reconcile(w, yhat, res, "mint_shrink")[:5].round(1))
# print(run(w)["wls"]["vs_base"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version reconciles the
retailer with any of the seven methods next to what happened: `pip install -r requirements.txt && streamlit run app.py`.
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
