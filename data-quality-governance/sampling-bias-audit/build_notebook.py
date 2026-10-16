"""Generate demo.ipynb.

Plumbing copied from `fairness-accuracy-frontier` (Day 198), content written fresh. The engine is EXTRACTED from
audit.py by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
test_notebook.py imports this module, and an import-time write would replace the executed notebook with a blank one
(the Day 177 defect).

Models are fitted, not closed-form, so the notebook runs the SAME three seeds as evidence.py and reproduces
evidence.txt to the digit (gradient boosting with a fixed random_state is deterministic on one machine).
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/sampling-bias-audit"


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


ENGINE = extract("audit.py")

EXPORTS = [
    "SEEDS",
    "CELLS",
    "CURVE",
    "WORLD",
    "QUOTA",
    "EMPTY",
    "pop_shares",
    "sample",
    "audit",
    "fit",
    "score",
    "cell_weights",
    "one_seed",
    "study",
    "parse_shares",
]

HEADER = """from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
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
# The training set is not the population

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A credit model's training set was assembled with quotas: 30% group B and 40% rural, exactly as in the applicants
it will score. The usual representativeness check compares those two numbers and passes. This notebook audits the
cells instead, and measures what the miss costs the one cell it hit: rural B, whose bureau files understate their
real track record.

1. The engine
2. The audit: marginals pass, cells do not
3. What the miss costs, cell by cell
4. Two fixes: reweight, or collect
5. The chart
6. Try your own
""")

md("""
## 1. The engine

Extracted from `audit.py` at build time. `sample` draws exact cell counts; `audit` computes training share /
population share for every marginal and every cell and flags a segment outside [0.80, 1.25] when the 99% Wilson
interval agrees; `fit` trains gradient boosting; `score` reports, per cell, the share of good payers the model
rejects on the deployment population. `study` runs the same three seeds as `evidence.py` (about 20 seconds), so the
numbers match `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE)

code("""r = study()
print(f"study done: {len(r['seeds'])} seeds, training n = {r['n_train']:,}, deployment n = {r['n_pop']:,}")""")

md("""
## 2. The audit: marginals pass, cells do not

Both one-attribute checks read a ratio of 1.00. Rural B, 12.0% of the applicants, is 0.5% of the training set: 100
rows, a ratio of 0.04. A proportional sample of the same size raises no flag, so the audit is not flagging noise.
""")

code("""print(f"{'segment':<14} {'kind':<9} {'rows':>6} {'train':>7} {'population':>11} {'ratio':>6}  flag")
for name, v in r["audit"].items():
    print(f"{name:<14} {v['kind']:<9} {v['rows']:>6,} {v['train_share']:>7.1%} {v['pop_share']:>11.1%} {v['ratio']:>6.2f}  {'FLAG' if v['flag'] else 'ok'}")
print(f"control (a proportional sample of the same size): {sum(v['flag'] for v in r['audit_control'].values())} segments flagged")""")

md("""
## 3. What the miss costs, cell by cell

The model trained on the quota sample rejects 50.5% of rural B's good payers; trained on a proportional sample, 14.9%.
The other three cells do not move. The team's own test split, drawn the same way as the training set, reads 82.5%
accuracy; in deployment it is 80.8%. A 1.7-point drop is the whole signal an overall metric gives.
""")

code("""pct = lambda v: f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"
print(f"{'cell':<10} {'quota-trained':>22} {'proportional-trained':>22}")
for c in ["overall"] + CELLS:
    print(f"{c:<10} {pct(r['quota'][c]['good_rejected']):>22} {pct(r['proportional'][c]['good_rejected']):>22}")
print(f"overall accuracy: the team's own test split {pct(r['quota_on_own_test']['overall']['acc'])}, deployment {pct(r['quota']['overall']['acc'])}, proportional-trained {pct(r['proportional']['overall']['acc'])}")""")

md("""
## 4. Two fixes: reweight, or collect

Reweighting the 100 rows up to their population share brings rural B's rejection to 21.0%, but it ranges 15.6% to
27.3% across seeds: 100 rows weighted 24x is a noisy fit. With no rural B rows at all the model rejects 64.6%, and
reweighting has nothing to weight. Collecting works: by 1,000 rows (5% of the set) the cell is at 17.3%.
""")

code("""print(f"reweight the 100 rows to the population cell shares: {pct(r['reweighted']['B-rural']['good_rejected'])}")
print(f"a training set with NO rural B rows: {pct(r['empty']['B-rural']['good_rejected'])}; reweighted: {pct(r['empty_reweighted']['B-rural']['good_rejected'])}")
print("collect more rural B rows (quota sample + k extra):")
for k, v in zip(CURVE, r["curve"]):
    print(f"  {100 + k:>5,} rural B rows: {pct(v)}")""")

md("""
## 5. The chart

The full figure, with the mechanism drawn, is `audit.png` (built by `make_chart.py` from `results.json`). Here, the
rows-needed curve.
""")

code("""xs = [100 + k for k in CURVE]
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(xs, [v["mean"] * 100 for v in r["curve"]], color="#c8562b", lw=2.2, marker="o", label="quota sample + extra rural B rows")
ax.axhline(r["proportional"]["B-rural"]["good_rejected"]["mean"] * 100, color="#2e8b57", ls="--", label="proportional sample")
ax.set_xscale("log")
ax.set_xlabel("rural B rows in a 20,000-row training set")
ax.set_ylabel("rural B good payers rejected (%)")
ax.set_title("How many rows a segment needs", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- A quota on each attribute matches every marginal and says nothing about the cells: rural B fell from 12% to 0.5%.
- Audit the cells (the intersections you will deploy on), with an interval, not just each attribute.
- The cost lands on one cell: 50.5% of its good payers rejected vs 14.9%, while overall accuracy moves 1.7 points.
- Reweighting a thin cell helps but is noisy; an empty cell cannot be reweighted. About 1,000 rows fixed it here.
""")

md("## 6. Try your own")

code("""# A quota sample where rural B is 3% - is that enough?
# q = parse_shares("A-urban=0.33, A-rural=0.37, B-urban=0.27, B-rural=0.03")
# x = one_seed(1, quota=q, n_pop=20_000, with_curve=False); print(x["quota"]["B-rural"], x["proportional"]["B-rural"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your own
training-set shares: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`disparate-impact`](../disparate-impact) found that fixing B's data beat every threshold, and assumed
you could get it; this build measures how much of it you need.
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
        json.dump(nb, f, indent=1)
    print(f"wrote demo.ipynb with {len(cells)} cells ({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
