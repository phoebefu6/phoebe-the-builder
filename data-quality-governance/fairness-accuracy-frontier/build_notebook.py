"""Generate demo.ipynb.

Plumbing copied from `proxy-variable-finder` (Day 197), content written fresh. The engine is EXTRACTED
from frontier.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Models are fitted, not closed-form, so the notebook runs the SAME three seeds as evidence.py and reproduces
evidence.txt to the digit (gradient boosting with a fixed random_state is deterministic on one machine).
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/fairness-accuracy-frontier"


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n\n".join(out).rstrip()


ENGINE = extract("frontier.py")

EXPORTS = [
    "SEEDS", "N", "SELECT", "RATIOS", "WORLD", "MODELS", "generate", "fit_score", "select_at_ratio", "frontier",
    "one_seed", "bias_sweep", "study", "parse_world",
]

HEADER = """from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
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
# What does fairness cost?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Everyone agrees fairness has a price. This notebook measures it: the whole trade-off curve between the selection
ratio of two groups and the hit rate of the people selected, scored against two rulers. Ruler 1 is the past
decisions the model learned from, the only one a team usually has. Ruler 2 is true merit, which this constructed
world knows. Group B really is a little less skilled here (0.25 sd), and past decisions also held B to a higher bar.

1. The engine
2. Where each model lands
3. The frontier and the price of parity
4. How much of the price is the ruler
5. The price as a team would estimate it
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `frontier.py` at build time. `fit_score` trains gradient boosting on the biased history.
`select_at_ratio` selects 30% with group-specific cut-offs so that rate(B) / rate(A) equals a target ratio; within a
group it takes the top scores, which is the best possible selection for that count, so sweeping the ratio draws the
frontier for that score. `study` repeats everything over the same three seeds as `evidence.py` (about 30 seconds),
so the numbers match `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE)

code("""r = study()
P, pr = r["points"], r["prices"]
print(f"study done: {len(r['seeds'])} seeds x n = {r['n']:,}, {len(RATIOS)} points on the frontier")""")

md("""
## 2. Where each model lands

With one cut-off for everyone, the aware model selects B at 0.21 of A's rate. Removing the group, dropping the zip
proxy, or reweighing the training data each move the ratio up to 0.50-0.65, and each lands within half a point of
the frontier. No group-specific cut-off is needed to get that far; none of them reaches four-fifths.
""")

code("""print(f"{'model':<30} {'B/A ratio':>16} {'hit vs history':>15} {'hit vs truth':>13} {'pts below frontier':>19}")
for k in MODELS:
    v = P[k]
    rng = f"{v['ratio']['mean']:.3f} [{v['ratio']['lo']:.2f}-{v['ratio']['hi']:.2f}]"
    gb = r["gap_below"].get(k)
    gbs = f"{gb['mean']:+.2f}" if gb else "off curve"
    print(f"{k:<30} {rng:>16} {v['hit_hist']['mean']:>15.1%} {v['hit_true']['mean']:>13.1%} {gbs:>19}")
print(f"selecting by true skill itself gives a ratio of {r['oracle_ratio']['mean']:.3f}")""")

md("""
## 3. The frontier and the price of parity

Against past decisions the hit rate is best near the ratio the history itself had, and parity costs 4.8 points.
Against true merit the curve peaks near 0.7, and the same move to parity GAINS 1.6 points. The constraint is
pulling the model back from a bias it learned, and only one ruler can see that.
""")

code("""print(f"hit vs history peaks at ratio {r['peak_hist']['ratio']['mean']:.2f} ({r['peak_hist']['hit']['mean']:.1%}); hit vs truth peaks at ratio {r['peak_true']['ratio']['mean']:.2f} ({r['peak_true']['hit']['mean']:.1%})")
print(f"moving from the unconstrained model (ratio {r['start']['mean']:.2f}) to four-fifths (0.80): vs history {-pr['fourfifths_hit_hist']['mean']:+.2f} pts, vs truth {-pr['fourfifths_hit_true']['mean']:+.2f} pts")
print(f"moving from the unconstrained model to parity (1.00): vs history {-pr['parity_hit_hist']['mean']:+.2f} pts [{-pr['parity_hit_hist']['hi']:+.2f} to {-pr['parity_hit_hist']['lo']:+.2f}], vs truth {-pr['parity_hit_true']['mean']:+.2f} pts [{-pr['parity_hit_true']['hi']:+.2f} to {-pr['parity_hit_true']['lo']:+.2f}]")""")

md("""
## 4. How much of the price is the ruler

Re-run the world with the history less and more biased. At `label_bias = 0` the two rulers are the same labels, and
parity costs 0.39 points: that is the real price of the real skill gap, and it is small because the curve is flat
near its top (the swapped applicants sit at the cut-off, where the two groups are almost equally likely to be
qualified). Every point of price above that, as the bias grows, is the ruler.
""")

code("""print(f"{'label_bias':>10} {'start ratio':>12} {'parity vs history':>18} {'parity vs truth':>16} {'truth peak at':>14}")
for row in r["sweep"]:
    print(f"{row['label_bias']['mean']:>10.2f} {row['start']['mean']:>12.2f} {-row['parity_hit_hist']['mean']:>+17.2f}  {-row['parity_hit_true']['mean']:>+15.2f} {row['peak_true']['mean']:>14.2f}")
c = r["sweep"][0]
print("control, hit vs truth along the curve: " + "  ".join(f"r={x}: {c[f'true_at_{x}']['mean']:.1f}%" for x in [0.25, 0.5, 0.8, 1.0, 1.25]))""")

md("""
## 5. The price as a team would estimate it

A team measures the cost of four-fifths on a validation set. On 2,000 rows the same 3.2-point cost reads anywhere
from 1.4 to 5.0 points; on 20,000 rows, 2.9 to 3.6. Quote the interval, not the point.
""")

code("""for m, v in r["audit"].items():
    print(f"n = {int(m):>6,}: median {v['p50']['mean']:.2f} pts, 90% of samples between {v['p05']['mean']:.2f} and {v['p95']['mean']:.2f}; samples showing no cost at all {v['share_le_0']['mean']:.1%}")""")

md("""
## 6. The chart

The full figure is `frontier.png` (built by `make_chart.py` from `results.json`). Here, the frontier against both
rulers.
""")

code("""fr = r["frontier"]
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(fr["ratio"], [v * 100 for v in fr["hit_hist"]], color="#c8562b", lw=2.2, label="vs past decisions")
ax.plot(fr["ratio"], [v * 100 for v in fr["hit_true"]], color="#2e8b57", lw=2.2, label="vs true merit")
for x in (0.8, 1.0):
    ax.axvline(x, color="#6b7684", ls=":", lw=1)
ax.set_xlabel("selection ratio, rate(B) / rate(A)")
ax.set_ylabel("hit rate of the selected 30% (%)")
ax.set_title("The frontier, scored by two rulers", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Against past decisions, moving from the unconstrained model to parity costs 4.8 points of hit rate.
- Against true merit the same move gains 1.6 points: the unconstrained model was reproducing the old bar.
- With an honest history the price of parity is 0.39 points, despite a real 0.25 sd skill gap: the curve is flat at
  the top. Everything above that is the ruler, and it grows with the bias.
- Removing the group, dropping a proxy or reweighing each lands on the frontier, short of four-fifths.
- On a 2,000-row validation set the price reads 1.4 to 5.0 points. Report it as a range, against both rulers if you can
  build the second one (a re-labelled audit sample, an outcome observed later).
""")

md("## 7. Try your own")

code("""# A world with a larger real gap and an honest history - is parity still almost free?
# w = parse_world("skill_gap=0.6, label_bias=0")
# x = one_seed(1, 30_000, w, with_audit=False); print(x["start"], x["prices"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your own
world: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`disparate-impact`](../disparate-impact) measures the gap and why fairness definitions conflict;
[`proxy-variable-finder`](../proxy-variable-finder) finds where the gap enters when the attribute is removed. This one
draws the curve both of them had points on.
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
    print(f"wrote demo.ipynb with {len(cells)} cells ({len(ENGINE.splitlines())} engine lines embedded)")


if __name__ == "__main__":
    write()
