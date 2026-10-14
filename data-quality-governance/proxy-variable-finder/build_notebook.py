"""Generate demo.ipynb.

Plumbing copied from `disparate-impact` (Day 196), content written fresh. The engine is EXTRACTED
from proxy.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
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
PATH = "data-quality-governance/proxy-variable-finder"


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


ENGINE = extract("proxy.py")

EXPORTS = [
    "SEEDS", "N", "SELECT", "FEATURES", "WORLD", "generate", "screen", "MODELS", "evaluate", "study", "flagged",
    "parse_world",
]

HEADER = """from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
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
# We removed the attribute and it still discriminates

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Two groups of applicants have the same distribution of true skill. Past hiring decisions, the training labels, held
one group to a higher bar. A model trained without the group attribute still selects that group far less often,
because other features stand in for it. This notebook finds those features with three screens, then measures what
removing or repairing each one does to fairness and to the two kinds of accuracy. The world is constructed and
declared; every model is a real gradient-boosting fit.

1. The engine
2. The world: same skill, biased history
3. Three proxy screens
4. Seven models
5. The chart
6. Try your own
""")

md("""
## 1. The engine

Extracted from `proxy.py` at build time. `generate` draws the world; `screen` runs three proxy screens; `evaluate`
trains a model on half the rows (biased labels), selects the top 30% of the other half, and scores it against both
the biased labels and true skill. `study` repeats everything over three seeds and keeps the min-max spread.

This runs the full study (about a minute), so the numbers match `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE)

code("""r = study()
t, s, fl = r["truth"], r["screen"], flagged(r["screen"])
print(f"study done: {len(r['seeds'])} seeds x n = {r['n']:,}, {len(MODELS)} models each")""")

md("""
## 2. The world: same skill, biased history

True skill has the same distribution in both groups, and the share truly qualified is the same. The labels the model
learns from are not: past decisions hired A at nearly three times B's rate.
""")

code("""print(f"skill gap A minus B {t['skill_gap']:+.3f} sd; truly qualified A {t['true_rate_a']:.1%}, B {t['true_rate_b']:.1%}")
print(f"past decisions (the training labels) selected A {t['hist_rate_a']:.1%}, B {t['hist_rate_b']:.1%}")""")

md("""
## 3. Three proxy screens

The screen most teams run checks one feature at a time. It flags `zip` and `exp` and passes `commute` and `shift`,
each at an AUC of 0.50 for predicting the group. Together, those two predict the group at 0.92: group B's shift sign
follows its commute sign and group A's opposes it, a pattern no single column shows. A drop-one ranking on the joint
model has the opposite blind spot: it ranks `zip`, the strongest single proxy, below the pair, because the pair can
stand in for it.
""")

code("""print("   one feature at a time:  " + "  ".join(f"{f} {v['mean']:.3f}" for f, v in s["single"].items()))
top_pairs = sorted(s["pairs"].items(), key=lambda kv: -kv[1]["mean"])
hidden = [(p, v) for p, v in top_pairs if p in fl["pairs_hidden_from_single"]]
print("   pairs (hidden from the single screen):  " + "  ".join(f"{p} {v['mean']:.3f}" for p, v in hidden))
print(f"   all five together: {s['joint']['mean']:.3f}")
print(f"   flagged at AUC > 0.60 by the single screen: {', '.join(fl['single'])}; missed by it, caught by pairs: {', '.join(fl['pairs_hidden_from_single'])}")
print("   drop-one importance (joint AUC lost when the feature is removed):  "
      + "  ".join(f"{f} {v['mean']:.3f}" for f, v in sorted(s["drop_one"].items(), key=lambda kv: -kv[1]["mean"])))""")

md("""
## 4. Seven models

All seven learn from the same biased history. Removing the group only moves the selection ratio from 0.22 to 0.39.
Linear repair (subtract each feature's group mean) reaches 0.58, because the pair's link to the group is not in its
mean. Dropping zip and the pair and repairing experience reaches 1.01 and leaves no trace of the group in the score.
The number the team can measure, AUC against past decisions, falls from 0.844 to 0.805 on the way. AUC against true
skill rises from 0.778 to 0.817. The model was being rewarded for reproducing the old bias.
""")

code("""print(f"{'model':<36} {'B/A ratio':>16} {'AUC vs history':>15} {'AUC vs truth':>13} {'group from score':>17} {'qualified sel A':>16} {'qualified sel B':>16}")
for k in MODELS:
    v = r["models"][k]
    rng = f"{v['ratio']['mean']:.3f} [{v['ratio']['lo']:.2f}-{v['ratio']['hi']:.2f}]"
    print(f"{k:<36} {rng:>16} {v['auc_hist']['mean']:>15.3f} {v['auc_true']['mean']:>13.3f} {v['group_from_score']['mean']:>17.3f} {v['qual_sel_a']['mean']:>16.1%} {v['qual_sel_b']['mean']:>16.1%}")""")

md("""
## 5. The chart

The full audit figure is `proxy_audit.png` (built by `make_chart.py` from `results.json`). Here, the pair drawn: the
two features that each look like noise, coloured by group.
""")

code("""d = generate(4000, SEEDS[0])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (title, gsel) in zip(axes, [("group A", 0), ("group B", 1)]):
    m = d["g"] == gsel
    ax.scatter(d["commute"][m], d["shift"][m], s=4, alpha=0.4, color="#2f6fdb" if gsel == 0 else "#c8562b")
    ax.axhline(0, color="#8a94a3", lw=0.8)
    ax.axvline(0, color="#8a94a3", lw=0.8)
    ax.set_title(f"{title}: commute vs shift", loc="left")
    ax.set_xlabel("commute")
axes[0].set_ylabel("shift")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Same skill, biased labels: a model with the group removed still selects B at 0.39 of A's rate.
- A one-feature-at-a-time screen passes two features that together predict the group at 0.92.
- A drop-one ranking hides the strongest single proxy behind a redundant one.
- Linear repair cannot reach a proxy that lives in an interaction.
- Dropping the proxies and repairing the mixed feature reaches parity, and AUC against true skill rises while AUC
  against the biased history falls. Judge a fairness fix against the outcome you want, not against the old decisions.
""")

md("## 6. Try your own")

code("""# A world where the pair is weaker and past decisions were less biased:
# w = parse_world("pair_link=0.4, label_bias=0.4")
# d = generate(30_000, 1, w); print(screen(d)["pairs"]["commute+shift"])
# for k, v in MODELS.items(): print(k, round(evaluate(d, v)["ratio"], 3))""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your own
world: `pip install -r requirements.txt && streamlit run app.py`.

Sibling build: [`disparate-impact`](../disparate-impact) measures the gap between groups and why fairness
definitions conflict; this build finds where the gap enters when the attribute is not even an input.
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
