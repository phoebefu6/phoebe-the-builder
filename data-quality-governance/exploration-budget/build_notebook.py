"""Generate demo.ipynb.

Plumbing copied from `selective-labels` (Day 200), content written fresh. The engine is EXTRACTED from
explore.py by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
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
PATH = "data-quality-governance/exploration-budget"


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

EXPORTS = [
    "SEEDS",
    "UNIFORM",
    "TARGETED",
    "GAIN",
    "LOSS",
    "PER",
    "WORLD",
    "draw",
    "fit",
    "decide",
    "uniform_est",
    "targeted_est",
    "tpr_b",
    "one_seed",
    "study",
    "cheapest",
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
# What does a random-approval slice cost, and what does it buy?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

`selective-labels` found that when past approvals used a signal nobody recorded, the labeled rows rank a worse credit
model first and only a random-approval slice ranks it right. Risk asks the next question: what does that slice cost
in bad loans, and what is the smallest one that still gets the ranking right? This notebook prices two designs.

1. The engine
2. What each design lends into
3. Cost and power, design by design
4. The cheapest slice that ranks right
5. What targeted cannot do: the fairness audit
6. The chart
7. Try your own
""")

md("""
## 1. The engine

Extracted from `explore.py` at build time. The world is the one `selective-labels` declared. **Uniform** approves a
random share of all applicants. **Targeted** uses production's own approvals, labeled for free, and approves a
random share `q` of only the rows a challenger wants and production refused, weighting them 1/q. Every design point
is replayed 400 times on the same applicants. `study` runs the same three seeds as `evidence.py` (about 8 seconds),
so the numbers match `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE)

code("""r = study()
pct = lambda v: f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"
usd = lambda v: f"${v:,.0f}"
print(f"study done: {len(r['seeds'])} seeds, {r['n']:,} applicants per period, {r['reps']} replays per point")""")

md("""
## 2. What each design lends into

The rows production refused repay 41.3%; uniform explore lends into all of them. The rows a challenger wants and
production refused repay 54.3%; that band is all targeted ever lends into. At a $300 margin and $1,500 loss both lose
money per loan, but targeted loses less per loan and needs fewer of them, because the 74% of approvals the two
challengers share with each other are mostly already labeled.
""")

code("""print(f"true repay among approvals: oracle {pct(r['truth']['oracle'])}, label-trained {pct(r['truth']['label'])}")
print(f"repay rate of rows production refused (uniform's extra loans): {pct(r['rejected_repay'])}")
print(f"repay rate of rows a challenger wants and production refused (targeted's pool): {pct(r['challenger_only_repay'])}")
print(f"oracle approvals the label-trained model also approves: {pct(r['agree'])}")""")

md("""
## 3. Cost and power, design by design

`P right` is the share of replayed slices in which the oracle (truly 6.4 points better) comes out ahead. Both designs
are unbiased (bias column, about zero); they differ in what each point costs. At about the same spend, uniform 0.5%
($22,702) ranks right 90.6% of the time with a +/-8.0 point interval on the gap; targeted 2% ($23,280) ranks right
99.5% of the time at +/-4.1.
""")

code("""print(f"{'design':<16} {'extra loans':>12} {'net cost':>10} {'P right (worst seed)':>22} {'95% CI on gap':>15} {'bias oracle':>12}")
for design, grid in (("uniform", UNIFORM), ("targeted", TARGETED)):
    for s in grid:
        p = r[f"{design}_{s:g}"]
        print(f"{design + ' ' + format(s, '.2%'):<16} {p['extra_loans']['mean']:>12,.0f} {usd(p['net_cost']['mean']):>10} "
              f"{p['p_right']['mean']:>12.1%} ({p['p_right']['lo']:.1%}) {'+/-' + format(p['diff_ci']['mean'] * 100, '.1f') + ' pts':>15} "
              f"{p['bias_oracle']['mean'] * 100:>+10.2f} pts")""")

md("""
## 4. The cheapest slice that ranks right

Asking for 95% at the worst of three seeds: uniform needs 2% ($90,810 per 10,000 applicants); targeted needs 2%
($23,280), 3.9x cheaper. Both are far below `selective-labels`' 5% ($227,025): that number was sized for a +/-2.6
point interval on one model's repay level. Ranking two models is a paired question, and the shared approvals cancel.
""")

code("""for design in ("uniform", "targeted"):
    b = cheapest(r, design)
    print(f"{design:<9} {b['share']:.2%} slice: {b['extra_loans']:,.0f} extra loans, net cost {usd(b['net_cost'])}")
print(f"targeted is {cheapest(r, 'uniform')['net_cost'] / cheapest(r, 'targeted')['net_cost']:.1f}x cheaper for the same verdict")
u, t = r["uniform_0.005"], r["targeted_0.02"]
print(f"at about the same spend: uniform 0.50% {usd(u['net_cost']['mean'])} -> P {u['p_right']['mean']:.1%}, CI +/-{u['diff_ci']['mean'] * 100:.1f}; "
      f"targeted 2.00% {usd(t['net_cost']['mean'])} -> P {t['p_right']['mean']:.1%}, CI +/-{t['diff_ci']['mean'] * 100:.1f}")""")

md("""
## 5. What targeted cannot do: the fairness audit

A true positive rate counts every good payer, including the ones that neither production nor any challenger would
approve. 15.2% of group B's good payers sit there, and targeted never lends to them. Its estimate of the oracle's
TPR for B is +9.5 points too high at 5%, and +9.6 at 50%: ten times the spend, the same bias. Uniform 5% is off by
-0.1. Use targeted to choose between models; keep a small uniform slice for the audit.
""")

code("""print(f"true TPR B {pct(r['true_tpr_b'])}; B good payers no design but uniform can ever label: {pct(r['unreachable_good_b'])}")
for design, s in (("uniform", 0.05), ("targeted", 0.05), ("targeted", 0.5)):
    p = r[f"{design}_{s:g}"]
    print(f"  {design} {s:.0%}: TPR B bias {p['tpr_b_bias']['mean'] * 100:+.1f} pts [{p['tpr_b_bias']['lo'] * 100:+.1f} to {p['tpr_b_bias']['hi'] * 100:+.1f}]")""")

md("""
## 6. The chart

The full figure, with the who-lends-to-whom strip drawn, is `audit.png` (built by `make_chart.py`). Here, cost
against power.
""")

code("""fig, ax = plt.subplots(figsize=(9, 4.4))
for design, grid, col in (("uniform", UNIFORM, "#2f6fdb"), ("targeted", TARGETED, "#c8562b")):
    ax.plot([r[f"{design}_{s:g}"]["net_cost"]["mean"] / 1000 for s in grid],
            [r[f"{design}_{s:g}"]["p_right"]["mean"] * 100 for s in grid], "o-", color=col, label=f"{design} explore")
ax.axhline(95, color="#6b7684", ls=":")
ax.set_xscale("log")
ax.set_xlabel(f"net cost per {PER:,} applicants ($k, log)")
ax.set_ylabel("chance of ranking the better model first (%)")
ax.set_title("Same verdict, a quarter of the bill", loc="left")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Choosing between two models is a paired question: a 2% uniform slice ranks them right where Day 200 sized 5% for
  a level estimate.
- Production's approvals are already labeled. Sampling only the rows a challenger wants and production refused gives
  the same verdict for about a quarter of the loss.
- That cheap design cannot audit fairness: good payers nobody would approve are never labeled, and no amount of
  targeted spend reaches them.
""")

md("## 7. Try your own")

code("""# A harsher loss: rerun one seed and compare the cheapest designs' costs.
# GAIN, LOSS = 300.0, 4_000.0
# x = study([200], n=20_000, reps=200); print(cheapest(x, "uniform"), cheapest(x, "targeted"))""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your own
loan economics: `pip install -r requirements.txt && streamlit run app.py`.

Sibling build: [`selective-labels`](../selective-labels) found the problem this build prices.
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
