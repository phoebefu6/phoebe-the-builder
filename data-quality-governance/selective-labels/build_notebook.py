"""Generate demo.ipynb.

Plumbing copied from `sampling-bias-audit` (Day 199), content written fresh. The engine is EXTRACTED from
selective.py by AST at build time, so the notebook holds the library's own source text. The write lives behind __main__:
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
PATH = "data-quality-governance/selective-labels"


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


ENGINE = extract("selective.py")

EXPORTS = [
    "SEEDS",
    "EXPLORE",
    "WORLD",
    "CONTROL",
    "draw",
    "fit",
    "decide",
    "propensity",
    "metrics",
    "evaluate",
    "one_seed",
    "study",
    "parse_world",
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
# We only know outcomes for the people we approved

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Every credit model is trained and judged on the loans someone once approved. Those officers used an interview read
the data never recorded, and held group B to a higher bar. This notebook measures what that label set does to a
model trained on it, to the dashboard that evaluates it, to a fairness audit, and to the standard fixes.

1. The engine
2. The label set is a selected sample
3. Two models, scored on the truth
4. Five estimators, one verdict each
5. How big a random-explore slice must be
6. Control: when selection runs on recorded data
7. The chart
8. Try your own
""")

md("""
## 1. The engine

Extracted from `selective.py` at build time. `draw` makes one history period and the officers' decisions; `fit`
trains gradient boosting on bureau score and group (never the interview read); `evaluate` scores a model on a
history period five ways. `study` runs the same three seeds as `evidence.py` (about 10 seconds per world), so the
numbers match `evidence.txt` exactly.
""")

code(HEADER + "\n" + ENGINE)

code("""r = study()
c = study(w=CONTROL)
pct = lambda v: f"{v['mean']:.1%} [{v['lo']:.1%}-{v['hi']:.1%}]"
pts = lambda v: f"{v['mean'] * 100:+.1f} [{v['lo'] * 100:+.1f} to {v['hi'] * 100:+.1f}]"
print(f"study done: {len(r['seeds'])} seeds, {r['n']:,} applicants per period")""")

md("""
## 2. The label set is a selected sample

61% of applicants repay; 91% of the labeled rows do. Group B's labeled rows repay at 95% against A's 90%, because
only B applicants who cleared a 0.6 sd higher bar were ever approved. The higher bar reads, in the labels, as lower
risk.
""")

code("""print(f"repay rate, all applicants: {pct(r['pop_repay'])}; labeled (approved) rows: {pct(r['hist_repay_labeled'])}")
print(f"labeled repay rate by group: A {pct(r['a_labeled_repay'])}, B {pct(r['b_labeled_repay'])} (B cleared a {WORLD['bar_b']} sd higher bar)")""")

md("""
## 3. Two models, scored on the truth

The oracle is trained on every outcome (only a simulation has them); the other on the labeled rows, as a team would.
The label-trained model approves 69% of group B against the oracle's 39%, and its approvals truly repay 76.4% against
82.8%. Past prejudice against B became, through the labels, a model that over-approves B.
""")

code("""print(f"{'model':<22} {'repay among approved':>22} {'share of B approved':>22} {'TPR A':>8} {'TPR B':>8}")
for key, name, sb in (("eval_full_model", "oracle (all outcomes)", "oracle"), ("eval_lab_model", "trained on labels", "labeled")):
    t = r[key]["truth"]
    print(f"{name:<22} {pct(t['repay']):>22} {pct(r['share_b_approved'][sb]):>22} {t['tpr_A']['mean']:>8.1%} {t['tpr_B']['mean']:>8.1%}")""")

md("""
## 4. Five estimators, one verdict each

The labeled rows put the label-trained model ahead, 94.3% against 93.7%, at every seed: they rank the worse model
first. Counting every rejected applicant as a default ranks correctly but reads 20 points low. IPW, with the
propensity of approval fitted on bureau score and group, reads 93% for both and calls it a tie: it cannot reweight
on a signal nobody recorded. Only a 5% random-explore slice gets both numbers right. The fairness audit is no safer:
the oracle's true TPR gap is +2.0 points, the labeled rows say -5.7.
""")

code("""EST = [("truth", "truth (all outcomes)"), ("labeled", "labeled rows only"), ("rejected_bad", "rejected = bad"),
       ("ipw", "IPW on x, group"), ("explore_0.05", "5% random explore")]
print(f"{'estimator':<22} {'oracle repay':>22} {'label-trained repay':>22} {'oracle TPR gap A-B (pts)':>28}  ranks right?")
for k, name in EST:
    o, lm = r["eval_full_model"][k], r["eval_lab_model"][k]
    right = "yes" if o["repay"]["lo"] > lm["repay"]["hi"] else "NO" if o["repay"]["hi"] < lm["repay"]["lo"] else "tie"
    print(f"{name:<22} {pct(o['repay']):>22} {pct(lm['repay']):>22} {pts(o['tpr_gap']):>28}  {right}")""")

md("""
## 5. How big a random-explore slice must be

Approving a random share of applicants regardless of score is the one label set nobody selected. 1% gives about 160
labeled approvals and a 95% interval of +/-6.1 points; 5% gives +/-2.6; 10% gives +/-1.8. The price is the loans
approved that the score would have refused.
""")

code("""print(f"truth {r['eval_full_model']['truth']['repay']['mean']:.1%}")
for e in EXPLORE:
    v = r["eval_full_model"][f"explore_{e:g}"]
    print(f"  explore {e:>4.0%}: {v['n_approved']['mean']:>6,.0f} labeled approvals, estimate {pct(v['repay'])}, 95% CI +/- {v['ci_half']['mean'] * 100:.1f} pts")""")

md("""
## 6. Control: when selection runs on recorded data

Set `sel_u = 0`: officers used only bureau score and group. Now IPW recovers the repay rate (82.5% against 82.8%),
and training on the labeled rows costs almost nothing (82.3%). So the method is not broken; the world was. One
negative result survives: IPW's TPR for group B still reads 67.2% against a true 53.2%, because a quarter of B's
good payers had under a 2% chance of approval and almost none of them carry a label. Weighting cannot reach
people who were never approved.
""")

code("""for k, name in EST[:4]:
    o = c["eval_full_model"][k]
    print(f"{name:<22} oracle repay {pct(o['repay'])}   TPR B {pct(o['tpr_B'])}")
print(f"label-trained model, true repay {pct(c['eval_lab_model']['truth']['repay'])}; share of B approved {pct(c['share_b_approved']['labeled'])} vs oracle {pct(c['share_b_approved']['oracle'])}")""")

md("""
## 7. The chart

The full figure, with the decision boundary drawn, is `audit.png` (built by `make_chart.py`). Here, the five
estimators side by side.
""")

code("""xs = np.arange(len(EST))
o, lm = r["eval_full_model"], r["eval_lab_model"]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.bar(xs - 0.2, [o[k]["repay"]["mean"] * 100 for k, _ in EST], 0.38, color="#2e8b57", label="oracle model")
ax.bar(xs + 0.2, [lm[k]["repay"]["mean"] * 100 for k, _ in EST], 0.38, color="#e3b23c", label="label-trained model")
ax.set_xticks(xs)
ax.set_xticklabels([n for _, n in EST], fontsize=9)
ax.set_ylim(40, 100)
ax.set_ylabel("repay rate among approvals (%)")
ax.set_title("Each estimator's view of the same two models", loc="left")
ax.legend(frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Outcomes exist only for past approvals, so every metric computed on labels is computed on a selected sample.
- When past decisions used something unrecorded, a model trained on labels is worse and the labeled rows rank it
  first; a higher bar for one group becomes a tilt toward it.
- IPW fixes selection on recorded data only, and never reaches people who had no chance of approval.
- A small random-explore slice is the only unbiased label set: 5% bought +/-2.6 points here.
""")

md("## 8. Try your own")

code("""# What if officers leaned on the interview read only half as much, and held B to a 1 sd bar?
# w = parse_world("sel_u=0.5, bar_b=1.0")
# x = one_seed(1, w, n=20_000); print(x["eval_full_model"]["ipw"]["repay"], x["eval_full_model"]["truth"]["repay"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your own
world settings: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`disparate-impact`](../disparate-impact), [`proxy-variable-finder`](../proxy-variable-finder),
[`fairness-accuracy-frontier`](../fairness-accuracy-frontier) and [`sampling-bias-audit`](../sampling-bias-audit)
all judged models against true merit; this build shows why that ruler is the hardest thing to get.
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
