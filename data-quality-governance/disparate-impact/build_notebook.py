"""Generate demo.ipynb.

Plumbing copied from `human-escalation-policy` (Day 195), content written fresh. The engine is EXTRACTED
from fairness.py by AST at build time, so the notebook holds the library's own source text. The write lives behind
__main__: test_notebook.py imports this module, and an import-time write would replace the executed notebook
with a blank one (the Day 177 defect).

Every rate is exact from the binormal scores, so the notebook reproduces evidence.txt to the digit. Only the Monte
Carlo calibration runs NB_REPS instead of the study's 200,000 - stated in the notebook, and a test asserts the note.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-quality-governance/disparate-impact"
NB_REPS = 50_000


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


ENGINE = extract("fairness.py")

EXPORTS = [
    "SEED", "MC_REPS", "GROUPS", "WORLD", "FOUR_FIFTHS", "GRID", "rates", "mix", "auc", "pooled_auc", "overall",
    "equalized_odds", "policies", "impossibility", "audit_power", "simulate", "wilson", "calibrate", "parse_world",
    "data_fix",
]

HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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

ORDER = ["one threshold (max accuracy)", "demographic parity", "equal opportunity (TPR)", "predictive parity (PPV)",
         "equalized odds (TPR+FPR)"]

POLICY_PRINT = """def show(pols):
    print(f"{'policy':<30} {'accuracy':>9} {'4/5 ratio':>10} {'sel A':>6} {'sel B':>6} {'TPR A':>6} {'TPR B':>6} "
          f"{'FPR A':>6} {'FPR B':>6} {'PPV A':>6} {'PPV B':>6}")
    for n in ORDER:
        p, o = pols[n]["per"], pols[n]["overall"]
        print(f"{n:<30} {o['acc']:>9.1%} {o['impact_ratio']:>10.3f} {p['A']['sel']:>6.1%} {p['B']['sel']:>6.1%} "
              f"{p['A']['tpr']:>6.1%} {p['B']['tpr']:>6.1%} {p['A']['fpr']:>6.1%} {p['B']['fpr']:>6.1%} "
              f"{p['A']['ppv']:>6.1%} {p['B']['ppv']:>6.1%}")"""

md(f"""
# The model is fine overall

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A screening model scores applicants from two groups. Overall it is good: AUC 0.846, 78.4% accurate. This notebook
asks the per-group questions an auditor asks, and shows why the three most common fairness definitions cannot all
hold when the groups' base rates differ. The world is constructed and declared; every rate is exact.

1. The engine
2. Calibration - exact vs simulation
3. Fine overall, not fine per group
4. Five policies, each enforcing one definition
5. Three definitions cannot all hold
6. Can an audit sample tell?
7. Fix the data instead
8. The chart
9. Try your own
""")

md("""
## 1. The engine

Extracted from `fairness.py` at build time. Within each group, scores are normal given the label, so selection rate,
true and false positive rate and precision at any threshold are closed-form. Equalized odds follows Hardt, Price and
Srebro (2016): the group with the better ROC curve is randomised between two thresholds down onto the other's.
""")

code(HEADER + "\n" + ENGINE + "\n\n\nORDER = " + repr(ORDER) + "\n\n\n" + POLICY_PRINT)

md(f"""
## 2. Calibration - exact vs simulation

`simulate` draws applicants, scores them and applies each policy, coin flips included. Each exact rate should sit
inside the 99% Wilson interval of its simulated share.

**Resolution of the instrument.** The study in `evidence.txt` draws {200_000:,} applicants per policy; this notebook
draws {NB_REPS:,} so it runs in seconds, so its intervals are about 2x wider. The exact numbers below do not depend
on this.
""")

code(f"""cal = calibrate(n={NB_REPS})
print(f"{{sum(r['inside'] for r in cal)}}/{{len(cal)}} exact rates inside the simulation's 99% Wilson interval")""")

md("""
## 3. Fine overall, not fine per group

Group B is 30% of applicants, 25% of them qualified against A's 40%, and the score separates B's qualified from its
unqualified less well (AUC 0.760 against 0.871), because the training data held fewer of them. One threshold for
everyone selects a qualified A applicant 65.9% of the time and a qualified B applicant 46.4% of the time.
""")

code("""print(f"{'group':<6} {'share':>6} {'base rate':>10} {'AUC':>6}")
for g in GROUPS:
    print(f"{g:<6} {WORLD[g]['share']:>6.0%} {WORLD[g]['base']:>10.0%} {auc(g):>6.3f}")
pols = policies()
one = pols["one threshold (max accuracy)"]
o = one["overall"]
print(f"pooled AUC {pooled_auc():.3f}, accuracy {o['acc']:.1%} at one threshold ({one['tA']:.2f}); "
      f"four-fifths ratio {o['impact_ratio']:.3f} (rule: >= {FOUR_FIFTHS})")
print(f"qualified applicants selected: A {one['per']['A']['tpr']:.1%}, B {one['per']['B']['tpr']:.1%}")""")

md("""
## 4. Five policies, each enforcing one definition

Demographic parity equalises selection (the four-fifths rule's view). Equal opportunity equalises the share of
qualified applicants selected. Predictive parity equalises precision. Equalized odds equalises both true and false
positive rates. Each policy meets its own definition and breaks at least one other. Equalized odds on this model
only works by levelling A down: 12% of A applicants are judged at a threshold that admits almost everyone and the
rest at a much stricter one, and accuracy falls 6.3 points. Predictive parity is the most accurate of the five and
selects 3.2% of B.
""")

code("""show(pols)
eo = pols["equalized odds (TPR+FPR)"]
t1, t2, q = eo["tA"]
print(f"equalized odds: A is randomised - threshold {t1:.2f} for {q:.0%} of A applicants, {t2:.2f} for the rest; B at {eo['tB']:.2f}")""")

md("""
## 5. Three definitions cannot all hold

Within any group, FPR = p / (1 - p) x (1 - PPV) / PPV x TPR (Chouldechova, 2017). If base rates p differ, equal
precision and equal true positive rate force unequal false positive rates. Checked by brute force: every
deterministic threshold pair on the grid.
""")

code("""imp = impossibility()
c = imp["closest"]
print(f"threshold pairs checked: {imp['pairs']:,}; within {imp['tol']:.0%} on two of PPV/TPR/FPR: {imp['two_of_three']:,}; on all three: {imp['all_three']}")
print(f"closest pair (A {c['tA']:.2f}, B {c['tB']:.2f}): PPV gap {c['ppv_gap']:.3f}, TPR gap {c['tpr_gap']:.3f}, FPR gap {c['fpr_gap']:.3f}")""")

md("""
## 6. Can an audit sample tell?

An auditor rarely sees the population. Drawing applicants in the real proportions and computing the four-fifths ratio
on the sample, the verdict matches the population's only 84% of the time with 100 applicants from B. It takes about
500 before it is right 98% of the time.
""")

code("""au = audit_power(one["tA"], [25, 50, 100, 200, 500, 1000, 2000])
print(f"{'n of B':>7} {'verdict agrees':>15} {'ratio 5th pct':>14} {'ratio 95th pct':>15}")
for r in au:
    print(f"{r['n_B']:>7} {r['agree']:>15.1%} {r['ratio_p5']:>14.3f} {r['ratio_p95']:>15.3f}")""")

md("""
## 7. Fix the data instead

Every policy above moves thresholds on a score that is weaker for B. Make B's score as informative as A's, base rates
unchanged, and one threshold passes the four-fifths rule (0.657 to 0.828) while accuracy rises (78.4% to 80.2%).
Equalized odds now costs nothing. The precision gap stays, because it comes from the base rates and no amount of
model quality removes it, and equalising true positive rates now fails the four-fifths rule.
""")

code("""fix = data_fix()
show(fix["policies"])
fo = fix["policies"]["one threshold (max accuracy)"]["overall"]
print(f"one threshold now: accuracy {o['acc']:.1%} -> {fo['acc']:.1%}, four-fifths ratio {o['impact_ratio']:.3f} -> {fo['impact_ratio']:.3f}")""")

md("""
## 8. The chart

The full audit figure is `fairness_audit.png` (built by `make_chart.py` from `results.json`). Here, the two ROC
curves: equalized odds can only live where both groups can reach, which is under B's curve.
""")

code("""ts = np.linspace(-4, 5, 300)
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for g, col in (("A", "#2f6fdb"), ("B", "#c8562b")):
    fpr = [rates(g, t)["fpr"] for t in ts]
    tpr = [rates(g, t)["tpr"] for t in ts]
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f"group {g} (AUC {auc(g):.3f})")
for n, mk in (("one threshold (max accuracy)", "o"), ("equalized odds (TPR+FPR)", "s")):
    p = pols[n]["per"]
    for g, col in (("A", "#2f6fdb"), ("B", "#c8562b")):
        ax.scatter(p[g]["fpr"], p[g]["tpr"], color=col, marker=mk, s=60, zorder=3)
a0, a1 = pols["one threshold (max accuracy)"]["per"]["A"], pols["equalized odds (TPR+FPR)"]["per"]["A"]
ax.annotate("", xy=(a1["fpr"] + 0.01, a1["tpr"] + 0.01), xytext=(a0["fpr"] + 0.01, a0["tpr"] - 0.01),
            arrowprops=dict(arrowstyle="->", color="#2f6fdb", lw=1.4))
ax.text(0.20, 0.38, "equalized odds pulls A off its own curve", fontsize=9)
ax.text(0.20, 0.34, "onto the point B already had (both squares)", fontsize=9)
ax.plot([0, 1], [0, 1], color="#8a94a3", ls=":", lw=1)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("Circles: one threshold. Squares: equalized odds.", loc="left")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=120)
plt.show()""")

md("""
## Summary

- Fine overall (AUC 0.846, 78.4% accurate), and a qualified B applicant is selected 46.4% of the time against 65.9%
  for A. The four-fifths ratio is 0.657.
- Each fairness definition can be met by a threshold policy, and each one breaks another. None of 56,977 threshold
  pairs meets equal precision, true positive rate and false positive rate within 2 points.
- Equalized odds on this model costs 6.3 points of accuracy by levelling A down. Predictive parity is the most
  accurate policy and selects 3.2% of B.
- A four-fifths audit with 100 applicants from B gets the verdict wrong about 1 time in 6.
- Fixing B's data does what no threshold can: passes four-fifths and raises accuracy. The precision gap remains,
  because base rates differ.
""")

md("## 9. Try your own")

code("""# Your groups, e.g. a smaller base-rate gap and a better score for B:
# mine = parse_world("B.base=0.35, B.mu_q=1.0")
# show(policies(mine)); print(impossibility(mine)["all_three"])""")

md("""
---
Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder). The Streamlit version takes your
own groups: `pip install -r requirements.txt && streamlit run app.py`.

Sibling builds: [`resume-screener`](../../document-intelligence/resume-screener) is the kind of model this audits;
[`pii-detector`](../../data-infra-toolkit/pii-detector) and [`consent-tracker`](../consent-tracker) cover privacy, which is a different
question from fairness.
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
