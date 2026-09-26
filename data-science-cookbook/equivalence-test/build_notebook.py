"""Generate demo.ipynb.

Plumbing copied from `anova-posthoc` (Day 182), content written fresh. The engine is EXTRACTED from
equiv.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The study numbers are exact integrals, so the notebook reproduces them to the digit. Only the
Monte Carlo cross-check runs at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/equivalence-test"
NB_REPS = 4000


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


ENGINE = extract("equiv.py")

EXPORTS = [
    "ALPHA", "MARGIN", "NS", "DELTAS", "OUTCOMES", "LABEL", "Z99", "MC_REPS", "MC_DESIGNS", "wilson",
    "analyse", "tost_batch", "outcome_probs", "p_equivalent", "p_not_significant", "n_for_power",
    "n_normal_approx", "grid", "calibrate", "monte_carlo", "exemplar",
]

HEADER = """from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats
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
# "No significant difference" is not "no difference"

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

"The new pricing engine made no significant difference to basket value (p = 0.34)." That sentence
gets read as "the two engines are the same". The test never said so. This notebook measures, exactly,
how often a real difference produces that sentence, and what it takes to earn the claim properly
with the two one-sided tests (TOST).

1. Calibration - the exact integral against scipy, the noncentral t, and raw-data Monte Carlo
2. How often the t-test says "no significant difference"
3. How often TOST says "equivalent"
4. How much evidence each report is
5. The n at which absence of evidence becomes evidence of absence
6. Four outcomes, not two
7. One report, read properly
8. Try your own
""")

md("""
## The engine

Extracted from `equiv.py` at build time, character for character. The key idea: with normal data
the mean difference and the pooled SD are independent, so given the SD every decision is an interval
in the mean difference. One integral over the SD gives each outcome's probability exactly.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every study number here is an exact integral and matches
`evidence.txt` to the digit. Only the raw-data Monte Carlo cross-check runs at {NB_REPS:,} replicates
instead of the study's 20,000, so its intervals are about 2.2x wider - lower resolution, not
disagreement. `evidence.txt` is the reference.
""")

md("## 1. Calibration")

code(f"""
print(calibrate())
for r in monte_carlo({NB_REPS}):
    print(f"n={{r['n']:>3}} delta={{r['delta_frac']:.1f}}*margin  exact equiv "
          f"{{r['exact']['equiv_only'] + r['exact']['both']:.4f}}  MC {{r['mc']['equiv_only'] + r['mc']['both']:.4f}}  "
          f"{{'inside' if r['inside_99'] else 'OUTSIDE'}}")
""")

md("""
Zero decision mismatches against `scipy.stats.ttest_ind`, zero against the 90% confidence interval
rule, and the integral agrees with the noncentral-t closed form to about 1e-13. Every Monte Carlo
outcome sits inside its 99% interval.

(One harness bug caught on the first run: the Wilson lower bound at zero hits came out as 2.7e-20,
not 0, so an outcome whose exact probability IS 0 failed the check. The bound is now clamped.)
""")

md("""
## 2. How often the t-test says "no significant difference"

Margin = 0.5 SD: the smallest difference the business said would matter. Columns are the TRUE
difference as a multiple of that margin.
""")

code("""
rows = grid()
def cell(n, f):
    return next(r for r in rows if r["n"] == n and r["delta_frac"] == f)
print(" n/group" + "".join(f"  {f:.1f}xM" for f in DELTAS))
for n in NS:
    print(f"{n:>7} " + "".join(f"{cell(n, f)['not_significant']:>7.3f}" for f in DELTAS))
""")

md("""
At n = 20 per group, a difference EXACTLY as large as the one that matters is reported as
"no significant difference" 66% of the time. At 1.5x the margin, still 36%.
""")

md("## 3. How often TOST says \"equivalent\"")

code("""
print(" n/group" + "".join(f"  {f:.1f}xM" for f in DELTAS))
for n in NS:
    print(f"{n:>7} " + "".join(f"{cell(n, f)['equivalent']:>7.3f}" for f in DELTAS))
""")

md("""
The 1.0xM column is TOST's false-alarm rate: it never exceeds 0.05, and at small n it is far BELOW
0.05 (0.0005 at n = 10) - TOST is very conservative when the sample is small. The 0.0xM column is its
power: at n = 20 it declares two truly identical engines equivalent 3% of the time. A small study
simply cannot show sameness at this margin, and TOST says so instead of pretending.
""")

md("""
## 4. How much evidence each report is

Likelihood ratio = P(report | truly equal) / P(report | true diff = margin). Above 1 the report
favours "equal"; the size says by how much.
""")

code("""
print(" n/group  'not significant'  'equivalent'")
for n in NS:
    a0, a1 = cell(n, 0.0), cell(n, 1.0)
    ns_lr = a0["not_significant"] / a1["not_significant"] if a1["not_significant"] > 0 else float("inf")
    print(f"{n:>7} {ns_lr:>18.2f} {a0['equivalent'] / a1['equivalent']:>13.2f}")
""")

md("""
The honest nuance, and it cuts against the slogan: **at large n, "not significant" IS strong evidence
of absence** - at n = 150 it is worth 100x. The failure is not in reading non-significance; it is
reading it without the n. At n = 20 the same sentence is worth 1.43x, next to nothing. TOST makes the
n explicit: its ratio is capped at power/alpha = 20, and it cannot be issued at all by a study too
small to support it.
""")

md("## 5. The n at which absence of evidence becomes evidence of absence")

code("""
for f in (0.0, 0.2, 0.5):
    print(f"true delta = {f:.1f}*margin  exact n/group = {n_for_power(0.8, f * MARGIN):>4}   "
          f"normal shortcut = {n_normal_approx(0.8, f):6.1f}")
""")

md("""
To have an 80% chance of showing equivalence at a 0.5 SD margin you need 70 per group when the
two are truly identical, and 199 if the real difference is half the margin. The normal-curve
shortcut undershoots by 1 to 5 per group - it ignores the t distribution's heavier tails.
""")

md("""
## 6. Four outcomes, not two

At a true difference of 0.2x the margin - real, but too small to matter - the two tests together
have four possible verdicts. A chart makes the mechanism visible.
""")

code("""
print(" n/group" + "".join(f"{LABEL[o]:>28}" for o in OUTCOMES))
for n in NS:
    print(f"{n:>7} " + "".join(f"{cell(n, 0.2)[o]:>28.4f}" for o in OUTCOMES))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ns_ = list(NS)
axes[0].plot(ns_, [cell(n, 1.0)["not_significant"] for n in ns_], "o-", color="#c8562b",
             label="'not significant' | diff = margin")
axes[0].plot(ns_, [cell(n, 0.0)["equivalent"] for n in ns_], "o-", color="#2f6fdb",
             label="'equivalent' | diff = 0")
axes[0].set_xscale("log"); axes[0].set_xlabel("n per group"); axes[0].set_ylabel("probability")
axes[0].legend(frameon=False); axes[0].set_title("Which report you get depends on n", loc="left")
bottom = np.zeros(len(ns_))
for o, c in zip(OUTCOMES, ("#2f6fdb", "#7fa8ef", "#b8860b", "#c8562b")):
    v = np.array([cell(n, 0.2)[o] for n in ns_])
    axes[1].bar(range(len(ns_)), v, bottom=bottom, color=c, label=LABEL[o])
    bottom += v
axes[1].set_xticks(range(len(ns_))); axes[1].set_xticklabels(ns_)
axes[1].set_xlabel("n per group"); axes[1].set_title("True diff = 0.2 x margin: four outcomes", loc="left")
axes[1].legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
Small n: almost everything is **inconclusive**, which the t-test alone would have reported as
"no difference". Large n: a growing share is **equivalent AND different** (35% at n = 500) - the
difference is real and immaterial, and a t-test alone would have reported it as a finding.
""")

md("## 7. One report, read properly")

code("""
ex = exemplar()
print(f"seed {ex['seed']}: basket value, SD $20, margin $10, TRUE difference ${ex['true_delta']:.0f}, n = 20 a group")
print(f"observed difference ${ex['delta']:.2f}, t-test p = {ex['p_diff']:.4f}  -> 'no significant difference'")
print(f"TOST p = {ex['p_tost']:.4f}, 90% CI (${ex['ci90'][0]:.2f}, ${ex['ci90'][1]:.2f})  -> {ex['outcome']}")
""")

md("""
The honest sentence is not "no difference". It is: *the study could not tell the two engines apart,
and at 20 baskets a group it was never going to - the interval still allows an $18 lift.*

## Summary

| claim | what it takes |
|---|---|
| "no significant difference" | nothing - a small study produces it 66% of the time for a difference that matters |
| "equivalent within +-M" (TOST) | the 90% CI inside +-M; about 70 per group at M = 0.5 SD |
| "not significant" as evidence | depends on n: worth 1.4x at n = 20, 100x at n = 150 |
""")

md("## 8. Try your own")

code("""
# control = [98.2, 104.1, 87.5, 110.3, 95.0, 101.7, 99.9, 92.4]
# new     = [101.5, 99.0, 105.2, 97.8, 108.1, 94.3, 103.6, 100.2]
# r = analyse(control, new, margin=10)
# print(r["outcome"], round(r["p_diff"], 4), round(r["p_tost"], 4), r["ci90"])
# print("n/group for 80% power at your margin:", n_for_power(0.8, 0.0, margin=0.3))
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own two groups and margin:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`sample-size-calc`](../sample-size-calc) sizes a study to DETECT a difference;
[`p-value-dance`](../p-value-dance) measured how widely p moves across replays;
[`ci-overlap-fallacy`](../ci-overlap-fallacy) measured the eye reading error bars.
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
