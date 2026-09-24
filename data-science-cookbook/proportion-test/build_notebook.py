"""Generate demo.ipynb.

Plumbing copied from `ci-overlap-fallacy` (Day 180), content written fresh. The engine is EXTRACTED
from proportion.py by AST at build time, so the notebook holds the library's own source text. The
write lives behind __main__: test_notebook.py imports this module, and an import-time write would
replace the executed notebook with a blank one (the Day 177 defect).

Everything in this build is exact, so the notebook runs the SAME computation as evidence.py at the
same designs and grids - no reduced replicate count, so its tables ARE evidence.txt's and no
resolution caveat is needed. A test asserts that.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/proportion-test"


def extract(path: str) -> str:
    """Every top-level def and assignment, in source order. Imports are left to HEADER.

    Tuple targets are walked (`BAND_LO, BAND_HI = ...`), because an extractor reading only
    `targets[0].id` drops them silently - the Day 178 defect.
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


ENGINE = extract("proportion.py")

EXPORTS = [
    "ALPHA", "TESTS", "LABEL", "BAND_LO", "BAND_HI", "NUISANCE", "verdict", "z_stat", "p_z",
    "chi2_uncorrected", "p_yates", "p_fisher", "p_barnard", "all_pvalues", "rate",
    "size_curve", "size_table", "power_table", "expected_count_rule", "exemplar",
    "lattice_regions", "SIZE_PS", "SIZE_DESIGNS", "POWER_DESIGNS",
]

HEADER = """from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
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
# Did conversion actually move?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Four tests get run on a 2x2 conversion table - the z-test, chi-square (with or without Yates),
Fisher's exact and Barnard's exact - and reported as though they were interchangeable. They are not.
For fixed arm sizes there are only `(n1+1)(n2+1)` possible tables, so every test's false-alarm rate
and power can be computed **exactly**. No simulation anywhere in this notebook.

1. Calibration - two of the four tests are the same test
2. Exact size - who runs hot, who runs cold
3. The expected-count-below-5 rule, scored
4. Exact power - what Fisher's caution costs
5. One table, four answers
6. Try your own
""")

md("""
## The engine

Extracted from `proportion.py` at build time, character for character.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md("""
## 1. Calibration

Every p-value is checked against scipy first. And one identity: the pooled z statistic squared is
Pearson's chi-square. Written from two different formulas, they agree to machine precision - so
"we ran a z-test and a chi-square and they agreed" is one test run twice.
""")

code("""
n1, n2 = 15, 12
pv = all_pvalues(n1, n2)
print("max |z^2 - chi2|     ", f"{np.abs(z_stat(n1, n2) ** 2 - chi2_uncorrected(n1, n2)).max():.1e}")
print("max |Fisher - scipy| ", f"{max(abs(stats.fisher_exact([[i, n1-i], [j, n2-j]])[1] - pv['fisher'][i, j]) for i in range(n1+1) for j in range(n2+1)):.1e}")
ref = stats.barnard_exact(np.array([[5, 6], [10, 6]])).pvalue
print("Barnard 5/15 vs 6/12  ", f"ours {pv['barnard'][5, 6]:.6f}  scipy {ref:.6f}  (grid = lower bound)")
""")

md("""
## 2. Exact size

Both arms at the same true rate, so every rejection is a false alarm. A test's size is its WORST
false-alarm rate over true rates 0.01 to 0.50. INFLATED above 0.055, CONSERVATIVE below 0.045.
""")

code("""
size_rows = size_table()
print("   n1/n2    " + "".join(f"{LABEL[t]:>22}" for t in TESTS))
for r in size_rows:
    print(f"  {r['n1']:>4}/{r['n2']:<4}" + "".join(f"{r[t]:>13.4f} {r[t + '_verdict'][:5]:<8}" for t in TESTS))
""")

md("""
The z-test runs hot on unbalanced arms (a 20 vs 80 split reaches 0.088 - 1.75x nominal). Fisher is
conservative on **every** design, and Yates lands on Fisher's size exactly on several of them: the
continuity correction does not repair the z-test, it turns it into an approximation of Fisher.
Barnard, which maximises over the unknown common rate, never exceeds 0.05 and is inside the band
on all but the smallest design.
""")

md("""
## 3. "Use Fisher when an expected count is below 5"

Score that rule as a classifier against the z-test's exact false-alarm rate, cell by cell.
""")

code("""
ecr = expected_count_rule(size_rows)
print("                         z fine   z inflated")
print(f"  rule says safe (>=5)   {ecr['safe_ok']:>6}   {len(ecr['safe_but_broken']):>8}")
print(f"  rule says unsafe (<5)  {len(ecr['unsafe_but_fine']):>6}   {ecr['unsafe_broken']:>8}")
w = max(ecr['safe_but_broken'], key=lambda c: c['size'])
print(f"\\nworst cell the rule calls safe: {w['n1']}/{w['n2']} at p={w['p']}, "
      f"min expected {w['min_expected']:.0f}, z false-alarm {w['size']:.4f}")
""")

md("""
Half the inflated cells sit where every expected count is at least 5 - some at expected counts of
25 and more - and almost everything the rule sends to Fisher was already fine. The z-test's error on
a 2x2 comes from the table lattice being discrete, not only from small counts.
""")

md("""
## 4. Exact power
""")

code("""
power_rows = power_table()
print("   p1 -> p2    n1/n2    " + "".join(f"{t:>9}" for t in TESTS) + "   Barnard-Fisher")
for r in power_rows:
    print(f"  {r['p1']:.2f} -> {r['p2']:.2f}  {r['n1']:>3}/{r['n2']:<4}"
          + "".join(f"{r[t]:>9.4f}" for t in TESTS) + f"   {r['barnard'] - r['fisher']:+.4f}")
""")

md("""
Barnard beats Fisher on every design here while holding size - the price of conditioning on both
margins is up to 11 points of power. The z-test's column is higher still, but wherever section 2
flags it INFLATED that lead is not power: an inflated test detects more because it rejects more
of everything.
""")

md("""
## 5. One table, four answers

Chosen by rule (the smallest table on 20/20 where z rejects and Fisher does not), then the whole
outcome lattice drawn with each table coloured by which tests reject it.
""")

code("""
ex = exemplar()
print(f"control {ex['x1']}/20, variant {ex['x2']}/20")
for t in TESTS:
    print(f"  {LABEL[t]:<22} p = {ex[t]:.4f}  {'moved' if ex[t] < ALPHA else 'not shown'}")

lat = lattice_regions()
regions = np.array(lat['code'])
from matplotlib.colors import ListedColormap
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.imshow(regions, origin="lower", cmap=ListedColormap(["#f1f3f6", "#c8562b", "#2f6fdb", "#2f7a5a", "#000"]),
          vmin=-0.5, vmax=4.5)
a1.plot([ex['x2']], [ex['x1']], "o", ms=14, mfc="none", mec="#1f2733", mew=2)
a1.set_xlabel("variant conversions / 20")
a1.set_ylabel("control conversions / 20")
a1.set_title("green: all reject   blue: z + Barnard only   orange: z only", loc="left", fontsize=10)
row = next(r for r in size_rows if (r['n1'], r['n2']) == (20, 80))
for t, col in zip(TESTS, ["#c8562b", "#b8860b", "#6b7684", "#2f6fdb"]):
    a2.plot(SIZE_PS, row[t + '_curve'], color=col, label=LABEL[t], lw=2)
a2.axhline(0.05, color="#1f2733", lw=1)
a2.set_xlabel("true common rate")
a2.set_ylabel("exact false-alarm rate, 20 vs 80")
a2.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=150)
plt.show()
print(lat['counts'])
""")

md("""
## Summary

| | |
|---|---|
| **z-test vs chi-square** | the same test - `z^2 = chi2` to 1e-14 |
| **z-test** | inflated on 6 of 9 designs, worst 0.088 on a 20/80 split |
| **Fisher / Yates** | conservative on every design; Yates often has exactly Fisher's size |
| **Barnard** | never above 0.05, and more power than Fisher on 7 of 7 designs (up to +0.109) |
| **Rule of five** | calls 12 inflated cells safe; 94% of cells it sends to Fisher were fine |

For an A/B conversion readout, the defensible default is **Barnard** (or, at large n, the z-test
on a balanced split) - and whichever test is used, report its size at your arm sizes, because the
worst-case false-alarm rate is a property of the design you can compute before launch.
""")

md("""
## Try your own
""")

code("""
# Your readout: conversions and visitors per arm.
# x1, n1, x2, n2 = 12, 400, 27, 410
# pv = all_pvalues(n1, n2)
# for t in TESTS:
#     size = size_curve(pv[t], n1, n2, SIZE_PS[::5]).max()
#     print(f"{LABEL[t]:<22} p = {pv[t][x1, x2]:.4f}   worst false-alarm at these sizes {size:.4f}")
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you type in your own counts and see all four tests plus each one's
calibration at your arm sizes:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`crosstab-chi2`](../crosstab-chi2) runs one chi-square test well;
[`ci-overlap-fallacy`](../ci-overlap-fallacy) measured exact coverage of four proportion INTERVALS;
[`t-test-variants`](../t-test-variants) did the same disagreement audit for means.
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
