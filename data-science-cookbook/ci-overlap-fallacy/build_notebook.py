"""Generate demo.ipynb.

Plumbing copied from the sibling `nonparametric-swap` (Day 179), content written fresh.

The engine block is EXTRACTED FROM overlap.py BY AST AT BUILD TIME rather than hand-copied, so the
notebook holds the library's own source text and cannot go stale. This module must NOT write the
notebook at import time, because test_notebook.py imports it - an unguarded write silently
replaces the executed notebook with a blank one (the Day 177 defect). The write lives behind
__main__.

The notebook uses the library's own design lists and seeds, so its Monte Carlo rows are the first
chunks of the evidence file's samples. It runs fewer replicates, which is stated where it happens.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/ci-overlap-fallacy"
NB_REPS = 8_000


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


ENGINE = extract("overlap.py")

EXPORTS = [
    "ALPHA", "BAND_LO", "BAND_HI", "Z_ALPHA", "wilson", "se_ratio", "matching_level",
    "overlap_fraction_at_significance", "rule_alpha", "paired_slack", "run_means",
    "run_paired", "prop_ci", "exact_coverage", "exact_prop_rule", "MEANS_DESIGNS",
    "CONTAINMENT_DESIGNS", "RHO_GRID", "COVERAGE_PS", "COVERAGE_NS", "METHODS",
    "MEANS_SEED_BASE", "CONTAINMENT_SEED_BASE", "PAIRED_SEED_BASE", "geometry_table",
]

HEADER = """from __future__ import annotations

import math
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
# "The error bars overlap, so it is not significant"

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

For two independent means, 95% bars that do not touch imply `p < 0.05`. The converse is false:
**a large share of real, significant results have overlapping bars.** Read as a test, the overlap
rule has a size of about 0.6%, not 5%. On paired data it fails the other way.

1. The geometry - exact
2. Calibration - the theorem, checked before any result is read
3. The dead zone - measured
4. The reversal on paired data - measured
5. The bars themselves - exact coverage of four proportion intervals
6. Try your own
""")

md("""
## The engine

Extracted from `overlap.py` at build time, character for character, so this notebook runs from a
bare Colab link with nothing else installed beyond numpy, scipy and matplotlib.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md("""
## 1. The geometry - exact

Two estimates with standard errors `s1`, `s2`. The bars separate when `|d| > z(s1 + s2)`; the test
rejects when `|d| > z_alpha * sqrt(s1^2 + s2^2)`. Since `s1 + s2 >= sqrt(s1^2 + s2^2)`, non-overlap
is always the stricter event - by a factor of `sqrt(2)` at equal standard errors.
""")

code('''
print(f"matching level at equal SE : {matching_level(1, 1) * 100:.1f}%   (not 95%)")
print(f"overlap allowed at p = 0.05: {overlap_fraction_at_significance(1, 1) * 100:.1f}% of an arm")
print(f"size of the overlap rule   : {rule_alpha(1, 1):.4f}   (labelled 0.05)")
print()
print("  sd ratio   threshold x   matching level   overlap still allowed")
for g in geometry_table():
    print(f"  {g['sd_ratio']:>8.2f}   {g['se_ratio']:>11.4f}   {g['matching_level'] * 100:>13.1f}%"
          f"   {g['overlap_at_sig'] * 100:>10.1f}%")
''')

md("""
The rule is **worst** at a ratio of 1 - two groups of the same size and spread, the design everyone
trusts. "A little overlap is fine" has no single number: it runs from 58.6% of an arm to near zero.
""")

md(f"""
## 2. Calibration - the theorem, checked first

If the bars and the test share their standard errors, "bars apart" must imply "significant" with
no exceptions. Any violation would mean the harness is broken. Then break the premise on purpose:
same bars, but test with the observed variance (Welch).

*This notebook runs {NB_REPS:,} replicates per design against the study's 200,000, using the same
seeds - so these rows are the first slice of the evidence file's sample. Lower counts widen every
interval; that is the resolution of the instrument, not a disagreement with `evidence.txt`.*
""")

code(f'''
NB_REPS = {NB_REPS}
print("  design          shared-SE violations   Welch violations")
for i, (n1, n2, s1, s2, d) in enumerate(CONTAINMENT_DESIGNS):
    a = run_means(n1, n2, s1, s2, d, NB_REPS, CONTAINMENT_SEED_BASE + i, known_sigma=True, test="z")
    b = run_means(n1, n2, s1, s2, d, NB_REPS, CONTAINMENT_SEED_BASE + 100 + i, known_sigma=True)
    print(f"  {{n1:>3}}/{{n2:<3}} sd {{s1}}/{{s2:<4}}  {{a['gap_and_not_sig']:>12.5f}}   {{b['gap_and_not_sig']:>16.5f}}")
''')

md("""
## 3. The dead zone - measured

95% t-intervals per group, Welch test at 0.05, normal data. `P(overlap | sig)` is the share of real,
significant results that a reader applying the overlap rule would throw away.
""")

code('''
dead = []
for i, (n1, n2, s1, s2, d) in enumerate(MEANS_DESIGNS):
    r = run_means(n1, n2, s1, s2, d, NB_REPS, MEANS_SEED_BASE + i)
    dead.append(r)
    print(f"  {n1:>3}/{n2:<3} sd {s1:g}/{s2:<3g}  P(sig) {r['sig']:.3f}   P(overlap | sig) {r['overlap_given_sig']:.3f}")
''')

md("""
## 4. The reversal on paired data

Pair the measurements and the test uses the correlation; the per-arm bars cannot show it. The
same picture now hides an overwhelming difference behind two bars sitting on top of each other.
""")

code('''
paired = []
for i, rho in enumerate(RHO_GRID):
    r = run_paired(30, rho, 0.35, 1.0, NB_REPS, PAIRED_SEED_BASE + i)
    paired.append(r)
    print(f"  rho {rho:.2f}   slack {r['slack']:.2f}x   P(sig) {r['sig']:.3f}   P(overlap | sig) {r['overlap_given_sig']:.4f}")
''')

md("""
## 5. The bars themselves - exact coverage

Before anyone compares two bars, is each one a 95% interval? For a proportion, coverage can be
computed **exactly** by summing over every possible count - no Monte Carlo at all.
""")

code('''
print("    n     p       wald   wilson   agresti   clopper")
cov = {}
for n in COVERAGE_NS:
    for p in COVERAGE_PS:
        row = [exact_coverage(m, p, n) for m in METHODS]
        cov[(n, p)] = row
        print(f"  {n:>4}  {p:>5.2f}   " + "   ".join(f"{v:.4f}" for v in row))
''')

md("""
## The picture

Left: the dead zone by design. Right: the reversal as correlation rises.
""")

code('''
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
lab = [f"{r['n1']}/{r['n2']}\\nsd {r['sd1']:g}:{r['sd2']:g}" for r in dead]
a1.bar(range(len(dead)), [r["overlap_given_sig"] for r in dead], color="#c8562b")
a1.set_xticks(range(len(dead)))
a1.set_xticklabels(lab, fontsize=7)
a1.set_ylabel("P(95% bars overlap | significant)")
a1.set_title("Real results the picture discards", loc="left", fontweight="bold")
a2.plot(RHO_GRID, [r["overlap_given_sig"] for r in paired], "o-", color="#c8562b",
        label="P(overlap | sig)")
a2.plot(RHO_GRID, [r["sig"] for r in paired], "s--", color="#2f6fdb", label="P(sig)")
a2.set_xlabel("correlation between paired measurements")
a2.set_ylim(0, 1.05)
a2.legend(frameon=False, loc="lower right")
a2.set_title("Paired data: the rule fails the other way", loc="left", fontweight="bold")
for ax in (a1, a2):
    ax.grid(True, color="#dfe4ea")
    ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=150)
plt.show()
''')

md("""
## Summary

| | |
|---|---|
| **Matching level** | bars agree with `p < 0.05` at **83.4%**, not 95% (equal standard errors) |
| **Rule size** | "bars do not touch" is a **0.6%** test labelled 5% |
| **Dead zone** | up to **52%** of significant results have overlapping 95% bars |
| **Paired data** | at rho = 0.95, **99.99%** of rejections sit behind overlapping bars |
| **Wald bars** | exact coverage **0.33** at p = 0.02, n = 20 - and zero width on zero events |

The fix is a caption, not a correction: draw the bars at the matching level and say so, or draw
the interval for the **difference** - the estimate the question was about in the first place.
""")

md("""
## Try your own
""")

code('''
# Your chart: two means, their SEs, and the correlation between them (0 if independent groups).
# m1, m2, se1, se2, rho = 10.2, 11.0, 0.35, 0.40, 0.0
# d, z = abs(m2 - m1), 1.959964
# print("bars overlap:", d <= z * (se1 + se2))
# se_d = math.sqrt(se1**2 + se2**2 - 2 * rho * se1 * se2)
# print("p =", 2 * stats.norm.sf(d / se_d))
# print("draw bars at", f"{matching_level(se1, se2) * 100:.1f}%", "to make overlap mean p < 0.05")
print("uncomment to explore")
''')

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you drag two bars around and watch the picture and the test disagree:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`effect-size-reader`](../effect-size-reader) (an estimate),
[`p-value-dance`](../p-value-dance) (p under replay),
[`prediction-interval`](../prediction-interval) (a different interval, about a future observation).
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
