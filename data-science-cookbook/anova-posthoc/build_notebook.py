"""Generate demo.ipynb.

Plumbing copied from `proportion-test` (Day 181), content written fresh. The engine is EXTRACTED
from posthoc.py by AST at build time, so the notebook holds the library's own source text. The
write lives behind __main__: test_notebook.py imports this module, and an import-time write would
replace the executed notebook with a blank one (the Day 177 defect).

The notebook runs the study at NB_REPS replicates, not the study's 20,000, using the library's own
seeds. Fewer replicates means wider intervals, so it can flag FEWER cells than evidence.txt - that
is stated in the notebook (the Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/anova-posthoc"
NB_REPS = 4000


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



ENGINE = extract("posthoc.py")

EXPORTS = [
    "ALPHA", "PROCS", "LABEL", "BAND_LO", "BAND_HI", "Z99", "N_PER_GROUP", "KS", "wilson", "verdict",
    "means_for", "pairs", "decisions", "holm_reject", "q_crit", "simulate", "run_design",
    "null_table", "partial_table", "power_table", "identity_checks", "calibrate_against_scipy",
    "analyse", "exemplar", "NULL_DESIGNS", "PARTIAL_DESIGNS", "POWER_DESIGNS", "SEED_BASE",
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
# ANOVA says something differs. Which pair?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A significant F says the k group means are not all equal. It does not say which pair differs, and
there are k(k-1)/2 pairs to hunt through. Five procedures do the hunting - no adjustment, Fisher's
protected LSD, Bonferroni, Holm, Tukey's HSD - and this notebook measures what each one costs.

1. Calibration - against scipy, and three identities counted
2. All means equal - the family-wise false-alarm rate as k grows
3. One group far away - where "protected" LSD stops protecting
4. Power - Tukey vs Holm, as a paired difference with its own noise
5. "Something differs" and nothing the post-hoc test will name
6. One dataset, five answers
7. Try your own
""")

md("""
## The engine

Extracted from `posthoc.py` at build time, character for character.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** The study behind the README runs 20,000 replicates per design.
This notebook runs {NB_REPS:,} with the SAME seeds, so every Wilson interval is about 2.2x wider:
a cell the study calls CONSERVATIVE can read `ok` here. That is lower resolution, not
disagreement - `evidence.txt` is the reference.
""")

md("""
## 1. Calibration
""")

code("""
print(calibrate_against_scipy())
print(identity_checks(reps=2000))
""")

md("""
Tukey's decisions match `scipy.stats.tukey_hsd` pair for pair. Holm never misses a pair Bonferroni
finds, and under the complete null the two agree on *whether anything* is significant on every
replicate - so their family-wise error is identical there by construction. Holm's advantage can
only ever show up as power.
""")

md("""
## 2. All means equal

Every "significant pair" is a false alarm. FWER = P(at least one pair called).
""")

code(f"""
NB_REPS = {NB_REPS}
null = null_table(NB_REPS)
print("  k " + "".join(f"{{LABEL[p]:>16}}" for p in PROCS))
for r in null:
    print(f"{{r['k']:>3}} " + "".join(f"{{r[p + '_fwer']:>9.4f}} {{r[p + '_fwer_verdict'][:5]:<6}}" for p in PROCS))
""")

md("""
With no adjustment the family calls a winner more than half the time by k = 8. Tukey sits on 5% at
every k - it is exact for this setting, so it doubles as the harness's calibration line.
Bonferroni and Holm run BELOW 5% as k grows: the pairwise tests share group means and are
correlated, and Bonferroni's bound assumes the worst case.
""")

md("""
## 3. One group far away

One group 3 SD from the rest, so the F test fires on essentially every replicate, and the
remaining k-1 groups are exactly equal. Any pair among them that gets called is a false alarm.
""")

code("""
partial = partial_table(NB_REPS)
print("  k  F sig " + "".join(f"{LABEL[p]:>16}" for p in PROCS))
for r in partial:
    print(f"{r['k']:>3} {r['F_sig']:>6.3f} " + "".join(f"{r[p + '_fwer']:>9.4f} {r[p + '_fwer_verdict'][:5]:<6}" for p in PROCS))
""")

md("""
Protected LSD is the "no adjustment" column exactly: once the F test has fired on the obvious group,
the protection is spent and every other pair is tested at a raw 0.05. It holds only at k = 3, where
one null pair is left. This is the textbook result that LSD controls family-wise error in the
*weak* sense (complete null) and not the *strong* sense (any configuration).
""")

md("""
## 4. Power

Per-pair power = share of truly different pairs a procedure names. A procedure is compared only if
its FWER in section 3 is not INFLATED at that k - otherwise its lead is the inflation restated
(marked `*`). Tukey minus Holm is a PAIRED difference on the same replicates, called a win only if
it clears its own 99% interval.
""")

code("""
power = power_table(partial, NB_REPS)
print("design        " + "".join(f"{LABEL[p]:>16}" for p in PROCS) + "   Tukey-Holm   winner")
for r in power:
    print(f"k={r['k']:>2} {r['shape']:<8}" + "".join(f"{r[p + '_per_pair']:>15.4f}{' ' if r[p + '_comparable'] else '*'}" for p in PROCS)
          + f"   {r['tukey_minus_holm']:+.4f}     {r['tukey_vs_holm']}")
""")

md("""
Tukey beats Holm from k = 5 up; at k = 3 Holm wins. With only three pairs Holm's step-down
thresholds (0.0167, 0.025, 0.05) are generous, and the studentized-range penalty is not yet paid
back. "Always use Tukey" and "Holm is uniformly better than Bonferroni" are both true - and Holm
still loses to Tukey at the k most people have.
""")

md("""
## 5. Something differs - and no pair is named

On the ladder designs every pair truly differs. How often is F significant while the post-hoc
procedure names nothing?
""")

code("""
spread = [r for r in power if r["shape"] == "spread"]
for r in spread:
    print(f"k={r['k']:>2}  P(F sig) {r['F_sig']:.3f}   P(no Tukey pair | F sig) {r['tukey_no_pair_given_F']:.3f}"
          f"   P(no Holm pair | F sig) {r['holm_no_pair_given_F']:.3f}   P(Tukey pair, F not sig) {r['tukey_pair_no_F']:.3f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
ks = [r["k"] for r in null]
cols = {"none": "#c8562b", "lsd": "#b8860b", "bonferroni": "#6b7684", "holm": "#2f7a5a", "tukey": "#2f6fdb"}
for p in PROCS:
    a1.plot(ks, [r[p + "_fwer"] for r in partial], marker="o", color=cols[p], label=LABEL[p])
a1.axhline(ALPHA, color="#1f2733", lw=0.8)
a1.set_yscale("log")
a1.set_xlabel("k")
a1.set_ylabel("FWER, one group far away")
a1.legend(frameon=False, fontsize=8)
for p in ("holm", "tukey"):
    a2.plot([r["k"] for r in spread], [r[p + "_no_pair_given_F"] for r in spread], marker="o", color=cols[p], label=LABEL[p])
a2.set_xlabel("k (evenly spaced means)")
a2.set_ylabel("P(no pair named | F significant)")
a2.legend(frameon=False)
plt.tight_layout()
plt.savefig("notebook_chart.png", dpi=150)
plt.show()
""")

md("""
The two answers disagree in both directions. F can be significant with no nameable pair (the
evidence is spread across a contrast no single pair captures), and Tukey can name a pair while F is
not significant. So "run the F test first, and only then the post-hoc" silently discards some
real Tukey findings - Tukey controls family-wise error on its own and does not need the gate.
""")

md("""
## 6. One dataset, five answers

Chosen by rule: the first seed of a 5-group ladder where F is significant, Tukey names nothing, and
unadjusted t-tests name at least one pair.
""")

code("""
ex = exemplar()
print(f"seed {ex['seed']}   group means " + ", ".join(f"{m:.2f}" for m in ex["means"]) + f"   F p = {ex['F_p']:.4f}")
for p in PROCS:
    print(f"  {LABEL[p]:<16} " + (", ".join(f"{chr(65+i)}-{chr(65+j)}" for i, j in ex["significant"][p]) or "none"))
""")

md("""
## Summary

| | |
|---|---|
| **No adjustment** | FWER 0.12 at k = 3 up to 0.62 at k = 10 |
| **Protected LSD** | fine when all means are equal; inflated at every k >= 4 once one group is clearly different |
| **Bonferroni / Holm** | identical FWER under the complete null; both conservative as k grows |
| **Tukey HSD** | exact FWER; beats Holm on power from k = 5, loses to it at k = 3 |
| **F vs post-hoc** | F significant with no Tukey pair up to about 1 in 9 at k = 10 |

For all-pairs comparisons with roughly equal n: **Tukey**, without waiting for the F gate. At k = 3,
Holm is the more powerful choice. Never protected LSD beyond three groups.
""")

md("""
## Try your own
""")

code("""
# groups = [
#     [12.1, 11.8, 13.0, 12.4, 12.9],
#     [13.2, 13.8, 12.9, 14.1, 13.5],
#     [12.0, 12.6, 11.9, 12.3, 12.8],
#     [14.0, 13.7, 14.4, 13.9, 14.6],
# ]
# res = analyse(groups)
# print("F p =", round(res["F_p"], 4))
# for p in PROCS:
#     print(LABEL[p], res["significant"][p])
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own groups:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`t-test-variants`](../t-test-variants) audited the two-group case;
[`stat-test-advisor`](../stat-test-advisor) picks a test; [`p-value-dance`](../p-value-dance)
measured the winner's curse that a family of pairwise tests multiplies.
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
