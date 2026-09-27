"""Generate demo.ipynb.

Plumbing copied from `equivalence-test` (Day 183), content written fresh. The engine is EXTRACTED from
paired.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The study numbers are exact (noncentral t + quadrature), so the notebook reproduces them to the digit.
Only the Monte Carlo cross-check runs at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/paired-power"
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


ENGINE = extract("paired.py")

EXPORTS = [
    "ALPHA", "DELTA", "RHOS", "NS", "GL_NODES", "MC_REPS", "Z99", "MC_DESIGNS", "wilson", "sd_diff",
    "power_paired", "power_independent", "n_for_power", "break_even_rho", "analyse", "grid", "calibrate",
    "monte_carlo", "exemplar",
]

HEADER = """from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats
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
# You have before and after. Analyse it as before and after.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Twelve stores, weekly sales before and after a process change. Someone puts "before" in one column,
"after" in another, and runs a two-sample t-test. That test counts every store's own level as noise.
This notebook measures, exactly, how much power that throws away at each level of pairing - and the
one corner where pairing costs you instead.

1. Calibration - the quadrature against Student's noncentral t, scipy, and raw Monte Carlo
2. Size: the wrong analysis is conservative at rho > 0 and anti-conservative at rho < 0
3. Power of the two analyses of the same data
4. Pairs needed for 80% power - the sample the wrong analysis wastes
5. Negative result: the break-even rho below which pairing costs power
6. One report, read properly
7. Try your own
""")

md("""
## The engine

Extracted from `paired.py` at build time, character for character. The paired test is a noncentral t.
The two-sample test on paired data is not: its denominator mixes two correlated variances. Under
normality, (n-1)(s_b^2 + s_a^2) = (1+rho)A + (1-rho)B with A, B independent chi-square(n-1), and the
mean difference is independent of both - so a 2-D Gauss-Legendre quadrature over A and B is exact.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every study number here is exact and matches `evidence.txt` to the
digit. Only the raw-data Monte Carlo cross-check runs at {NB_REPS:,} replicates instead of the study's
20,000, so its intervals are about 2.2x wider - lower resolution, not disagreement. `evidence.txt` is
the reference. A design that misses its 99% interval is re-run at 10x the reps on a fresh seed, and
both results are printed.
""")

md("## 1. Calibration")

code(f"""
print(calibrate())
for r in monte_carlo({NB_REPS}):
    for k in ("paired", "independent"):
        x = r[k]
        extra = f"  recheck {{x['recheck']['mc']:.4f}}" if "recheck" in x else ""
        print(f"n={{r['n']:>2}} rho={{r['rho']:>5}} delta={{r['delta']:.1f}} {{k:>11}}: exact {{x['exact']:.4f}} "
              f"MC {{x['mc']:.4f}} {{'inside' if x['inside_99'] else 'OUTSIDE'}}{{extra}}")
""")

md("""
At rho = 0 the quadrature agrees with Student's noncentral t to about 3e-7, doubling the nodes moves
it by about 1e-7, and the vectorised tests make the same decision as `scipy.stats.ttest_rel` /
`ttest_ind` on 800 of 800 random datasets.

In the full 20,000-rep study one of 12 checks missed (n = 40, rho = 0.95, exact 0.7951, MC 0.8025).
Twelve checks at 99% miss once about 11% of the time by chance; re-run at 200,000 reps it lands at
0.7949. Both runs are in `evidence.txt`.
""")

md("""
## 2. Size - how often each analysis rejects a TRUE null

The paired test is exact at 0.05 for every rho. The two-sample test on the same data is not.
""")

code("""
rows = grid()
def cell(n, rho):
    return next(r for r in rows if r["n"] == n and r["rho"] == rho)
print(" n pairs" + "".join(f"{r:>8}" for r in RHOS))
for n in NS:
    print(f"{n:>7} " + "".join(f"{cell(n, r)['size_independent']:>8.4f}" for r in RHOS))
""")

md("""
At rho > 0 the two-sample test is conservative: 0.0033 at rho = 0.6, n = 20. It looks "safe", and that
is exactly why it survives review. At rho < 0 - a see-saw, a budget moved from one line to another -
it is **anti-conservative**: 0.109 at rho = -0.5, more than double its label.
""")

md(f"## 3. Power at a true shift of {0.5} SD - paired / two-sample, same data")

code("""
print(" n pairs" + "".join(f"{'rho=' + str(r):>14}" for r in RHOS))
for n in NS:
    print(f"{n:>7} " + "".join(f"{cell(n, r)['power_paired']:>6.3f}/{cell(n, r)['power_independent']:<7.3f}" for r in RHOS))
""")

md("""
At n = 20 and rho = 0.8 the paired test finds the shift 92% of the time; the two-sample test on the
same numbers finds it 22% of the time. And as the pairing tightens the two-sample test gets WORSE at
small n (0.108 -> 0.025 at n = 5): its false-positive rate falls toward zero and takes its power with it.
""")

md("## 4. Pairs needed for 80% power")

code("""
for rho in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95):
    a, b = n_for_power(0.8, rho, "paired"), n_for_power(0.8, rho, "independent")
    print(f"rho = {rho}   paired {a:>3}   two-sample {b:>3}   wasted {b - a:>3} pairs ({b / a:.1f}x)")
""")

md("""
At rho = 0.9 the paired analysis needs 9 pairs and the wrong analysis 43. The two-sample test cannot
go below about 31 pairs whatever rho is, because it prices the mean difference at variance 2/n; the
paired test prices it at its true 2(1-rho)/n.
""")

md("""
## 5. Negative result - pairing is not free

The paired test uses n-1 degrees of freedom instead of 2n-2. If the pairing is weak, that loss is
bigger than what the correlation buys back.
""")

code("""
for n in (3, 5, 10, 20, 40, 80):
    print(f"n = {n:>3} pairs   break-even rho = {break_even_rho(n):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
rs = list(RHOS)
axes[0].plot(rs, [cell(20, r)["power_paired"] for r in rs], "o-", color="#2f6fdb", label="paired t")
axes[0].plot(rs, [cell(20, r)["power_independent"] for r in rs], "o-", color="#c8562b", label="two-sample t, same data")
axes[0].plot(rs, [cell(20, r)["size_independent"] for r in rs], "--", color="#c8562b", label="two-sample false-positive rate")
axes[0].axhline(ALPHA, color="#6b7684", lw=0.8, ls=":")
axes[0].set_xlabel("rho"); axes[0].set_ylabel("P(reject), n = 20 pairs"); axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("Same data, two analyses", loc="left")
ns_ = [3, 5, 10, 20, 40, 80]
axes[1].plot(ns_, [break_even_rho(n) for n in ns_], "o-", color="#b8860b")
axes[1].set_xscale("log"); axes[1].set_xlabel("n pairs"); axes[1].set_ylabel("break-even rho")
axes[1].set_title("Below this rho, pairing costs power", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
The break-even is small: 0.107 at 3 pairs, 0.020 at 80. The cost below it is small too (at n = 5,
rho = 0: 0.095 vs 0.108). So this is not a reason to ignore pairing - measurements on the same unit
almost always correlate far above 0.1 - but it is why "always pair" is not a free lunch on tiny samples
with a weak link. And the choice must be made from the design, not from which p-value came out smaller.
""")

md("## 6. One report, read properly")

code("""
ex = exemplar()
print(f"seed {ex['seed']}: 12 stores, SD $4k, rho 0.85, true lift $2k")
print(f"observed lift ${ex['delta']:.2f}k, sample rho {ex['rho_hat']:.3f}")
print(f"paired t p = {ex['p_paired']:.4f}    two-sample t p = {ex['p_independent']:.4f}")
print(f"at this design only the paired test rejects {ex['p_only_paired']:.1%} of the time; "
      f"only the two-sample test {ex['p_only_independent']:.3%}")
""")

md("""
The same twelve pairs are a finding (p = 0.041) or "no significant difference" (p = 0.254) depending on
whether the analysis remembers that each row is one store. At this design that disagreement is not a
fluke: it happens on 75% of datasets.

## Summary

| claim | what the numbers say |
|---|---|
| "two-sample t on paired data is just conservative" | at rho = 0.8, n = 20 it keeps 22% power of the paired test's 92% |
| "it is at least safe" | not at rho < 0: false-positive rate 0.109 at rho = -0.5 |
| "pairing is free" | no: below rho ~0.02-0.11 (depending on n) it costs a little power |
| the sample the wrong analysis wastes | 34 of 43 pairs at rho = 0.9 |
""")

md("## 7. Try your own")

code("""
# before = [12.1, 14.3, 9.8, 11.0, 13.5, 10.2, 12.8, 11.9]
# after  = [12.9, 15.0, 10.1, 11.8, 14.6, 10.4, 13.9, 12.3]
# r = analyse(before, after)
# print(round(r["rho_hat"], 3), round(r["p_paired"], 4), round(r["p_independent"], 4))
# print("pairs for 80% power at rho 0.7:", n_for_power(0.8, 0.7, "paired"), "vs", n_for_power(0.8, 0.7, "independent"))
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you paste your own before and after:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`t-test-variants`](../t-test-variants) measured the five t-tests' error rates;
[`ci-overlap-fallacy`](../ci-overlap-fallacy) showed that on paired data overlapping bars hide
significant results; [`cuped-variance`](../cuped-variance) buys the same variance reduction from a
pre-period covariate in an A/B test.
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
