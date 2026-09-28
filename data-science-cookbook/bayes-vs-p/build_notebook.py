"""Generate demo.ipynb.

Plumbing copied from `paired-power` (Day 184), content written fresh. The engine is EXTRACTED from
bayes.py by AST at build time, so the notebook holds the library's own source text. The write lives
behind __main__: test_notebook.py imports this module, and an import-time write would replace the
executed notebook with a blank one (the Day 177 defect).

The study numbers are exact (closed-form normal integrals + quadrature), so the notebook reproduces them to the digit.
Only the Monte Carlo cross-check runs at NB_REPS instead of the study's 20,000 - stated in the notebook (the
Day 179 resolution note), and a test asserts the note exists.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/bayes-vs-p"
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


ENGINE = extract("bayes.py")

EXPORTS = [
    "ALPHA", "Z_CRIT", "TAU", "R_JZS", "NS", "TAUS", "MC_REPS", "MC_DESIGNS", "Z99", "wilson", "bf01_normal",
    "bf01_normal_by_quadrature", "z_threshold_sq", "p_abs_below", "verdicts", "bf01_at_p",
    "n_where_p_supports_null", "max_bf10_over_normal_priors", "sbb_bound", "bf10_jzs", "bf10_jzs_by_delta",
    "analyse", "n_for_power", "calibrate", "monte_carlo", "exemplar",
]

HEADER = """from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, stats
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
# Two analysts, one dataset, two verdicts

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

10,000 users, a one-sample t-test, p = 0.047. Analyst one writes "significant". Analyst two runs the
default Bayesian t-test in JASP and writes "strong evidence of no effect" (BF01 = 12). Both ran their
method correctly. This notebook measures, exactly, when the two verdicts must split, how often, and what
decides it.

1. Calibration - closed form vs quadrature, two JZS integrals vs each other and vs pingouin, raw Monte Carlo
2. Lindley's curve - hold p at 0.05 and grow n
3. The ceiling - the most any normal prior can make of p = 0.05
4. When the null is true - how many "findings" the Bayes factor calls null
5. Negative result - a real but tiny effect: the Bayes factor swaps one error for another
6. Prior sensitivity - one dataset, six Bayes factors
7. Try your own
""")

md("""
## The engine

Extracted from `bayes.py` at build time, character for character. The study model is a normal mean with
known unit SD, so z = mean * sqrt(n) is exactly N(delta * sqrt(n), 1). Against H1: delta ~ N(0, tau^2),

BF01 = sqrt(1 + n tau^2) * exp(-z^2/2 * n tau^2 / (1 + n tau^2)),

and "BF01 > k" is the event |z| < a threshold - so every verdict probability is a closed-form normal
integral. Real data use the t statistic and the JZS (Cauchy prior) Bayes factor that JASP, BayesFactor and
pingouin report.
""")

code(HEADER + "\n" + ENGINE + "\n\nprint('engine loaded')")

md(f"""
**Resolution of the instrument.** Every study number here is exact and matches `evidence.txt` to the
digit. Only the raw-data Monte Carlo cross-check runs at {NB_REPS:,} replicates instead of the study's
20,000, so its intervals are about 2.2x wider - lower resolution, not disagreement. `evidence.txt` is
the reference.
""")

md("## 1. Calibration")

code(f"""
print(calibrate())
for r in monte_carlo({NB_REPS}):
    print(f"n={{r['n']:>5}} delta={{r['delta']:<5}} " + "  ".join(
        f"{{k}}: {{r[k]['exact']:.4f}}/{{r[k]['mc']:.4f}} {{'in' if r[k]['inside_99'] else 'OUT'}}"
        for k in ("p_significant", "bf01_over_3", "sig_and_bf01_over_1")))
""")

md("""
The closed form and the explicit integral over the prior agree to about 1e-15 in log. The JZS Bayes
factor computed over g and computed over delta (an independent route through the noncentral t) agree to
about 1e-13, and both agree with `pingouin.bayesfactor_ttest` to about 1e-11. The t-test makes the same
decision as `scipy.stats.ttest_1samp` on 400 of 400 datasets. In the 20,000-rep study all 15 Monte Carlo
checks land inside their 99% interval.
""")

md("## 2. Lindley's curve - p held at exactly 0.05")

code("""
print(f"{'n':>9}   BF01 tau=1   BF01 JZS   BF01 tau=0.1")
for n in NS:
    t = float(stats.t.isf(ALPHA / 2, n - 1))
    print(f"{n:>9}   {bf01_at_p(0.05, n):>10.3f}   {1 / bf10_jzs(t, n):>8.3f}   {bf01_at_p(0.05, n, 0.1):>12.3f}")
for p in (0.05, 0.01, 0.005):
    print(f"p = {p}: evidence FOR the null (BF01 >= 1) from n = {n_where_p_supports_null(p):,.1f}, "
          f">= 3 from n = {n_where_p_supports_null(p, 3):,.1f}")
""")

md("""
With the unit-information prior a p = 0.05 result is already evidence for the null at n = 42, 3:1 for
it at n = 415, and 130:1 at a million (JZS). Even p = 0.005 - the "redefine significance" threshold -
tips over at n = 2,634. The curve is U-shaped in n: it dips first, then rises like sqrt(n). The first
version of the root finder landed on the dip and returned log n instead of n; both defects have tests.
""")

md("## 3. The ceiling")

code("""
mx = max_bf10_over_normal_priors(Z_CRIT)
print(f"best possible normal prior at p = 0.05: BF10 = {mx['closed_form']:.3f}  (search: {mx['by_search']:.3f})")
print(f"Sellke-Bayarri-Berger bound 1/(-e p ln p) at p = 0.05: {sbb_bound(0.05):.3f}")
""")

md("""
Choose the prior AFTER seeing the data, to flatter H1 as much as possible, and p = 0.05 is still only
2.1 to 1 for an effect. That is the gap the two analysts are arguing across: a p of 0.05 sounds like
"1 in 20", and as evidence it is worth about 2 to 1.
""")

md("## 4. When the null is true")

code("""
print(f"{'n':>9}   P(p<.05)   P(BF10>3)   share of 'findings' the BF calls null")
for n in NS:
    v = verdicts(n, 0.0)
    print(f"{n:>9}   {v['p_significant']:.4f}     {v['bf10_over_3']:.4f}      {v['sig_and_bf01_over_1'] / v['p_significant']:.1%}")
""")

md("""
The p-value's false-positive rate is 0.05 at every n - that is its whole contract. The Bayes factor's
chance of a false "evidence for an effect" falls toward zero, and at n = 10,000, 95% of the null's
"significant" results come with a Bayes factor pointing the other way.
""")

md("## 5. Negative result - a real but tiny effect")

code("""
tn = [1000, 5000, 10000, 20000, 50000, 100000, 1000000]
rows = [verdicts(n, 0.02) for n in tn]
for v in rows:
    print(f"n = {v['n']:>8,}   P(p<.05) {v['p_significant']:.3f}   P(BF10>3) {v['bf10_over_3']:.3f}   "
          f"P(BF01>3, 'no effect') {v['bf01_over_3']:.3f}")
for d in (0.5, 0.2, 0.1, 0.05):
    print(f"delta = {d}: n for 80% -> p-test {n_for_power(d, 'p_significant')}, Bayes factor {n_for_power(d, 'bf10_over_3')}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].loglog(NS, [bf01_at_p(0.05, n) for n in NS], "o-", color="#c8562b", label="tau = 1")
axes[0].loglog(NS, [bf01_at_p(0.05, n, 0.1) for n in NS], "s-", color="#b8860b", label="tau = 0.1")
axes[0].axhline(1, color="#1f2733", lw=0.8)
axes[0].set_xlabel("n (p fixed at 0.05)"); axes[0].set_ylabel("BF01"); axes[0].legend(frameon=False)
axes[0].set_title("Lindley's curve", loc="left")
axes[1].semilogx(tn, [v["p_significant"] for v in rows], "o-", color="#2f6fdb", label="p < 0.05")
axes[1].semilogx(tn, [v["bf10_over_3"] for v in rows], "o-", color="#c8562b", label="BF10 > 3")
axes[1].semilogx(tn, [v["bf01_over_3"] for v in rows], "o--", color="#c8562b", label="BF01 > 3 ('no effect')")
axes[1].set_xlabel("n"); axes[1].set_ylabel("P(verdict), true delta = 0.02"); axes[1].legend(frameon=False)
axes[1].set_title("A real, tiny effect", loc="left")
plt.tight_layout(); plt.savefig("notebook_chart.png", dpi=120); plt.show()
""")

md("""
The tempting conclusion from sections 2-4 is "use Bayes factors, they fix p-values". This is where it
fails. The effect here is real (0.02 SD). At n = 20,000 the p-value finds it 81% of the time; the Bayes
factor says "evidence of NO effect" 48% of the time. It is not malfunctioning - on a prior that expects
effects near 1 SD, 0.02 really is closer to zero - but it has traded false positives for false "no
effect" verdicts on small true effects, and it needs 1.4x to 2.2x the sample for the same 80% chance
of finding an effect.
""")

md("## 6. Prior sensitivity - one dataset, six Bayes factors")

code("""
ex = exemplar()
print(f"seed {ex['seed']}: n = {ex['n']:,}, observed d = {ex['d']:.4f}, t = {ex['t']:.3f}, p = {ex['p']:.4f}")
print(f"JZS BF01 at the default r = 0.707: {ex['bf01']:.2f}")
for tau, v in ex["bf01_by_tau"].items():
    print(f"  normal prior tau = {tau:<5} BF01 = {v:.3f}")
""")

md("""
The same 10,000 numbers are 12:1 for "no effect" under the software default, 28:1 under a wide prior,
and 1.3:1 for "an effect" under a prior that expects effects of 0.05 SD. The Bayes factor is not a
property of the data; it is a comparison of the data against a stated alternative, and the alternative
has to be justified from the domain before the data arrive.

## Summary

| claim | what the numbers say |
|---|---|
| "p = 0.05 means strong evidence" | the best any normal prior can do is 2.1 to 1 |
| "a significant result is evidence against the null" | at tau = 1 a p = 0.05 result favours the null from n = 42 |
| "Bayes factors fix this" | a real 0.02 SD effect at n = 20k: 48% chance of 'evidence of no effect' |
| "the Bayes factor is objective" | one dataset: BF01 from 0.76 to 27.6 as the prior width goes 0.05 to 2 |
""")

md("## 7. Try your own")

code("""
# values = [0.8, -0.2, 1.4, 0.3, 0.9, -0.5, 1.1, 0.6, 0.2, 1.3]   # one-sample, or after - before
# r = analyse(values, r=0.707)
# print(round(r["p"], 4), round(r["bf01"], 3))
# a reported result: n and p only
# n, p = 50000, 0.03
# t = float(stats.t.isf(p / 2, n - 1)); print("BF01 =", round(1 / bf10_jzs(t, n), 2))
print("uncomment to explore")
""")

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

Streamlit version, where you enter your own n and p (or raw values) and move the prior:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sibling builds: [`p-value-dance`](../p-value-dance) showed a p-value is itself a random variable;
[`equivalence-test`](../equivalence-test) gives the frequentist route to "no difference";
[`peeking-cost`](../peeking-cost) measured what optional stopping does to a p-value.
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
