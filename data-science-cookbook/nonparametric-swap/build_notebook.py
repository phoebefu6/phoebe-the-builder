"""Generate demo.ipynb.

The engine block is EXTRACTED FROM nonparam.py BY AST AT BUILD TIME rather than hand-copied,
so the notebook holds the library's own source text and cannot go stale. That pattern started in
the sibling build `p-value-dance` (Day 177) and it carries a trap worth restating: this module
must NOT write the notebook at import time, because test_notebook.py imports it - unguarded, that
import silently replaces the executed, pre-rendered notebook with a blank one, and no later test
can see it because the test suite is what did it. The write lives behind __main__ and a regression
test asserts the import leaves the file alone.

The notebook also uses the library's own DESIGN LISTS and seeds (the seed is the index into the
list), so its printed rows ARE the cells in evidence.txt rather than a second sample of them.
Replicate counts are cut where a cell would otherwise dominate the nbconvert run; every such cut
is stated in the cell that makes it, because a quieter number read against a loud one is exactly
how Day 175's notebook came to contradict its own evidence file.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/nonparametric-swap"

EXPORTS = [
    "ALPHA", "Z99", "CHUNK", "BAND_LO", "BAND_HI",
    "LOGNORMAL_S", "CONTAM_SCALE", "CONTAM_SHARE",
    "wilson", "mc_interval", "verdict",
    "_standard", "SHAPES", "SYMMETRIC",
    "density", "integral_f_squared", "density_mass", "asymptotic_are", "ARE_CLOSED_FORM",
    "MIX_SD", "MIX_WEIGHTS", "MIX_MEANS",
    "mix_mean", "mix_sd", "mix_prob_superiority", "mix_cohens_d", "draw_disagreement",
    "welch", "mannwhitney", "mannwhitney_uncorrected", "_chunks",
    "run_disagreement", "run_location", "comparable", "_monotone_search",
    "efficiency_from_curve", "run_unequal_spread",
    "to_levels", "tie_variance_ratio", "run_ties",
    "STUDY_REPS", "NULL_REPS", "DISAGREE_REPS", "TIE_REPS",
    "DISAGREE_NS", "LOCATION_D", "LOCATION_NS", "LOCATION_REF_N",
    "UNEQUAL_RATIOS", "UNEQUAL_NS", "TIE_LEVELS", "TIE_N", "TIE_D",
    "DISAGREE_SEED_BASE", "LOCATION_SEED_BASE", "NULL_SEED_BASE",
    "UNEQUAL_SEED_BASE", "TIE_NULL_SEED_BASE", "TIE_POWER_SEED_BASE",
    "LOCATION_DESIGNS", "UNEQUAL_DESIGNS",
]


def extract(path: str, names: List[str]) -> str:
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        # A node is taken if ANY of the names it binds was asked for. The tuple case is not
        # hypothetical: `BAND_LO, BAND_HI = 0.045, 0.055` binds two names through a Tuple target,
        # and an extractor that only looks at `targets[0].id` skips it silently - the notebook
        # then fails at nbconvert time with a NameError from inside a function body, several
        # minutes into the run.
        bound: List[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = [node.name]
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                bound += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        elif isinstance(node, ast.AnnAssign):
            bound = [n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)]
        if any(b in names for b in bound):
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n".join(out).rstrip()


ENGINE = extract("nonparam.py", EXPORTS)

HEADER = """from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
# "The data is not normal, just use Mann-Whitney"

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

It is the most-repeated piece of statistical advice in applied work, and it quietly swaps one
question for another.

A two-sample t-test asks about **`mu_B - mu_A`**. The Mann-Whitney U test asks about
**`P(B > A)`**. Shift one distribution sideways and those two agree in sign, so nobody notices.
Step off that model and they can point in **opposite directions** - both correctly, on the same
data - and at a large enough sample **both will be significant while disagreeing**.

This notebook builds a design where that is arithmetic rather than bad luck, and then measures
what the two tests actually do with it.

1. **Calibration** - reproduce three closed forms before reporting anything that has none
2. **The headline** - one dataset, two significant results, opposite signs
3. **Where the swap is fine** - power under a pure location shift, gated
4. **The cost nobody mentions** - unequal spread, where the Mann-Whitney null is exactly true
5. **Ties** - a rank test on a rating scale
""")

md("""
## Setup

The engine is the repository's own `nonparam.py`, extracted verbatim at build time so this
notebook and the library cannot disagree. The design lists and seeds are the library's too, so
the rows printed below are the same cells that appear in `evidence.txt`.
""")

code(f'''
{HEADER}
{ENGINE}

import matplotlib
import matplotlib.pyplot as plt

print("shapes:", SHAPES)
print("disagreement design: control N(0, {{MIX_SD}}^2), treated", 
      " + ".join(f"{{w}}*N({{m}}, {{MIX_SD}}^2)" for w, m in zip(MIX_WEIGHTS, MIX_MEANS)))
print(f"alpha = {{ALPHA}}, a rate is only called broken when its whole 99% Wilson interval")
print(f"clears [{{BAND_LO}}, {{BAND_HI}}]")
''')

md("""
## 1. Calibration

Under a **location shift** the asymptotic relative efficiency of Mann-Whitney to the t-test is

    ARE = 12 * sigma^2 * (integral of f^2)^2

and every population here is standardised to `sigma = 1`, so the whole thing is that one
integral. Three of the six shapes have a closed form. If those three do not land, nothing the
other three say is worth reading - so they are checked first.
""")

code('''
print(f"{'shape':<14} {'integral f':>12} {'integral f^2':>14} {'ARE':>9} {'closed form':>13} {'match':>7}")
ares = {}
for shape in SHAPES:
    mass = density_mass(shape)
    i2 = integral_f_squared(shape)
    are = asymptotic_are(shape)
    ares[shape] = are
    exact = ARE_CLOSED_FORM.get(shape)
    ok = abs(mass - 1.0) < 1e-6 and (exact is None or abs(are - exact) < 1e-6)
    print(f"{shape:<14} {mass:>12.6f} {i2:>14.6f} {are:>9.4f} "
          f"{(f'{exact:.6f}' if exact is not None else '-'):>13} {('OK' if ok else 'FAIL'):>7}")

print("\\nnormal -> 3/pi, uniform -> exactly 1, centred exponential -> exactly 3.")
print("Above 1 means Mann-Whitney needs FEWER observations for the same power.")
''')

md("""
And the hand-rolled rank test used in section 5 has to be SciPy's asymptotic test whenever there
are no ties, or that section measures an implementation difference instead of a tie effect.
""")

code('''
rng = np.random.default_rng(90_001)
a = rng.standard_normal((400, 25))
b = rng.standard_normal((400, 25)) + 0.3
pc, _ = mannwhitney(a, b)
pu = mannwhitney_uncorrected(a, b)
print(f"max |p_scipy - p_handrolled| with no ties: {float(np.max(np.abs(pc - pu))):.3e}")
''')

md("""
## 2. The headline

Here is the design. Nothing about it is exotic - it is *"most treated cases get a little worse,
a minority get a lot better"*, which is the shape of a great many real treatment effects.

    control    N(0, 0.5^2)
    treated    0.7 * N(-1, 0.5^2)  +  0.3 * N(5, 0.5^2)

Both of the numbers that matter are closed-form, so the disagreement below is not something the
simulation produced. It is arithmetic.
""")

code('''
print(f"true mean difference        mu_B - mu_A = {mix_mean():+.4f}")
print(f"  -> the t-test's target says treated is HIGHER")
print(f"true P(treated > control)               = {mix_prob_superiority():.5f}")
print(f"  -> Mann-Whitney's target says treated is LOWER")
print()
print(f"treated SD           = {mix_sd():.4f}")
print(f"population Cohen's d = {mix_cohens_d():.4f}")
''')

md("""
Now hand that to both tests, thousands of times, at a range of sample sizes.

The replicate count here is cut from the study's 40,000 to keep the notebook quick; the seeds are
the library's, so these rows are the evidence file's cells measured at lower precision, not a
different experiment.
""")

code('''
NB_REPS = 6_000
rows = []
print(f"{'n/group':>8} {'t sig UP':>10} {'t sig DOWN':>11} {'MW sig UP':>11} {'MW sig DOWN':>12} "
      f"{'BOTH, OPPOSITE':>15}")
for i, n in enumerate(DISAGREE_NS):
    r = run_disagreement(n, NB_REPS, DISAGREE_SEED_BASE + i)
    rows.append(r)
    print(f"{n:>8} {r['t_up']:>10.4f} {r['t_down']:>11.4f} {r['mw_up']:>11.4f} "
          f"{r['mw_down']:>12.4f} {r['opposite']:>15.4f}")

print()
print("The last column is two tests, one dataset, both p < 0.05, opposite conclusions about")
print("the direction of the effect.")
''')

md("""
Look at the shape of that last column rather than its size. It **rises** with n.

More data does not resolve the contradiction - more data guarantees it. At small samples the two
tests fail to disagree only because neither has the power to say anything at all. The
contradiction arrives exactly when both tests become trustworthy.
""")

code('''
rising = all(rows[i]["opposite"] <= rows[i + 1]["opposite"] for i in range(len(rows) - 1))
print(f"monotone increasing in n: {rising}")

small = rows[0]
share = small["t_down"] / (small["t_up"] + small["t_down"])
print(f"\\nSide finding at n = {int(small['n'])}: the t-test fires DOWNWARD {small['t_down']:.4f}")
print(f"of the time against {small['t_up']:.4f} upward - {share:.1%} of its significant results")
print("at that size point the wrong way about its OWN target. A sample that happens to miss the")
print("high component has both a low mean and a low variance, and the low variance shrinks the")
print("standard error the low mean is divided by.")
''')

md("""
### The picture behind the numbers

A table cannot carry this. Two densities can.
""")

code('''
fig, ax = plt.subplots(figsize=(11, 5))
x = np.linspace(-4.5, 7.2, 1200)
control = stats.norm.pdf(x, 0.0, MIX_SD)
treated = sum(w * stats.norm.pdf(x, m, MIX_SD) for w, m in zip(MIX_WEIGHTS, MIX_MEANS))
ax.fill_between(x, control, color="#2f6fdb", alpha=0.16)
ax.plot(x, control, color="#2f6fdb", lw=2.2, label="control")
ax.fill_between(x, treated, color="#c8562b", alpha=0.16)
ax.plot(x, treated, color="#c8562b", lw=2.2, label="treated")
ax.axvline(0.0, color="#2f6fdb", ls=":", lw=1.5)
ax.axvline(mix_mean(), color="#c8562b", ls=":", lw=1.5)
top = float(max(control.max(), treated.max()))
ax.annotate("", xy=(mix_mean(), top * 1.06), xytext=(0.0, top * 1.06),
            arrowprops=dict(arrowstyle="-|>", color="#1f2733", lw=1.6))
ax.text(mix_mean() / 2, top * 1.10, f"mean difference {mix_mean():+.2f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1f2733")
ax.text(-4.4, top * 1.35,
        f"P(treated > control) = {mix_prob_superiority():.3f}\\nthe bulk sits BELOW the control",
        fontsize=11, fontweight="bold", color="#c8562b", va="top")
ax.set_ylim(0, top * 1.55)
ax.set_yticks([])
ax.set_xlabel("outcome")
ax.set_title("Both tests are right. They are answering different questions.",
             loc="left", fontweight="bold")
ax.legend(loc="center right", frameon=False)
ax.grid(True, color="#dfe4ea", lw=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
''')

md("""
### A note on replicate counts, before the verdict columns start

From here on the tables carry **verdicts** - `ok`, `INFLATED`, `CONSERVATIVE` - and a verdict is
a 99% Wilson interval at that cell's replicate count, not a point estimate against a fixed band.
That rule is deliberate (an earlier build in this series flagged its own exactly-calibrated cell
by comparing a point estimate to a band), and it has a consequence worth stating out loud:

**a cell measured at fewer replicates is flagged less often.** This notebook runs 6,000 to 12,000
replicates a cell so it finishes in half a minute; `evidence.py` runs 40,000 to 100,000. The
*rates* below will match the evidence file to Monte Carlo noise, but the *counts of flagged
cells* will be smaller here, because a wider interval is a weaker claim rather than a different
measurement. Where a count matters, the study's number is the one in `evidence.txt`.

That is not a discrepancy between the two artifacts. It is the resolution of the instrument,
reported rather than hidden.
""")

md("""
## 3. Where the swap is fine

Under a **pure location shift** the two hypotheses coincide, so comparing power is a fair
question rather than a category error.

One gate before reading any of it: power is only comparable in a cell where **both** tests
control their Type I error. An inflated test "detects more" because it rejects more of
everything, and a power table that ignores this is just the inflation restated. The null is
measured on the same design at `d = 0`.
""")

code('''
NB_NULL_REPS = 12_000
NB_POWER_REPS = 6_000
loc_rows = []
for i, (shape, n) in enumerate(LOCATION_DESIGNS):
    null = run_location(shape, 0.0, n, NB_NULL_REPS, NULL_SEED_BASE + i)
    pw = run_location(shape, LOCATION_D, n, NB_POWER_REPS, LOCATION_SEED_BASE + i)
    loc_rows.append({"shape": shape, "n": n, "t_null": null["t"], "mw_null": null["mw"],
                     "t_power": pw["t"], "mw_power": pw["mw"],
                     "comparable": comparable(null["t"], null["mw"], NB_NULL_REPS)})

for shape in SHAPES:
    rs = [r for r in loc_rows if r["shape"] == shape]
    print(f"\\n{shape}   (predicted ARE {ares[shape]:.4f})")
    print(f"  {'n':>5} {'t null':>8} {'MW null':>8} {'t power':>9} {'MW power':>9} {'comparable':>11}")
    for r in rs:
        print(f"  {r['n']:>5} {r['t_null']:>8.4f} {r['mw_null']:>8.4f} "
              f"{r['t_power']:>9.4f} {r['mw_power']:>9.4f} "
              f"{('yes' if r['comparable'] else 'NO'):>11}")
''')

md("""
Turn those curves into one number per shape: how many observations the t-test needs for every
one Mann-Whitney needs, at matched power. Above 1 means the rank test wins.

The measured column sits **below** the predicted one almost everywhere, and the gap grows with
the prediction. That is expected rather than a discrepancy - ARE is a limit as the effect goes
to zero and n goes to infinity, while these are finite-n readings where the high-efficiency
shapes have already pushed Mann-Whitney into the flat top of the power curve. The two columns
are checked for agreement in **order**, not in value.
""")

code('''
print(f"{'shape':<14} {'predicted ARE':>14} {'measured n_t/n_MW':>19} {'cells usable':>13}")
pred, meas = [], []
for shape in SHAPES:
    rs = [r for r in loc_rows if r["shape"] == shape and r["comparable"]]
    ref = next((r for r in rs if r["n"] == LOCATION_REF_N), None)
    eff = None
    if ref is not None and len(rs) >= 2:
        eff = efficiency_from_curve([r["n"] for r in rs], [r["t_power"] for r in rs],
                                    LOCATION_REF_N, ref["mw_power"])
    if eff is not None:
        pred.append(ares[shape])
        meas.append(eff)
    print(f"{shape:<14} {ares[shape]:>14.4f} "
          f"{(f'{eff:.4f}' if eff is not None else '-'):>19} {len(rs):>13}")

rho = float(stats.spearmanr(pred, meas).statistic)
print(f"\\nSpearman(predicted, measured) across {len(pred)} shapes = {rho:.4f}")
print("\\nSo as a POWER decision the folk advice is sound: the rank test gives up a few per cent")
print("on normal data and wins outright on every other shape here. The cost of the swap is not")
print("power. It is that the question changed.")
''')

md("""
## 4. The cost nobody mentions

Two symmetric populations, same centre, different spread.

For symmetric distributions that makes `P(B > A)` **exactly 0.5**, so Mann-Whitney's own null
hypothesis is true. The mean null is true too. Any departure from 5% below is therefore not a
violated hypothesis - it is the test's variance formula, which assumes the two distributions are
*identical* rather than merely balanced.

Welch's t-test is the comparator.
""")

code('''
NB_UNEQ_REPS = 12_000
uneq = []
for i, (shape, ratio, n1, n2) in enumerate(UNEQUAL_DESIGNS):
    r = run_unequal_spread(shape, ratio, n1, n2, NB_UNEQ_REPS, UNEQUAL_SEED_BASE + i)
    uneq.append({"shape": shape, "ratio": ratio, "n1": n1, "n2": n2,
                 "welch": r["welch"], "mw": r["mw"],
                 "welch_v": verdict(r["welch"], NB_UNEQ_REPS),
                 "mw_v": verdict(r["mw"], NB_UNEQ_REPS)})

print(f"{'shape':<14} {'SD ratio':>9} {'n1':>4} {'n2':>4} {'Welch':>8} {'Welch?':>13} "
      f"{'MannWhitney':>12} {'MW?':>13}")
for r in uneq:
    print(f"{r['shape']:<14} {r['ratio']:>9.1f} {r['n1']:>4} {r['n2']:>4} {r['welch']:>8.4f} "
          f"{r['welch_v']:>13} {r['mw']:>12.4f} {r['mw_v']:>13}")
''')

md("""
Read the direction, not just the size. It is set by **which group is the wide one**.

Put the wide group in the small arm and the rank test fires several times too often. Put it in
the large arm and it all but stops firing - the power is simply gone, and a test that never
rejects does not look broken to anyone reading the output.

And unlike the classic Student's-versus-Welch problem, unequal n is **not** the precondition
here. Balanced designs break too; unequal n changes the size and the sign.
""")

code('''
bal = [r for r in uneq if r["n1"] == r["n2"]]
unb = [r for r in uneq if r["n1"] != r["n2"]]
print(f"Mann-Whitney off nominal on {sum(r['mw_v'] != 'ok' for r in bal)} of {len(bal)} BALANCED "
      f"cells and {sum(r['mw_v'] != 'ok' for r in unb)} of {len(unb)} unbalanced ones.")
print(f"Welch off nominal on {sum(r['welch_v'] != 'ok' for r in uneq)} of {len(uneq)} "
      f"(evidence.txt, at {NULL_REPS:,} replicates rather than {NB_UNEQ_REPS:,}, resolves 7).")

worst_w = max(uneq, key=lambda r: abs(r["welch"] - ALPHA))
worst_m = max(uneq, key=lambda r: max(r["mw"], ALPHA) / max(min(r["mw"], ALPHA), 1e-9))
print(f"\\nWelch's worst departure        {worst_w['welch']:.4f}")
print(f"Mann-Whitney's worst departure {worst_m['mw']:.4f}")
print("\\nWelch is not perfect here either - it is mildly conservative on the contaminated")
print("population. But a test that is a fifth conservative and a test that is an order of")
print("magnitude too quiet are not comparably wrong.")
''')

md("""
## 5. Ties

A rank test on a 5-point rating scale is an extremely common thing to do, and it is where the
tie correction stops being a footnote.

Ties can only ever **shrink** `Var(U)`. The no-ties formula therefore divides by a standard error
that is too large, and the test goes quiet. SciPy applies the correction. A lot of hand-rolled
rank tests and spreadsheet macros do not.
""")

code('''
NB_TIE_REPS = 8_000
print(f"{'levels':>7} {'SD corr/uncorr':>15} {'null corr':>10} {'null uncorr':>12} "
      f"{'power corr':>11} {'power uncorr':>13}")
tie_rows = []
for i, levels in enumerate(TIE_LEVELS):
    null = run_ties(levels, TIE_N, 0.0, NB_TIE_REPS, TIE_NULL_SEED_BASE + i)
    pw = run_ties(levels, TIE_N, TIE_D, NB_TIE_REPS, TIE_POWER_SEED_BASE + i)
    tie_rows.append((levels, null, pw))
    print(f"{levels:>7} {null['sd_ratio']:>15.4f} {null['corrected']:>10.4f} "
          f"{null['uncorrected']:>12.4f} {pw['corrected']:>11.4f} {pw['uncorrected']:>13.4f}")

lv, null, pw = tie_rows[0]
print(f"\\nOn a {lv}-point scale the correction shrinks the standard error to "
      f"{null['sd_ratio']:.4f} of the")
print(f"no-ties value. Dropping it takes a 5% test down to {null['uncorrected']:.4f} and costs")
print(f"{pw['corrected'] - pw['uncorrected']:.4f} of power. It is not a rounding detail.")
''')

md("""
## Summary

| | |
|---|---|
| **Mann-Whitney is not a robust t-test** | it is a test of `P(B > A)`, not of `mu_B - mu_A` |
| **On the disagreement design** | true mean difference `+0.80`, true `P(B>A)` `0.355` - both closed form |
| **The contradiction** | both tests significant, opposite directions, and the rate **rises** with n |
| **Power** | under a pure location shift the rank test wins on five of six shapes |
| **Unequal spread** | its own null is exactly true and it still misses 5%, in both directions, on balanced designs too |
| **Ties** | the correction is worth real power on a coarse scale |

The honest report is not one test or the other. It is **the mean difference and `P(B > A)`,
side by side** - because when those two disagree, that disagreement *is* the finding, and
picking a single test is how it gets hidden.
""")

md("""
## Try your own

The disagreement is a property of the design, and the design is three numbers. Move them and
watch the window where the two tests part company open and close.
""")

code('''
# Change these and re-run. The mean is sum(w * mu); P(B>A) is sum(w * Phi(mu / (sqrt(2) * sd))).
# Anywhere those two land on opposite sides of their nulls, the two tests will contradict.
#
# import nonparam
# nonparam.MIX_WEIGHTS = (0.8, 0.2)
# nonparam.MIX_MEANS = (-0.8, 4.0)
# print("mean", nonparam.mix_mean(), " P(B>A)", nonparam.mix_prob_superiority())
# print(nonparam.run_disagreement(200, 4000, 1)["opposite"])
#
# Or sweep the minority weight to find the crossing point:
# for w in [0.60, 0.70, 0.80, 0.857, 0.90]:
#     nonparam.MIX_WEIGHTS = (w, 1 - w)
#     print(f"w={w:.3f}  mean={nonparam.mix_mean():+.3f}  P={nonparam.mix_prob_superiority():.4f}")
print("uncomment to explore")
''')

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

The same study runs as a Streamlit app where you drag the mixture around and watch the two
verdicts separate:

```bash
pip install -r requirements.txt
streamlit run app.py
```

`evidence.py` regenerates every number above into `evidence.txt` and `results.json`, and `pytest`
asserts the findings - including that each verdict branch is *reachable*, so none of them is a
constant wearing a test.

Sibling builds in this domain: [`t-test-variants`](../t-test-variants) (which t-test),
[`normality-test-trap`](../normality-test-trap) (the ONE-sample rank swap),
[`assumption-pretest-cost`](../assumption-pretest-cost) (Student vs Welch),
[`p-value-dance`](../p-value-dance) (what p does under replay),
[`effect-size-reader`](../effect-size-reader) (which measured `P(X>Y)` as an *estimate*; this one
hands it to a *test*).
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
