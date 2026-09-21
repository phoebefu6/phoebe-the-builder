"""Generate demo.ipynb.

The engine block is EXTRACTED FROM effectsize.py BY AST AT BUILD TIME rather than hand-copied,
so the notebook holds the library's own source text and cannot go stale. That pattern started in
the sibling build `p-value-dance` (Day 177) and it carries a trap worth restating: this module
must NOT write the notebook at import time, because test_notebook.py imports it - unguarded, that
import silently replaces the executed, pre-rendered notebook with a blank one, and no later test
can see it because the test suite is what did it. The write lives behind __main__ and a regression
test asserts the import leaves the file alone.

The notebook also uses the library's own DESIGN LISTS and seeds (the seed is the index into the
list), so its printed rows ARE the cells in evidence.txt rather than a second sample of them.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/effect-size-reader"

EXPORTS = [
    "ALPHA", "Z99", "CHUNK", "BAND_LO", "BAND_HI",
    "LOGNORMAL_S", "CONTAM_SCALE", "CONTAM_SHARE", "REFERENCE_DRAWS",
    "wilson", "mc_interval", "_standard", "SHAPES", "SKEWED", "draw_pair",
    "true_prob_superiority", "analytic_prob_superiority", "METRICS", "metrics",
    "nct_power", "normal_power", "analytic_power", "power_disagreement",
    "_monotone_search", "n_for_significance", "smallest_significant_d",
    "STUDY_REPS", "STUDY_REFERENCE", "STUDY_DS", "STUDY_NS", "DICH_D", "DICH_NS",
    "TRUTH_DESIGNS", "CELL_DESIGNS", "DICH_DESIGNS", "DICH_SEED_BASE",
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


ENGINE = extract("effectsize.py", EXPORTS)

HEADER = """from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
# "It's a medium effect"

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

Cohen's d = 0.5 gets reported as a fact about how big something is. It is a *ratio* - a difference
in means over a standard deviation - and the denominator is doing far more work than anyone says
out loud.

This notebook fixes d **exactly** and changes nothing else but the shape of the distribution.

That is the whole trick, and it is worth being precise about it: every population below is
standardised to population mean 0 and population SD 1 using its **analytic** moments, never sample
ones, and then the treated group is shifted by exactly `d`. So the population Cohen's d is exactly
`d` for a normal, a skewed, a heavy-tailed and a contaminated distribution alike.

Any disagreement you see between the other effect-size metrics therefore **cannot** be a
difference in how big the effect is. It is the metrics answering different questions.

What gets measured:

1. **Calibration** - reproduce a closed form before reporting anything that has none
2. **The headline** - one d, six shapes, six different answers to "how often does treatment win?"
3. **The estimators** - what a sample says, and where Hedges' correction stops working
4. **Robustness** - one contaminated value in a hundred, and which metric breaks
5. **Significance versus magnitude** - plus a SciPy defect that put a wrong table in this notebook
6. **The median split** - and the shapes where the advice against it is simply false
""")

md("""
## Setup

The engine is the repository's own `effectsize.py`, extracted verbatim so this notebook and the
library cannot disagree. The design lists and seeds are the library's too, so the rows printed
below are the same cells that appear in `evidence.txt`.
""")

code(f'''
{HEADER}
{ENGINE}

import matplotlib
import matplotlib.pyplot as plt

print("shapes:", SHAPES)
print("true effect sizes:", STUDY_DS, " sample sizes:", STUDY_NS)
print(f"{{STUDY_REPS:,}} replicates a cell, {{STUDY_REFERENCE:,}} draws per population truth")
''')

md("""
### First: are the populations actually what they claim to be?

If a "shape" is secretly not standardised, "the same d" is not the same d and the headline below
becomes an artefact of the generator rather than a finding. So this gets checked before anything
else, on four million draws each.
""")

code('''
print(f"{'shape':<14} {'mean':>9} {'sd':>9} {'skew':>9} {'excess kurtosis':>17}")
rng = np.random.default_rng(7)
for shape in SHAPES:
    x = _standard(shape, (4_000_000,), rng)
    print(f"{shape:<14} {x.mean():>9.4f} {x.std(ddof=1):>9.4f} "
          f"{float(stats.skew(x)):>9.3f} {float(stats.kurtosis(x)):>17.3f}")
print("\\nMean 0 and SD 1 everywhere - by construction, not by luck.")
print("The skew and kurtosis columns are what actually differs between these populations.")
''')

md("""
## 1. Calibration

For **normal** data the probability of superiority has a closed form: `Phi(d / sqrt(2))`. The
reference sampler has to reproduce it, or nothing it says about the other five shapes - which have
no closed form - is worth reading.
""")

code('''
truths = {}
print(f"{'shape':<14} {'d':>5} {'P(X>Y) measured':>17} {'+/- 99%':>9} {'closed form':>13} {'match':>7}")
for i, (shape, d) in enumerate(TRUTH_DESIGNS):
    p, err = true_prob_superiority(shape, d, STUDY_REFERENCE, seed=i)
    truths[(shape, d)] = (p, err)
    if shape == "normal":
        exact = analytic_prob_superiority(d)
        ok = abs(p - exact) <= max(err, 1e-4)
        print(f"{shape:<14} {d:>5.1f} {p:>17.5f} {err:>9.5f} {exact:>13.5f} "
              f"{('OK' if ok else 'FAIL'):>7}")
''')

md("""
## 2. The headline

Every number in a column below comes from a population whose Cohen's d is **the same**.
""")

code('''
print(f"{'shape':<14}" + "".join(f"{'d=' + format(d, '.1f'):>14}" for d in STUDY_DS))
for shape in SHAPES:
    print(f"{shape:<14}" + "".join(f"{truths[(shape, d)][0]:>14.4f}" for d in STUDY_DS))
print(f"\\n{'spread':<14}" + "".join(
    f"{max(truths[(s, d)][0] for s in SHAPES) - min(truths[(s, d)][0] for s in SHAPES):>14.4f}"
    for d in STUDY_DS))

d = 0.5
vals = {s: truths[(s, d)][0] for s in SHAPES}
lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
print(f"\\nAt d = {d}: '{lo}' says the treated value wins {vals[lo]:.1%} of the time,")
print(f"           '{hi}' says {vals[hi]:.1%}. Same d.")
''')

md("""
"A medium effect, d = 0.5" is routinely glossed as "the treated case beats the control about 64%
of the time". That gloss is **correct for normal data and for nothing else here**.

And note how this differs from what the sibling build
[`normality-test-trap`](../normality-test-trap) found about the *t-test*: there, symmetric
non-normality was harmless and skew was the entire problem. Here both matter and both move the
answer the **same way**. The contaminated population is perfectly symmetric and still reads about
0.686 at d = 0.5 against normal's 0.638.

The mechanism: a heavy tail inflates the SD that d divides by, so a shift of "one d" is a larger
shift relative to the bulk of the distribution - which is where the comparisons actually happen.
Robustness of the *test* and robustness of the *effect size* are different questions.
""")

md("""
## 3. What a sample says

True d is 0.5 in every row. `bias` is the average estimate minus the truth.
""")

code('''
print(f"{'shape':<14} {'n':>5} {'d-hat':>8} {'bias':>8} {'sd(d-hat)':>10} {'g':>8} {'bias':>8} "
      f"{'P(X>Y)':>8} {'bias':>8}")
cells_out = {}
for i, (shape, n) in enumerate(CELL_DESIGNS):
    ps_true = truths[(shape, 0.5)][0]
    rng = np.random.default_rng(i)
    acc = {m: [] for m in METRICS}
    done = 0
    while done < STUDY_REPS:
        take = min(CHUNK, STUDY_REPS - done)
        a, b = draw_pair(shape, 0.5, n, n, take, rng)
        for k, v in metrics(a, b).items():
            acc[k].append(v)
        done += take
    vals = {k: np.concatenate(v) for k, v in acc.items()}
    row = {"d": float(vals["cohens_d"].mean()), "g": float(vals["hedges_g"].mean()),
           "sd": float(vals["cohens_d"].std(ddof=1)),
           "ps": float(vals["prob_superiority"].mean()), "ps_true": ps_true}
    cells_out[(shape, n)] = row
    print(f"{shape:<14} {n:>5} {row['d']:>8.4f} {row['d'] - 0.5:>+8.4f} {row['sd']:>10.4f} "
          f"{row['g']:>8.4f} {row['g'] - 0.5:>+8.4f} {row['ps']:>8.4f} "
          f"{row['ps'] - ps_true:>+8.4f}")
''')

md("""
### Hedges' correction only works where it was derived

Cohen's d is biased upward at small n, and Hedges' *g* applies a correction for it. The correction
is derived **under normality**. So the interesting question is not whether it changes the number -
it always does - but whether it removes the bias.
""")

code('''
print(f"{'shape':<14} {'d-hat bias':>12} {'g bias':>10} {'P(X>Y) bias':>13} {'corrected?':>12}")
for shape in SHAPES:
    r = cells_out[(shape, 10)]
    fixed = abs(r["g"] - 0.5) <= 0.01
    print(f"{shape:<14} {r['d'] - 0.5:>+12.4f} {r['g'] - 0.5:>+10.4f} "
          f"{r['ps'] - r['ps_true']:>+13.4f} {('yes' if fixed else 'NO'):>12}")

n20 = cells_out[("normal", 20)]
print(f"\\nAnd the scale of it: at n = 20 on normal data the bias in d-hat is "
      f"{n20['d'] - 0.5:+.4f}")
print(f"while its standard deviation across replicates is {n20['sd']:.4f} - a ratio of "
      f"{abs(n20['d'] - 0.5) / n20['sd']:.1%}.")
print("Correcting the third decimal of a number whose first decimal is not settled.")
''')

md("""
Two things fall out, and the second is the one worth carrying:

- The correction lands on **normal** data. On skewed and contaminated populations it removes only
  a fraction of a bias it was never designed for - on lognormal data at n = 10 the corrected
  estimate is still about +0.09 out. The excess comes from the sample SD, which a skewed sample
  underestimates in exactly the draws where the sample mean is also low.
- **P(X>Y) is unbiased on every shape at n = 10.** The rank summary needs no small-sample
  correction because it never divides by an estimated standard deviation.
""")

md("""
## 4. One contaminated value in a hundred

1% of the treated values are replaced with +10 SD. The outlier points the *same way* as the
effect - a big genuine-looking response, not a data-entry error. That is the charitable case.
""")

code('''
rng = np.random.default_rng(31)
a, b = draw_pair("normal", 0.5, 100, 100, STUDY_REPS, rng)
clean = metrics(a, b)
dirty_b = b.copy()
dirty_b[:, :1] = 10.0
dirty = metrics(a, dirty_b)

print(f"{'metric':<24} {'clean':>10} {'contaminated':>14} {'change':>10}")
rows = []
for m in METRICS:
    c, dd = float(clean[m].mean()), float(dirty[m].mean())
    pct = (dd - c) / abs(c) * 100.0
    rows.append((m, pct))
    print(f"{m:<24} {c:>10.4f} {dd:>14.4f} {pct:>+9.1f}%")

worst = max(rows, key=lambda r: abs(r[1]))
print(f"\\nMost moved: {worst[0]} at {worst[1]:+.1f}%.")
''')

md("""
The result nobody expects: **Glass's delta** is the metric that moves, and it is the one
recommended precisely for the case where the treatment changes the spread.

It divides by the *control* group's SD, which the contamination never touches - so the numerator
grows and the denominator does not. Cohen's d barely moves, because its pooled SD absorbs the same
outlier that inflated the mean. The safer-sounding choice is the fragile one, and it fails in the
dangerous direction: it reports a **larger** effect.
""")

md("""
## 5. Significance versus magnitude

How large does n have to be before a difference nobody would act on clears p < 0.05?
""")

code('''
print(f"{'d':>7} {'n per group for 80% power':>28} {'P(X>Y) at that d':>18}")
for d in (0.8, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
    print(f"{d:>7.2f} {n_for_significance(d):>28,} {analytic_prob_superiority(d):>18.4f}")

print(f"\\n{'n per group':>13} {'smallest d at 50% power':>26} {'P(X>Y)':>10} {'in words':>32}")
for n in (100, 1000, 10000, 100000, 1000000):
    dd = smallest_significant_d(n)
    ps = analytic_prob_superiority(dd)
    print(f"{n:>13,} {dd:>26.5f} {ps:>10.4f} {'treated wins ' + format(ps, '.2%'):>32}")
''')

md("""
### A defect this notebook shipped once, and nothing warned about it

The first version of the table above said `d = 0.5` needs **n = 11,417** per group for 80% power.
The real answer is 64.

`scipy.stats.nct` returns **nan** - not an error, not a warning - over a band of large degrees of
freedom. A nan is silently `False` in every comparison, so `power >= target` reads as "not
powerful enough", and a monotone binary search walks straight past the answer and returns a
larger, entirely plausible number. Nothing looks wrong. The table just is.
""")

code('''
ns = list(range(5, 400)) + [500, 1000, 2000, 5000, 10000, 20000, 100000]
print(f"{'n':>8} {'exact (nct)':>14} {'normal approx':>15}")
for n in (10, 100, 500, 1000, 5000, 20000):
    print(f"{n:>8} {nct_power(0.5, n):>14.6f} {normal_power(0.5, n):>15.6f}")

g = power_disagreement(0.5, ns)
print(f"\\nnan at {g['n_nan']} of the {len(ns)} sample sizes probed, "
      f"from n = {g['first_nan_n']:,} to n = {g['last_nan_n']:,}")
print(f"worst gap between the two formulas where BOTH are finite: {g['worst_gap']:.6f} "
      f"(n = {g['worst_gap_n']})")
print(f"worst gap once df >= 400: {g['worst_gap_large_df']:.2e} (n = {g['worst_gap_large_df_n']})")
print("\\nThe fallback is trusted because that gap is MEASURED, not assumed - and the nan band")
print("starts far above df = 400, where the approximation is already good to five decimals.")

try:
    _monotone_search(lambda n: False, 4, 1000)
except ValueError as e:
    print(f"\\nAnd the search now refuses instead of guessing: {e}")
''')

md("""
## 6. What a median split costs

Cut the same data at the pooled median and run a two-proportion test on it.

A **null cell** is run first, at d = 0 on the same shape and n, and the power comparison is only
read where *both* tests hold a nominal 5%. A test that over-rejects detects more of everything, so
its power advantage would just be the inflation restated.
""")

code('''
print(f"{'shape':<14} {'n':>5} {'power t':>9} {'power split':>12} {'null t':>8} "
      f"{'null split':>11} {'read?':>7} {'winner':>8}")
for i, (shape, n) in enumerate(DICH_DESIGNS):
    rng = np.random.default_rng(DICH_SEED_BASE + i)

    def run(dd, r):
        st, sc = [], []
        done = 0
        while done < STUDY_REPS:
            take = min(CHUNK, STUDY_REPS - done)
            a, b = draw_pair(shape, dd, n, n, take, r)
            st.append(np.asarray(stats.ttest_ind(b, a, axis=-1, equal_var=True).pvalue) < ALPHA)
            both = np.concatenate([a, b], axis=-1)
            cut = np.median(both, axis=-1, keepdims=True)
            ah, bh = (a > cut).sum(axis=-1), (b > cut).sum(axis=-1)
            pool = (bh + ah) / (2 * n)
            se = np.sqrt(np.maximum(pool * (1 - pool) * 2.0 / n, 1e-300))
            sc.append(np.abs((bh / n - ah / n) / se) > stats.norm.ppf(1 - ALPHA / 2))
            done += take
        return float(np.concatenate(st).mean()), float(np.concatenate(sc).mean())

    pt, pc = run(DICH_D, rng)
    nt, nc = run(0.0, np.random.default_rng(DICH_SEED_BASE + i + 50_000))
    ok = all(not (hi < BAND_LO or lo > BAND_HI) for lo, hi in
             (wilson(nt, STUDY_REPS), wilson(nc, STUDY_REPS)))
    t_lo, t_hi = wilson(pt, STUDY_REPS)
    c_lo, c_hi = wilson(pc, STUDY_REPS)
    winner = "-" if not ok else ("t" if t_lo > c_hi else ("split" if c_lo > t_hi else "-"))
    print(f"{shape:<14} {n:>5} {pt:>9.4f} {pc:>12.4f} {nt:>8.4f} {nc:>11.4f} "
          f"{('yes' if ok else 'NO'):>7} {winner:>8}")
''')

md("""
Two findings, and the guard produced both:

- **Most rows cannot be read at all.** The median-split test is anticonservative at n = 50 -
  it rejects a true null at roughly 0.065 to 0.072 against a nominal 0.05, on *every* shape.
  Before dichotomising costs you any power, it costs you the error rate you thought you had.
- Among the readable rows, **the winner depends on the shape**. On normal data the t-test wins and
  the split throws away about a third of the sample. On lognormal and contaminated data the split
  *wins* - it is a rank method and the t-test is not.

"Never dichotomise your data" is real advice about **normal** data. Measured on skewed data it is
false. It was never a rule about dichotomisation; it is a rule about matching the yardstick to the
shape of what you measured - which is what section 2 said in a different vocabulary.
""")

md("""
## 7. The picture
""")

code('''
INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
SC = {"normal": "#41506b", "uniform": "#2f6fdb", "t5": "#1a8f7a",
      "contaminated": "#b8860b", "lognormal": "#c8562b", "exponential": "#9a3f8f"}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# Dots on a shared baseline, not bars: the quantity of interest is the SPREAD within each d.
ax = axes[0]
x = np.arange(len(STUDY_DS))
for i, shape in enumerate(SHAPES):
    off = (i - (len(SHAPES) - 1) / 2) * 0.12
    ax.plot(x + off, [truths[(shape, d)][0] for d in STUDY_DS], "o", color=SC[shape], ms=10,
            mec=BG, mew=1.2, label=shape, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f"d = {d:g}" for d in STUDY_DS])
ax.set_ylabel("P(treated beats control)")
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
ax.set_title("Same d, six answers", color=INK, fontsize=12, weight="bold", loc="left")

ax = axes[1]
y = np.arange(len(SHAPES))
for i, shape in enumerate(SHAPES):
    r = cells_out[(shape, 10)]
    ax.barh(i + 0.16, r["d"] - 0.5, height=0.3, color=SC[shape])
    ax.barh(i - 0.16, r["g"] - 0.5, height=0.3, color=SC[shape], alpha=0.42)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(y); ax.set_yticklabels(SHAPES, fontsize=9)
ax.set_xlabel("bias at n = 10 (solid: Cohen's d, pale: Hedges' g)")
ax.set_title("The correction assumes normality", color=INK, fontsize=12, weight="bold",
             loc="left")

ax = axes[2]
rs = sorted(rows, key=lambda r: r[1])
ax.barh(np.arange(len(rs)), [r[1] for r in rs],
        color=[SC["lognormal"] if abs(r[1]) > 10 else MUTED for r in rs], height=0.6)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(np.arange(len(rs)))
ax.set_yticklabels([r[0].replace("_", " ") for r in rs], fontsize=9)
ax.set_xlabel("% change from one outlier in a hundred")
ax.set_title("Which metric breaks", color=INK, fontsize=12, weight="bold", loc="left")

fig.tight_layout()
fig.savefig("notebook_figure.png", dpi=140, facecolor=BG)
plt.show()
''')

md("""
## What to take away

1. At an **identical** true Cohen's d of 0.5, the probability of superiority ranges over about
   0.083 across six distribution shapes. d is not wrong - it is a ratio, and its denominator means
   different things in different shapes.
2. Reporting an effect size without saying *which one*, on *what shape*, with *what interval*, is
   reporting a number rather than a magnitude.
3. Glass's delta - recommended exactly when the treatment changes the spread - moves about +19% on
   one contaminated value in a hundred. Cohen's d moves about one percent, in the other direction.
4. At n = 100,000 the smallest effect a study can see is a 50.25% win rate. Significance stopped
   carrying information about magnitude long before that.
5. "Never dichotomise" is a claim about normal data, and measured on skewed data it is false.

The through-line: every one of these is the same mistake in a different costume - treating a
**summary** as if it were the **thing**. d, p, and "significant" are all summaries, and each one
throws away something different on the way out.
""")

md("""
## Try your own
""")

code('''
# --- Your comparison -------------------------------------------------------
# MY_SHAPE = "lognormal"     # normal / uniform / t5 / contaminated / lognormal / exponential
# MY_D     = 0.5             # true Cohen's d, held EXACTLY by construction
# MY_N     = 40              # per group
#
# ps_true, err = true_prob_superiority(MY_SHAPE, MY_D, 4_000_000, seed=0)
# rng = np.random.default_rng(0)
# a, b = draw_pair(MY_SHAPE, MY_D, MY_N, MY_N, 20000, rng)
# m = metrics(a, b)
#
# print(f"true Cohen's d          {MY_D}")
# print(f"true P(X>Y)             {ps_true:.4f}   (normal would say "
#       f"{analytic_prob_superiority(MY_D):.4f})")
# for k in METRICS:
#     print(f"{k:<24} {float(m[k].mean()):>8.4f}")
# print(f"n for 80% power         {n_for_significance(MY_D):,} per group")

# --- Or: how much does the shape alone move the answer? --------------------
# for s in SHAPES:
#     p, _ = true_prob_superiority(s, 0.5, 4_000_000, seed=0)
#     print(f"{s:<14} P(X>Y) = {p:.4f}")
''')

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

The same study runs as a Streamlit app where you pick the shape and drag d around:

```bash
pip install -r requirements.txt
streamlit run app.py
```

`evidence.py` regenerates every number above into `evidence.txt` and `results.json`, and `pytest`
asserts the findings - including that each predicate is *capable* of returning either answer, so
none of them is a constant wearing a test.

Sibling builds in this domain, all of which interrogate the **test**:
[`t-test-variants`](../t-test-variants) (which one),
[`normality-test-trap`](../normality-test-trap) and
[`assumption-pretest-cost`](../assumption-pretest-cost) (what the pretests cost),
[`p-value-dance`](../p-value-dance) (what the p-value does under replay). This one is the first
to leave the test alone and interrogate the **estimate**.
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
