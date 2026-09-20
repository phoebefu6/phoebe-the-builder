"""Generate demo.ipynb.

The cell-emitting plumbing is copied from the assumption-pretest-cost build (which took it from
normality-test-trap, from t-test-variants, from prediction-interval). It gets two things right
that fail silently until nbconvert runs: nbformat wants each source line to KEEP its trailing
newline, and every cell needs an id.

One change from the sibling, and it is an improvement worth carrying forward. Those builds
hand-copied the engine into the notebook and then wrote a test to diff the copy against the
library. That works, but it detects drift instead of preventing it. Here the engine block is
EXTRACTED FROM pvalue.py BY AST AT BUILD TIME, so the notebook cannot contain a stale copy - it
contains the library's own source text. test_notebook.py still asserts it, because a guarantee
nobody checks is a guarantee that quietly stops holding.

The notebook also uses the library's own SEEDS (the index into DESIGNS), so its rows ARE the cells
in evidence.txt rather than a second sample of them. Standard since Day 175, where a borderline
cell landed either side of a threshold on different seeds and read as two artifacts contradicting
each other.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/p-value-dance"

# The engine, lifted from the library rather than retyped.
EXPORTS = [
    "ALPHA", "Z99", "CHUNK", "wilson", "analytic_power", "required_n",
    "_one_chunk", "simulate", "NS", "DS", "DESIGNS", "REPS",
]


def extract(path: str, names: List[str]) -> str:
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    out: List[str] = []
    for node in tree.body:
        got = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            got = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            got = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            got = node.target.id
        if got in names:
            start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
            out.append("".join(lines[start:node.end_lineno]))
    return "\n".join(out).rstrip()


ENGINE = extract("pvalue.py", EXPORTS)

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
# "p = 0.049"

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A p-value is a random variable. Everybody who has taken a stats class knows that sentence. Almost
nobody has watched what it means, because doing so requires running the *same study* a few tens of
thousands of times, which nobody gets to do.

This notebook does it. One true effect, one sample size, forty thousand replays, nothing changing
between them except the sample.

**The test is not the story here.** Every number below comes from Student's two-sample t on
equal-n, equal-variance, normal data. The sibling build [`t-test-variants`](../t-test-variants)
verified that this is the one configuration where Student's t is *exact* - its Type I error sits
at 0.0500 and its power matches the noncentral-t formula. So nothing you see can be blamed on a
violated assumption, a wrong test choice, or a shaky approximation. The test is right. The dance
is the p-value's own.

What gets measured:

1. **Calibration** - can the harness reproduce a known truth before it reports an unknown one
2. **The dance** - how far p moves when literally nothing changes
3. **The reversal** - the spread is *widest* where it matters *least*, which kills the usual telling
4. **The winner's curse** - the part that is not the power calculation restated
5. **Replication** - what an identical rerun of "p = 0.05" actually says
6. **What a significant result is worth** - once most hypotheses are wrong
""")

md("""
## Setup

The engine is the repository's own `pvalue.py`, extracted verbatim so this notebook and the
library cannot disagree. The seeds are the library's seeds, so the rows below are the same cells
that appear in `evidence.txt` - not a second sample of them.
""")

# The extraction pulls named objects only, so the imports the engine needs are stated here. The
# __future__ line has to be first in the cell for Python 3.9 compatibility.
HEADER = """from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
"""

code(f'''
{HEADER}
{ENGINE}

import matplotlib
import matplotlib.pyplot as plt

print(f"{{len(DESIGNS)}} designs, {{REPS:,}} replays each, alpha = {{ALPHA}}")
print("effect sizes:", DS)
print("sample sizes:", NS)
''')

md("""
## 1. Calibration first

A harness that cannot reproduce a known truth cannot be trusted on an unknown one. Two known
truths are available here and both are exact, not approximate:

- Under a true null, a p-value is **Uniform(0, 1)** by construction. Its 5th percentile is 0.05,
  its median is 0.50.
- Under a true effect, the power of Student's t is given exactly by the **noncentral t**
  distribution.

If either check fails, nothing further in the notebook is worth reading.
""")

code('''
from scipy import stats as _st

print(f"{'design':<14} {'measured':>10} {'analytic':>10} {'99% CI':>22} {'verdict':>9}")
grid = {}
for i, (d, n) in enumerate(DESIGNS):
    p, d_hat = simulate(d, n, REPS, seed=i)
    grid[(d, n)] = (p, d_hat)
    power = float((p < ALPHA).mean())
    exact = analytic_power(d, n) if d > 0 else ALPHA
    lo, hi = wilson(power, REPS)
    ok = lo <= exact <= hi
    print(f"{f'd={d:g}, n={n}':<14} {power:>10.4f} {exact:>10.4f} "
          f"{'[' + format(lo, '.4f') + ', ' + format(hi, '.4f') + ']':>22} "
          f"{('OK' if ok else 'FAIL'):>9}")

nulls = [(d, n) for (d, n) in DESIGNS if d == 0]
ks = [_st.kstest(grid[k][0], "uniform").pvalue for k in nulls]
print(f"\\nNull designs passing the uniformity test (KS p > 0.01): "
      f"{sum(1 for v in ks if v > 0.01)} of {len(nulls)}")
print(f"5th percentile of p under the null (should be 0.05): "
      f"{float(np.percentile(grid[(0.0, 50)][0], 5)):.4f}")
''')

md("""
## 2. The dance

Now replay one study. Nothing changes between replicates except which sample you happened to draw.

Read the `p05` and `p95` columns as a range: nine times out of ten, a rerun of that exact study
lands somewhere between them.
""")

code('''
print(f"{'design':<14} {'power':>7} {'p05':>11} {'median':>11} {'p95':>11} {'orders':>8}")
spread = {}
for (d, n) in DESIGNS:
    p = grid[(d, n)][0]
    q = {k: float(np.percentile(p, k)) for k in (5, 50, 95)}
    orders = math.log10(q[95]) - math.log10(max(q[5], 1e-300))
    spread[(d, n)] = (orders, q, float((p < ALPHA).mean()))
    print(f"{f'd={d:g}, n={n}':<14} {float((p < ALPHA).mean()):>7.4f} {q[5]:>11.2e} "
          f"{q[50]:>11.2e} {q[95]:>11.2e} {orders:>8.2f}")
''')

md("""
### The reversal

Look at the `orders` column against the `power` column. The folk version of this story says the
p-value is unreliable because it bounces around; the implied fix is to distrust p-values.

The measurement says something else. The spread is **narrowest under the null** (about 1.3 orders
of magnitude - which is just what Uniform(0,1) looks like on a log scale) and **widest at the
highest power in the grid**. The bouncing is not the hazard. It is not even pointed the right way
to be the hazard.
""")

code('''
best = max(spread, key=lambda k: spread[k][0])
worst = min(spread, key=lambda k: spread[k][0])
print(f"widest dance:    d={best[0]:g}, n={best[1]:<4} {spread[best][0]:>6.2f} orders "
      f"at power {spread[best][2]:.4f}")
print(f"narrowest dance: d={worst[0]:g}, n={worst[1]:<4} {spread[worst][0]:>6.2f} orders "
      f"at power {spread[worst][2]:.4f}")

powers = [spread[k][2] for k in DESIGNS]
orders = [spread[k][0] for k in DESIGNS]
rank = lambda v: np.argsort(np.argsort(v))
print(f"\\nrank correlation between power and spread across all {len(DESIGNS)} designs: "
      f"{np.corrcoef(rank(powers), rank(orders))[0, 1]:.3f}")
print("Positive. More power, MORE dance - measured in orders of magnitude.")
''')

md("""
### So what is the decision-relevant version?

Not the width. Whether the central 90% **straddles 0.05** - whether an identical rerun routinely
returns the opposite verdict.

And here is the sharpest negative result in this build. That predicate is an *identity*:
`p95 < alpha` if and only if more than 95% of replicates fell below alpha, which is to say
`power > 0.95`. Straddling 0.05 **is** having power below 0.95, exactly, with nothing left over.

The decision-relevant dance is not an extra hazard sitting on top of low power. It is low power,
restated in a more alarming vocabulary. Anything this notebook has to add beyond the power
calculation is in sections 3, 4 and 5 - not here.
""")

code('''
print(f"{'design':<14} {'straddles .05':>14} {'power < 0.95':>13} {'identical?':>11}")
for (d, n) in DESIGNS:
    if d == 0:
        continue   # degenerate: under the null p05 IS alpha, so the predicate flips on noise
    orders_, q, power = spread[(d, n)]
    straddles = q[5] < ALPHA < q[95]
    print(f"{f'd={d:g}, n={n}':<14} {str(straddles):>14} {str(power < 0.95):>13} "
          f"{str(straddles == (power < 0.95)):>11}")

print(f"\\nn per group needed for 80% power:  "
      + "   ".join(f"d={d}: {required_n(d)}" for d in (0.2, 0.5, 0.8)))
''')

md("""
## 3. The winner's curse - the part that is NOT power restated

Power tells you how often you will detect the effect. It says nothing about what the detected
effects *look like*.

Two quantities, both of them invisible in the p-value:

- **Type M (magnitude)**: the mean absolute observed effect among *significant* replicates,
  divided by the truth. How inflated a published estimate is, given that it got published.
- **Type S (sign)**: the share of *significant* replicates pointing the wrong way.

A study can be honestly run, correctly analysed, and statistically significant, and still be a
number in the wrong direction.
""")

code('''
print(f"{'design':<14} {'power':>7} {'type M':>8} {'type S':>9} {'type S 99% CI':>22} "
      f"{'median published d':>19}")
curse = {}
for (d, n) in DESIGNS:
    if d == 0:
        continue
    p, d_hat = grid[(d, n)]
    sig = p < ALPHA
    winners = d_hat[sig]
    type_m = float(np.mean(np.abs(winners)) / d)
    type_s = float((winners < 0).sum() / sig.sum())
    lo, hi = wilson(type_s, int(sig.sum()))
    curse[(d, n)] = (float(sig.mean()), type_m, type_s)
    print(f"{f'd={d:g}, n={n}':<14} {sig.mean():>7.4f} {type_m:>8.2f} {type_s:>9.4f} "
          f"{'[' + format(lo, '.4f') + ', ' + format(hi, '.4f') + ']':>22} "
          f"{float(np.median(winners)):>19.3f}")

k = max(curse, key=lambda k: curse[k][1])
print(f"\\nWorst: d={k[0]:g}, n={k[1]} - power {curse[k][0]:.4f}, but its significant results are")
print(f"{curse[k][1]:.2f}x the true effect and {curse[k][2]:.1%} of them have the WRONG SIGN.")
''')

md("""
## 4. "Would it replicate?"

You ran a study once and got p = 0.05. Everyone's instinct is that this number is now *evidence
about the rerun* - that a 0.05 is a shaky result which might come back as anything.

Test it directly. Draw studies until one lands in p in [0.045, 0.055]. Then rerun *that same
study*, from the same true effect, and record the new p.
""")

code('''
print(f"{'design':<14} {'hits':>7} {'rep p05':>10} {'rep med':>9} {'rep p95':>9} "
      f"{'sig again':>10} {'power':>8} {'differs?':>9}")
for i, (d, n) in enumerate([(0.2, 50), (0.5, 20), (0.5, 50), (0.8, 20)]):
    rng = np.random.default_rng(900 + i)
    keep = []
    done = 0
    while done < 200000:
        take = min(CHUNK, 200000 - done)
        p, _ = _one_chunk(d, n, take, rng)
        hit = int(((p >= 0.045) & (p <= 0.055)).sum())
        if hit:
            keep.append(_one_chunk(d, n, hit, rng)[0])
        done += take
    rep = np.concatenate(keep)
    rate = float((rep < ALPHA).mean())
    lo, hi = wilson(rate, rep.size)
    exact = analytic_power(d, n)
    print(f"{f'd={d:g}, n={n}':<14} {rep.size:>7} {np.percentile(rep, 5):>10.2e} "
          f"{np.percentile(rep, 50):>9.4f} {np.percentile(rep, 95):>9.3f} {rate:>10.4f} "
          f"{exact:>8.4f} {str(not (lo <= exact <= hi)):>9}")
''')

md("""
The replication significance rate is the study's **plain unconditional power**, every time. Not
close to it - inside a 99% interval of it, on every design.

Which makes sense the moment you say it out loud: with the true effect held fixed, replicate
p-values are independent. Observing 0.05 told you nothing about the rerun that you did not already
know before you looked.

So the famous "a replication of p = 0.05 could come back anywhere from 0.0001 to 0.44" is real,
but it is not a fact about the 0.05. It is the ordinary sampling distribution of p at that power,
and it was there before the first study ran. What people *mean* when they find this alarming is
that they do not know the true effect, and one p = 0.05 is weak evidence about it - which is the
next section, not this one.
""")

md("""
## 5. What a significant result is actually worth

Alpha is the false-positive rate **among true nulls**. It is not the error rate of the claims you
publish, and the two are not close.

Run a mixture: a share `prior` of studies have a real effect, the rest are exactly null. Among the
ones that come out significant, how many were null all along?

The closed form - `alpha(1-prior) / (alpha(1-prior) + prior x power)` - is used here as a **check
on the simulation**, not as a substitute for it.
""")

code('''
print(f"{'prior':>7} {'d':>5} {'n':>5} {'power':>7} {'n sig':>8} {'false share':>12} "
      f"{'99% CI':>20} {'formula':>9} {'match':>6}")
for i, (prior, d, n) in enumerate([(0.5, 0.5, 50), (0.2, 0.5, 50), (0.1, 0.5, 50),
                                   (0.1, 0.2, 20), (0.5, 0.5, 20)]):
    rng = np.random.default_rng(700 + i)
    real_l, sig_l, done = [], [], 0
    while done < 200000:
        take = min(CHUNK, 200000 - done)
        real = rng.random(take) < prior
        a = rng.standard_normal((take, n))
        b = rng.standard_normal((take, n)) + np.where(real, d, 0.0)[:, None]
        _, p = _st.ttest_ind(b, a, axis=1, equal_var=True)
        real_l.append(real)
        sig_l.append(np.asarray(p, dtype=float) < ALPHA)
        done += take
    real, sig = np.concatenate(real_l), np.concatenate(sig_l)
    n_sig = int(sig.sum())
    share = float((sig & ~real).sum() / n_sig)
    power = analytic_power(d, n)
    formula = ALPHA * (1 - prior) / (ALPHA * (1 - prior) + prior * power)
    lo, hi = wilson(share, n_sig)
    print(f"{prior:>7.2f} {d:>5.1f} {n:>5} {power:>7.4f} {n_sig:>8} {share:>12.4f} "
          f"{'[' + format(lo, '.4f') + ', ' + format(hi, '.4f') + ']':>20} "
          f"{formula:>9.4f} {str(lo <= formula <= hi):>6}")
''')

md("""
## 6. The picture

Four panels. The true effect size is the only thing colour encodes; power is always position,
never colour.
""")

code('''
INK, MUTED, GRID, BG = "#1f2733", "#6b7684", "#dfe4ea", "#ffffff"
C = {0.0: "#9aa4b0", 0.2: "#c8562b", 0.5: "#b8860b", 0.8: "#2f6fdb"}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor=BG)
for ax in axes:
    ax.set_facecolor(BG); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)

# Range bars, not histograms: the question is whether the central 90% straddles 0.05, which a
# range against a reference line answers and a histogram does not.
ax = axes[0]
rows = [k for k in DESIGNS if k[1] <= 100 and (k[0] > 0 or k[1] == 50)]
for i, (d, n) in enumerate(rows):
    _, q, _ = spread[(d, n)]
    ax.plot([q[5], q[95]], [i, i], color=C[d], lw=6, solid_capstyle="round", alpha=0.85)
    ax.plot([q[50]], [i], "o", color=BG, ms=6.5, zorder=3)
    ax.plot([q[50]], [i], "o", color=C[d], ms=4.5, zorder=4)
ax.axvline(ALPHA, color=INK, ls="--", lw=1.4)
ax.set_xscale("log"); ax.set_xlim(1e-13, 3)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"d={d:g}, n={n}" for d, n in rows], fontsize=8)
ax.set_xlabel("p-value (log) - 5th to 95th percentile of 40,000 replays")
ax.set_title("Replay the same study", color=INK, fontsize=12, weight="bold", loc="left")

ax = axes[1]
for d in DS:
    pts = sorted([(spread[k][2], spread[k][0]) for k in DESIGNS if k[0] == d])
    ax.plot([p for p, _ in pts], [s for _, s in pts], "o-", color=C[d], lw=1.6, ms=6,
            label=f"d = {d:g}" if d else "null")
ax.axvline(0.95, color=INK, ls=":", lw=1.2)
ax.set_xlabel("measured power"); ax.set_ylabel("spread of p, orders of magnitude")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Widest where it matters least", color=INK, fontsize=12, weight="bold", loc="left")

ax = axes[2]
pts = sorted(curse.items(), key=lambda kv: kv[1][0])
ax.plot([v[0] for _, v in pts], [v[1] for _, v in pts], "-", color=MUTED, lw=1.2, zorder=1)
for k, v in pts:
    ax.plot([v[0]], [v[1]], "o", color=C[k[0]], ms=7, zorder=3)
ax.axhline(1.0, color=INK, ls="--", lw=1.2)
ax.set_xlabel("measured power"); ax.set_ylabel("published effect / true effect")
ax.set_title("The winner's curse", color=INK, fontsize=12, weight="bold", loc="left")

fig.tight_layout()
fig.savefig("notebook_figure.png", dpi=140, facecolor=BG)
plt.show()
''')

md("""
## What to take away

1. The p-value really is a random variable and its spread really is enormous - but the spread is
   **largest at high power**, where it changes no decision. Raw spread is not the finding.
2. The decision-relevant version of the dance is **exactly** "power < 0.95". It carries no
   information beyond the power calculation you should already have run.
3. What power does *not* tell you is the **winner's curse**: at the lowest power measured here the
   significant results are inflated about 5.9x and roughly 13% of them point the wrong way.
4. **Conditioning on the observed p = 0.05 buys nothing.** The rerun's chance of significance is
   the plain unconditional power, on every design tested.
5. With a realistic prior, up to **83%** of significant findings here were null all along - and
   alpha never sees it.

The practical version: a p-value is not a bad instrument. It is a *narrow* one. It answers "how
surprising is this data if nothing is going on", and people routinely read it as an answer to
"how big is the effect", "will this replicate", and "how likely is my hypothesis" - three
questions it does not contain and that sections 3, 4 and 5 have to be run separately to answer.
""")

md("""
## Try your own

Every table above is the same handful of moving parts: a true effect, a sample size, and how many
times you are willing to replay it. Change them.
""")

code('''
# --- Your design -----------------------------------------------------------
# MY_D    = 0.35      # the true effect, in population SDs
# MY_N    = 30        # per group
# MY_REPS = 40000
#
# p, d_hat = simulate(MY_D, MY_N, MY_REPS, seed=0)
# sig = p < ALPHA
# q = {k: float(np.percentile(p, k)) for k in (5, 50, 95)}
#
# print(f"power           {sig.mean():.4f}   (analytic {analytic_power(MY_D, MY_N):.4f})")
# print(f"p, 5th pct      {q[5]:.2e}")
# print(f"p, median       {q[50]:.4f}")
# print(f"p, 95th pct     {q[95]:.4f}")
# print(f"straddles 0.05  {q[5] < ALPHA < q[95]}")
# print(f"type M          {np.mean(np.abs(d_hat[sig])) / MY_D:.2f}x")
# print(f"type S          {(d_hat[sig] < 0).mean():.4f}")
# print(f"n for 80% power {required_n(MY_D)} per group")

# --- Or: what would it take to make this study trustworthy? ----------------
# for n in (10, 20, 50, 100, 200, 400):
#     print(f"n={n:>4}  power={analytic_power(0.35, n):.3f}")
''')

md(f"""
---

**Built by Phoebe Fu** - one mini product a day. [Repo](https://github.com/{REPO}) ·
[This build]({PATH})

The same study runs as a Streamlit app where you drag the design around instead of editing
constants:

```bash
pip install -r requirements.txt
streamlit run app.py
```

`evidence.py` regenerates every number quoted here into `evidence.txt` and `results.json`, and
`pytest` asserts the findings - including that each predicate is *capable* of returning either
answer, so none of them is a constant wearing a test.

Sibling builds in this domain: [`t-test-variants`](../t-test-variants) (which test),
[`normality-test-trap`](../normality-test-trap) and
[`assumption-pretest-cost`](../assumption-pretest-cost) (what the pretests cost). All three live
under a true null and measure how often a procedure rejects when it should not. This one is the
other side of that coin.
""")

# The write is guarded because test_notebook.py IMPORTS this module (to assert the notebook's
# embedded engine is this extraction, character for character). Unguarded, that import re-emits
# demo.ipynb - silently replacing the executed, pre-rendered notebook with a blank one, which is
# exactly what happened the first time and which no test could detect afterwards because the
# test had caused it.
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
