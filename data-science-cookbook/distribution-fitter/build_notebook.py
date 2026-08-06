"""Generate demo.ipynb for distribution-fitter. Run once, then nbconvert --execute.

The notebook is self-contained: the setup cell writes `fitting.py` and `evidence.py` to
disk from embedded source, so Colab and Binder get the same modules the repo has without a
clone step, and there is no second copy of the logic to drift out of sync.
"""

from __future__ import annotations

import json
import pathlib

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def md(src: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    nb["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}
    )


BASE = "data-science-cookbook/distribution-fitter"

FITTING = pathlib.Path("fitting.py").read_text()
EVIDENCE = pathlib.Path("evidence.py").read_text()
for src in (FITTING, EVIDENCE):
    assert "'''" not in src, "embedded source must not contain triple single-quotes"


# --------------------------------------------------------------------------------------

md(f"""# Distribution Fitter

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/{BASE}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath={BASE}/demo.ipynb)

> Fitting ten distributions and ranking them by AIC is four lines of scipy. The ranking will
> always produce a winner - including when the data came from something that is not in the
> list. **AIC answers a relative question and gets read as an absolute one.**

**What this covers**

1. The standard workflow, on data where it works
2. **The main event**: the KS p-value is wrong when you fit the parameters first
3. AIC's confident winner on data no candidate can describe
4. Support is a constraint, not a nuisance
5. What a free location parameter actually buys
6. Rounding, sample size, and rejecting the true family
7. Akaike weight vs bootstrap selection stability
8. The verdict the tool is entitled to print
9. Try your own column

*Fully offline - all data is generated, numpy + scipy + matplotlib only.*""")

# --------------------------------------------------------------------------------------

md("""## 0. Setup

The two modules are written to disk from embedded source, so this runs identically in Colab,
in Binder, and next to the repo.""")

code(
    "import pathlib\n\n"
    "FITTING_SRC = r'''" + FITTING + "'''\n\n"
    "EVIDENCE_SRC = r'''" + EVIDENCE + "'''\n\n"
    'pathlib.Path("fitting.py").write_text(FITTING_SRC)\n'
    'pathlib.Path("evidence.py").write_text(EVIDENCE_SRC)\n'
    'print("wrote fitting.py and evidence.py")'
)

code(
    "from __future__ import annotations\n\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n\n"
    "from fitting import (\n"
    "    bootstrap_ks,\n"
    "    diagnose,\n"
    "    family,\n"
    "    fit_distributions,\n"
    "    fit_params,\n"
    "    qq_points,\n"
    "    sample_book,\n"
    ")\n\n"
    'plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.35,\n'
    '                     "axes.spines.top": False, "axes.spines.right": False,\n'
    '                     "font.size": 9})\n'
    'PASS, FAIL, ACCENT, MUTED = "#2a7f62", "#b4451f", "#4460a0", "#9a9aa8"\n\n'
    "book = sample_book()\n"
    "for name, col in book.items():\n"
    '    d = diagnose(col)\n'
    '    print(f"{name:<18} n={d.n:>5}  unique={d.n_unique:>5}  '
    'ties={d.tie_fraction:.3f}  skew={d.skew:+.2f}")'
)

# --------------------------------------------------------------------------------------

md("""## 1. The standard workflow, on data where it works

`session_seconds` really is lognormal. This is the case the usual four lines of scipy handle
correctly, and it is worth seeing what "correct" looks like before breaking it.

The table below adds three columns a normal workflow does not have: `p naive`, `p boot`, and
`win%`. Ignore them for now - the rest of the notebook is about why they are there.""")

code(
    'rep = fit_distributions(book["session_seconds"], n_boot=200, stability_reps=100, seed=1,\n'
    "                        probe_location=False)\n"
    "print(rep.table())\n"
    'print()\nprint(rep.verdict())'
)

md("""The QQ plot is the picture of that first row. Points on the dashed line mean the fitted
quantiles match the observed ones.""")

code(
    'x = book["session_seconds"]\n'
    'fam = family("lognormal")\n'
    "params = fit_params(fam, x)\n"
    "theo, emp = qq_points(fam, x, params)\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "lo, hi = float(emp.min()), float(emp.max())\n"
    'axes[0].plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)\n'
    "axes[0].scatter(theo, emp, s=6, color=PASS, alpha=0.5, linewidths=0)\n"
    "axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)\n"
    'axes[0].set_title("QQ: lognormal fit, lognormal data")\n'
    'axes[0].set_xlabel("theoretical quantile"); axes[0].set_ylabel("observed quantile")\n\n'
    'axes[1].hist(x, bins=60, density=True, color="#e6e6ee", edgecolor="#c9c9d6", lw=0.3)\n'
    "grid = np.linspace(x.min(), np.percentile(x, 99.5), 400)\n"
    "for name, colour in ((\"lognormal\", PASS), (\"gamma\", FAIL), (\"weibull\", MUTED)):\n"
    "    f = family(name)\n"
    "    axes[1].plot(grid, f.dist.pdf(grid, *fit_params(f, x)), color=colour, lw=1.6, label=name)\n"
    "axes[1].legend(frameon=False)\n"
    'axes[1].set_title("Top three fitted densities"); axes[1].set_xlabel("session seconds")\n'
    "plt.tight_layout(); plt.show()"
)

# --------------------------------------------------------------------------------------

md("""## 2. The main event: the KS p-value is wrong when you fit the parameters first

Look at the `p naive` column in the table above. Every rejected family has `p naive = 0.000`,
which looks decisive. Now look at the winning row: `p naive` is around **0.90**.

That number is not a 90% chance the data is lognormal, and it is not even a correctly
calibrated p-value. `scipy.stats.ks_1samp(x, dist.cdf, args=fitted_params)` computes the KS
distance between the data and a CDF **that was chosen to be close to that data**, then
compares it against a reference distribution that assumes the parameters were fixed in
advance. The estimation shrinks the distance; the reference does not know that.

The consequence is measurable. Simulate 200 datasets **from the family being tested**, so the
null hypothesis is true by construction. A correctly calibrated test rejects 5% of the time
and its p-values are uniform on [0, 1], averaging 0.5.""")

code(
    "from evidence import ks_calibration, ks_power, calibration_table\n\n"
    "cal_true = ks_calibration(\"normal\", n=200, n_datasets=200, n_boot=150, seed=11)\n"
    "cal_ln = ks_calibration(\"lognormal\", n=200, n_datasets=120, n_boot=150, seed=12)\n"
    "print(calibration_table([cal_true, cal_ln]))"
)

md("""The naive test rejects **essentially never**, and its p-values average around 0.8 instead
of 0.5. That is not conservatism you can bank - it is the estimation shrinkage showing up as
fake evidence *in favour of* the null, and a test that never rejects a true null has no power
left for a false one.

The fix is a parametric bootstrap, and the whole content of the fix is one word: **refit**.

```python
for b in range(B):
    xs      = dist.rvs(*fitted_params, size=n)   # simulate from the fitted model
    fit_b   = dist.fit(xs)                       # <-- refit, on the simulated sample
    D_b     = ks_distance(xs, dist.cdf, fit_b)   # this D enjoys the same shrinkage
```

Simulating and then comparing against the *original* fitted CDF reproduces the textbook null
and is just as wrong. Refitting each replicate reproduces the shrinkage the observed distance
also got, which is what makes the two comparable. (This is the Lilliefors construction,
generalised past the normal.)""")

code(
    "bins = np.linspace(0, 1, 21)\n"
    "p_naive, p_boot = cal_true.p_naive_all, cal_true.p_bootstrap_all\n\n"
    "fig, ax = plt.subplots(figsize=(8, 3.8))\n"
    'ax.hist(p_naive, bins=bins, color=FAIL, alpha=0.65,\n'
    '        label=f"naive  (rejects {np.mean(p_naive < 0.05):.1%}, mean {p_naive.mean():.2f})")\n'
    'ax.hist(p_boot, bins=bins, color=PASS, alpha=0.65,\n'
    '        label=f"bootstrap  (rejects {np.mean(p_boot < 0.05):.1%}, mean {p_boot.mean():.2f})")\n'
    'ax.axhline(len(p_naive) / 20, color="#1b1b1f", ls="--", lw=1)\n'
    'ax.set_title("p-values over 200 datasets where the null is TRUE")\n'
    'ax.set_xlabel("p-value"); ax.set_ylabel("datasets")\n'
    "ax.legend(frameon=False, loc=\"upper left\")\n"
    "plt.tight_layout(); plt.show()"
)

md("""The dashed line is what a flat (correctly calibrated) histogram would look like. The green
bars are roughly flat. The red bars are stacked against the right edge.

Losing the false-null case is the part that costs money. Same machinery, data now drawn from a
Student-t while a normal is fitted, so rejection is the right answer:""")

code(
    "pw = ks_power(\"normal\", \"student_t\", n=400, n_datasets=100, n_boot=150, seed=13)\n"
    "print(calibration_table([cal_true, pw]))"
)

# --------------------------------------------------------------------------------------

md("""## 3. AIC's confident winner on data no candidate can describe

`latency_ms` is a two-component mixture: a fast path and a slow path. No single-family fit can
represent that, and the candidate list contains no mixtures.

AIC does not care. It ranks what it was given and reports a winner with an Akaike weight of
1.00 - the same output shape it produced in section 1, when the answer was right.""")

code(
    'rep_mix = fit_distributions(book["latency_ms"], n_boot=200, stability_reps=100, seed=17,\n'
    "                            probe_location=False)\n"
    "print(rep_mix.table())\n"
    'print()\nprint(rep_mix.verdict())'
)

code(
    'x = book["latency_ms"]\n'
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
    "bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 60)\n"
    'axes[0].hist(x, bins=bins, density=True, color="#e6e6ee", edgecolor="#c9c9d6", lw=0.3)\n'
    "grid = np.logspace(np.log10(x.min()), np.log10(x.max()), 500)\n"
    "for name, colour in ((rep_mix.best.name, FAIL), (\"lognormal\", ACCENT)):\n"
    "    f = family(name)\n"
    "    axes[0].plot(grid, f.dist.pdf(grid, *fit_params(f, x)), color=colour, lw=1.6, label=name)\n"
    'axes[0].set_xscale("log"); axes[0].legend(frameon=False)\n'
    'axes[0].set_title("Two modes, and no candidate has two")\n'
    'axes[0].set_xlabel("latency (ms, log scale)")\n\n'
    "best = rep_mix.best\n"
    "theo, emp = qq_points(best.family, x, best.params)\n"
    "lo, hi = float(emp.min()), float(emp.max())\n"
    'axes[1].plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1)\n'
    "axes[1].scatter(theo, emp, s=6, color=FAIL, alpha=0.5, linewidths=0)\n"
    "axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)\n"
    'axes[1].set_title(f"QQ: the AIC winner ({best.name}), rejected at p={best.ks.p_bootstrap:.3f}")\n'
    'axes[1].set_xlabel("theoretical quantile"); axes[1].set_ylabel("observed quantile")\n'
    "plt.tight_layout(); plt.show()"
)

md("""**The AIC column cannot tell these two cases apart.** Section 1 and section 3 both produce a
winner with weight 1.00 and a 100% bootstrap win share. The only column that separates "this is
the distribution" from "this is the nearest thing in a list that does not contain the
distribution" is the absolute test.

That is why the tool prints `NO ADEQUATE FIT` rather than a winner, and why the winner is still
shown underneath it: least-bad is genuinely useful, as long as it is labelled.""")

# --------------------------------------------------------------------------------------

md("""### What "no adequate fit" costs downstream

A fitted distribution is not the deliverable. The deliverable is a number read off its tail: a
p99 for an SLA, a p99.9 for capacity, a VaR, a simulation input. The tail is the part of the fit
the body of the data constrains least, so this is exactly where "least-bad" stops being harmless.""")

code(
    "from evidence import tail_error, tail_table\n\n"
    "print(tail_table(tail_error()))"
)

md("""Every candidate understates the p99 by 30-58%. Then the AIC winner turns around and
**overstates the p99.9 by nearly 6x**, while the others understate it by half.

There is no safe direction to round, and no constant to multiply by. An SLA set from the winning
fit would be simultaneously too aggressive at p99 and absurdly slack at p99.9 - and nothing in
the AIC table, the weight, or the win share would have warned you. Only the absolute test does.""")

md("""## 4. Support is a constraint, not a nuisance

`daily_return` has 481 negative values. A lognormal cannot describe it - not "fits badly",
*cannot*, because the density is zero below the origin.

The tempting move is to drop the offending rows and fit anyway. That silently changes the
dataset, and an AIC computed on 419 rows is not comparable to one computed on 900. So
out-of-support families are excluded with a reason instead.""")

code(
    'rep_ret = fit_distributions(book["daily_return"], n_boot=200, stability_reps=100, seed=19,\n'
    "                            probe_location=False)\n"
    "print(rep_ret.table())\n"
    'print()\nprint(rep_ret.verdict())'
)

md("""Note the second row. `logistic` is rejected by the bootstrap test at p = 0.02 while its
naive p-value is around 0.47 - a family that a conventional goodness-of-fit check would have
kept for a returns model, where the tail is the entire point.""")

# --------------------------------------------------------------------------------------

md("""## 5. What a free location parameter actually buys

`lognorm.fit(x)` estimates three parameters, not two: shape, **loc**, and scale. Positive-support
families here are fit with `floc=0`, and that choice is worth defending.

Below: data generated from `gamma(2.4, scale=18)` with **loc = 0**, so the third parameter has
nothing to find.""")

code(
    "from evidence import free_location_cost, location_table\n\n"
    "print(location_table(free_location_cost(n=1200, seed=31)))"
)

md("""Three different failures, none of them "it recovers zero":

- **gamma** is the true family. The free parameter buys ~0 log-likelihood, so AIC's +2 penalty
  correctly rejects it. This is the case the textbook describes.
- **lognormal** is a wrong family, and the shift lets it imitate a gamma - so it gains a large
  amount of log-likelihood and its 3-parameter version beats its own 2-parameter version by a
  wide AIC margin. **The reward for the extra parameter scales with how wrong the family is**,
  which is precisely backwards in a contest meant to identify the right one.
- **weibull** is worse than either. The 3-parameter MLE is unbounded as `loc` approaches
  `min(x)` from below, the optimiser walks toward that singularity and steps over it, and scipy
  returns a `loc` **above** the smallest observation. Observed data points then have zero
  density under the fitted model and the log-likelihood is `-inf`. It is not a bad fit; it is
  not a probability model for this data at all.

Mixing pinned and unpinned fits in one AIC table scores all three of these against each other
as though they were comparable.""")

# --------------------------------------------------------------------------------------

md("""## 6. Rounding, sample size, and rejecting the true family

Every real measurement is rounded. Prices to cents, durations to milliseconds, ages to years.
Rounding makes the data discrete, which makes the empirical CDF a step function with visible
jumps, which inflates the KS distance for reasons that have nothing to do with shape.

Below, the data is **genuinely normal in every row**. Only `n` and the rounding change.""")

code(
    "from evidence import rounding_vs_n, rounding_table\n\n"
    "sizes = (100, 400, 1600, 6400, 20000)\n"
    "print(rounding_table(rounding_vs_n(sizes=sizes, decimals=1, n_boot=200, seed=29)))"
)

md("""And the control - the same sizes, unrounded:""")

code(
    "print(rounding_table(rounding_vs_n(sizes=sizes, decimals=None, n_boot=200, seed=29)))"
)

code(
    "seeds = (29, 131, 233, 337, 439)\n"
    "def mean_curve(decimals):\n"
    "    out = []\n"
    "    for n in sizes:\n"
    "        vals = [rounding_vs_n(sizes=(n,), decimals=decimals, n_boot=150, seed=s)[0].p_bootstrap\n"
    "                for s in seeds]\n"
    "        out.append(float(np.mean(vals)))\n"
    "    return out\n\n"
    "rounded_curve, raw_curve = mean_curve(1), mean_curve(None)\n\n"
    "fig, ax = plt.subplots(figsize=(8, 3.8))\n"
    'ax.plot(sizes, rounded_curve, marker="o", color=FAIL, lw=1.5, label="rounded to 1 dp")\n'
    'ax.plot(sizes, raw_curve, marker="o", color=PASS, lw=1.5, label="unrounded")\n'
    'ax.axhline(0.05, color="#1b1b1f", ls="--", lw=1)\n'
    'ax.text(sizes[0], 0.075, "alpha = 0.05", fontsize=8)\n'
    'ax.set_xscale("log"); ax.set_ylim(0, 1)\n'
    'ax.set_xlabel("n (log scale)"); ax.set_ylabel("mean bootstrap KS p-value")\n'
    'ax.set_title("The data is genuinely normal in both lines")\n'
    "ax.legend(frameon=False)\n"
    "plt.tight_layout(); plt.show()\n\n"
    'print("averaged over", len(seeds), "independent datasets per point - a single dataset per")\n'
    'print("point would plot uniform noise, because that is what a p-value under a true null is")'
)

md("""The red line crosses alpha somewhere around n = 1600 and never comes back. Nothing about
the data's shape changed; the reference distribution shrank like 1/sqrt(n) while the rounding
artefact stayed put.

This is not a bug in the test. It is the test correctly answering a question nobody wanted to
ask. Past a few thousand rows, "is this exactly normal" is always answerable and always no. The
useful question becomes "is the departure big enough to matter for the decision downstream",
which is a question about your loss function, not about a p-value.

Hence the tie diagnostic, which runs before any of this:""")

code(
    'for name in ("session_seconds", "basket_value", "latency_ms"):\n'
    "    print(name)\n"
    "    print(diagnose(book[name]).describe())\n"
    "    print()"
)

# --------------------------------------------------------------------------------------

md("""## 7. Akaike weight vs bootstrap selection stability

The Akaike weight is a transformation of one dataset's delta-AIC. It is routinely read as "the
probability this is the best model", but it is computed from the single delta you happened to
observe, on the single sample you happened to draw.

The bootstrap win share measures the same intent directly: resample the rows, rerun the entire
comparison, count how often each family comes first.""")

code(
    "from evidence import stability_vs_n, stability_table\n\n"
    "rows = stability_vs_n(true_family=\"gamma\",\n"
    "                      rivals=(\"gamma\", \"lognormal\", \"weibull\", \"normal\"),\n"
    "                      sizes=(100, 400, 1600, 6400), reps=100, seed=23)\n"
    "print(stability_table(rows))"
)

code(
    "small = np.random.default_rng(41).gamma(2.4, 18.0, 150)\n"
    "rep_small = fit_distributions(\n"
    "    small, families=[family(n) for n in (\"gamma\", \"lognormal\", \"weibull\", \"exponential\")],\n"
    "    n_boot=0, stability_reps=300, seed=41, probe_location=False)\n\n"
    "ranked = rep_small.ranked\n"
    "names = [r.name for r in ranked]\n"
    "idx = np.arange(len(names))\n"
    "fig, ax = plt.subplots(figsize=(8, 3.4))\n"
    'ax.barh(idx + 0.19, [r.aic_weight for r in ranked], height=0.36, color=ACCENT,\n'
    '        label="Akaike weight")\n'
    'ax.barh(idx - 0.19, [r.win_share for r in ranked], height=0.36, color=PASS,\n'
    '        label="bootstrap win share")\n'
    "for i, r in enumerate(ranked):\n"
    '    ax.text(r.aic_weight + 0.012, i + 0.19, f"{r.aic_weight:.2f}", va="center", fontsize=8)\n'
    '    ax.text(r.win_share + 0.012, i - 0.19, f"{r.win_share:.2f}", va="center", fontsize=8)\n'
    "ax.set_yticks(idx); ax.set_yticklabels(names); ax.invert_yaxis(); ax.set_xlim(0, 1.15)\n"
    'ax.set_xlabel("share"); ax.set_title("n=150 gamma sample: weight says certain, resampling does not")\n'
    "ax.legend(frameon=False, loc=\"lower right\")\n"
    "plt.tight_layout(); plt.show()"
)

md("""On this n=150 gamma sample the Akaike weight puts **0.82** on gamma; resampling the rows has
gamma actually winning **0.62** of the time. Twenty points of confidence that only exists because
the delta was computed once.

The two numbers converge as n grows - which is the point. The weight does not shrink its own
confidence when the sample is small; the bootstrap does it for you.

The n=400 row of the table above is the sharper case: the AIC winner there is **weibull**, with
the true gamma trailing by dAIC = 1.41. Under the usual "delta-AIC below 2 means indistinguishable"
convention that is a tie you would report as a tie. The win shares say something more useful -
neither family holds a majority, so there is no winner to report at all.""")

# --------------------------------------------------------------------------------------

md("""## 8. The verdict

Putting the whole thing together on all four sample columns. Note that the four verdicts are
genuinely different sentences, not four copies of one template with the family name swapped.""")

code(
    "for name, col in book.items():\n"
    '    r = fit_distributions(col, n_boot=200, stability_reps=100, seed=2, probe_location=False)\n'
    '    print("=" * 88)\n'
    '    print(f"{name}  (n={r.diagnostics.n})")\n'
    '    print("-" * 88)\n'
    "    print(r.verdict())\n"
    "    print()"
)

md("""| column | truth | what the tool says |
|---|---|---|
| `session_seconds` | lognormal | lognormal, not rejected, stable |
| `basket_value` | gamma, rounded to cents | gamma, not rejected, with a ties warning |
| `latency_ms` | two-component mixture | **no adequate fit**, winner labelled least-bad |
| `daily_return` | Student-t | student_t, not rejected; normal rejected |

Four different answers, and the one that matters most is the one that refuses to name a
distribution.""")

# --------------------------------------------------------------------------------------

md("""## 9. Try your own column

Uncomment, point it at your own data, and read the verdict before the table.""")

code(
    "# import pandas as pd\n"
    "# frame = pd.read_csv(\"your_data.csv\")\n"
    "# x = frame[\"your_column\"].dropna().to_numpy(dtype=float)\n"
    "#\n"
    "# report = fit_distributions(\n"
    "#     x,\n"
    "#     n_boot=300,          # resolution of the absolute test; min p is 1/(B+1)\n"
    "#     stability_reps=200,  # resolution of the win shares\n"
    "#     alpha=0.05,\n"
    "#     seed=0,\n"
    "# )\n"
    "# print(report.diagnostics.describe())   # check the ties BEFORE reading any p-value\n"
    "# print()\n"
    "# print(report.table())\n"
    "# print()\n"
    "# print(report.verdict())\n"
    "#\n"
    "# Three things to read, in this order:\n"
    "#   1. tie fraction   - heavy ties invalidate the absolute tests before you start\n"
    "#   2. the verdict    - is ANY candidate not rejected, or is the winner just least-bad\n"
    "#   3. win share      - would the winner win again on resampled rows\n"
    "# The AIC column is the last thing to look at, not the first.\n"
    "print(\"ready - uncomment the block above and point it at your own CSV\")"
)

# --------------------------------------------------------------------------------------

md(f"""---

## What to take away

1. **AIC ranks; it does not test.** A winner is produced whether or not the candidate set
   contains the answer. Always pair the ranking with an absolute goodness-of-fit test.
2. **A KS p-value computed on fitted parameters is not a KS p-value.** Over 200 true-null
   datasets it rejected 0% of the time and averaged 0.79 instead of 0.5. Refit inside a
   parametric bootstrap; the fix is one line and it restores both calibration and power.
3. **Support violations are exclusions, not filters.** Fitting a family to the subset of rows
   it can describe puts its AIC on a different dataset from every other candidate's.
4. **A free location parameter rewards the wrong family more than the right one**, and its MLE
   is unbounded at `min(x)` - scipy will hand back fits that assign zero density to observed
   points.
5. **Past a few thousand rows, everything is rejected, including the truth.** Measure the ties
   first and treat "is this exactly parametric" as a question about n.

## Run the code

```bash
git clone https://github.com/phoebefu6/phoebe-the-builder
cd phoebe-the-builder/{BASE}
pip install -r requirements.txt

python3 test_fitting.py     # 21 tests over the fitting core
python3 test_evidence.py    #  9 tests over the experiments above
python3 make_chart.py       # the six-panel audit figure
streamlit run app.py        # the interactive version
```

Part of [phoebe-the-builder](https://github.com/phoebefu6/phoebe-the-builder) - Day 136,
Data Science Cookbook.""")


pathlib.Path("demo.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote demo.ipynb with {len(nb['cells'])} cells")
