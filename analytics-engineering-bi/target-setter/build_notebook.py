"""Generate demo.ipynb.

The engine cell is the *actual text of* ``targets.py``, read at build time,
so the notebook and the module cannot drift apart. Everything after it calls
into that engine, exactly as ``evidence.py`` does.
"""

from __future__ import annotations

import nbformat as nbf

REPO = "phoebefu6/phoebe-the-builder"
PATH = "analytics-engineering-bi/target-setter"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


ENGINE = open("targets.py").read()

cells = []

cells.append(md(f"""
# Where did the number come from?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

**Day 159 of the FDE portfolio - Target Setter.**

Somebody asks what next quarter's number should be. Six people in the room have
six ways of answering, all of them defensible, and one of them ends up on the
slide. Then, three months later, the same number is used to decide whether the
team did well.

This notebook sets one quarter's target **twelve** defensible ways on one
history, and then re-runs the same eleven years hundreds of times to see what
the resulting hit rates actually measure.

The short version:

| # | what gets measured | the number |
|---|---|---|
| 1 | how far apart twelve defensible targets are | **1.40x**, a spread of **31.1%** of the base, on a quarter that moved 11.7% |
| 2 | Spearman(ambition, hit rate) across 500 re-runs | **-0.91** - the hit rate is a property of the method |
| 3 | hit rate of a target set at the true *mean* of the future | **0.486** - an unbiased target is missed more often than it is met |
| 4 | quarters needed to tell a 0.50 hitter from a 0.65 hitter | **134**, or 33.5 years |
| 5 | swing in one method's target by the month it is set in | **51.8%** |
| 6 | the compromise between top-down and bottom-up | closest of the three to the forecast, met **3.4x** less often |
| 7 | value of choosing the softer of two defensible methods | **12.6 months** of real growth, and +48 points of hit rate |
| 9 | pairwise disagreements smaller than the 80% interval | **52 of 66** |

Those figures come from the full audit (`python evidence.py`, 500 re-runs).
The cells below use 120-150 re-runs so the notebook executes in seconds, so the
sampling-error digits move a little. Nothing that matters does.

Everything below runs from the standard library plus numpy, scipy and
matplotlib. No data is loaded and no API is called.
"""))

cells.append(md("""
## 1. A world where we know the answer

The only way to say something true about a target is to know what it was aiming
at. So the metric here is simulated, from a process written down in five
constants:

$$y_t = \\text{BASE}\;(1+g)^t\;S_{t \\bmod 12}\;e^{\\varepsilon_t},
\\qquad \\varepsilon_t \\sim N(0, \\sigma^2)$$

Monthly signups for a mid-market SaaS product: compounding growth, a December
peak, a summer trough, and proportional noise. Because the process is known,
every target can be scored against **the truth it was aiming at** as well as
against what happened to occur.

The cell below is the whole engine - the same file the tests and the Streamlit
app import, pasted in so this notebook stands alone.
"""))

cells.append(code(ENGINE))

cells.append(md("""
## 2. Twelve defensible targets for the same quarter

Each of these is a sentence somebody says out loud in a planning meeting. None
of them is a straw man; four of them are in every planning deck ever written.

The target covers months 120-122 - the start of year eleven.
"""))

cells.append(code('''
import numpy as np
import pandas as pd

ORIGIN = 120
series = make_history()
tg = targets_at(series, ORIGIN)
truth_mean, _ = truth_quarter(ORIGIN)
last_q = float(series[ORIGIN - HORIZON:ORIGIN].sum())
actual = float(series[ORIGIN:ORIGIN + HORIZON].sum())

rows = []
for name in sorted(tg, key=lambda k: tg[k]):
    rows.append({
        "method": name,
        "provenance": PROVENANCE[name],
        "target": round(tg[name]),
        "vs last quarter": f"{tg[name] / last_q - 1:+.1%}",
        "vs the truth": round(tg[name] / truth_mean, 3),
        "met?": "met" if actual >= tg[name] else "missed",
    })
print(f"last quarter = {last_q:,.0f}   E[next quarter] = {truth_mean:,.0f}   "
      f"actual = {actual:,.0f}")
pd.DataFrame(rows)
'''))

cells.append(code('''
lo, hi = min(tg.values()), max(tg.values())
print(f"highest / lowest              : {hi / lo:.3f}x")
print(f"spread as a share of last qtr : {(hi - lo) / last_q:.1%}")
print(f"the quarter itself moved      : {actual / last_q - 1:+.1%}")
print()
print("The twelve methods disagree by more than the metric moved.")
print("The choice of method is a larger number than the thing being targeted.")
'''))

cells.append(md("""
## 3. The hit rate is a property of the method, not the team

*Ambition* is the only thing about a target that is fixed the moment it is set:
the target as a multiple of the truth it aims at. If the hit rate is a function
of ambition, then a hit rate grades the choice of method, not the work.

Re-run the same eleven years from 150 independent draws of the same process.
"""))

cells.append(code('''
from scipy import stats

mp = multipath(150, 12_345)
names = list(METHODS)
amb = np.array([mp[n]["ambition"].mean() for n in names])
hit = np.array([mp[n]["hit_rate"].mean() for n in names])

tbl = pd.DataFrame({
    "method": names,
    "ambition": np.round(amb, 3),
    "hit rate": np.round(hit, 3),
    "sd across re-runs": np.round([mp[n]["hit_rate"].std() for n in names], 3),
}).sort_values("ambition").reset_index(drop=True)

rho, p = stats.spearmanr(amb, hit)
print(f"Spearman(ambition, hit rate) = {rho:.3f}   p = {p:.1e}")
print(f"hit rate spans {hit.min():.3f} to {hit.max():.3f} on identical work")
tbl
'''))

cells.append(md("""
Ambition explains most of it, and not all of it. Some **harder** targets are met
**more** often, because a target is a random variable too: how often it is met
depends on its correlation with the actual, not only on how high it sits.
"""))

cells.append(code('''
import itertools

inv = [
    (names[i], names[j], hit[i], hit[j])
    for i, j in itertools.combinations(range(len(names)), 2)
    if (amb[i] - amb[j]) * (hit[i] - hit[j]) > 0
]
print(f"{len(inv)} of {len(names) * (len(names) - 1) // 2} pairs are inverted:")
for a, b, ha, hb in inv:
    harder, softer = (b, a) if amb[names.index(b)] > amb[names.index(a)] else (a, b)
    print(f"   {harder} is harder and met more often than {softer}")
'''))

cells.append(md("""
## 4. An unbiased target is missed more often than it is hit

The metric is lognormal, so its **mean sits above its median**. A target set at
the expected value is above the middle of the distribution before anybody has
done any work.

These are *oracle* targets - set at the true mean and the true median of the
future, with no estimation involved at all. They are the ceiling on what any
forecast-based target can claim.
"""))

cells.append(code('''
oracle = oracle_hit_rates(120, 33_000)
print(f"P(one month >= its own mean)   : {1 - stats.norm.cdf(SIGMA / 2):.4f}")
print(f"target at the TRUE mean        : {oracle['mean_target']:.4f}")
print(f"target at the TRUE median      : {oracle['median_target']:.4f}")
print()
gap = tg["trend_seasonal"] / tg["trend_seasonal_median"] - 1
d = mp["trend_seasonal_median"]["hit_rate"].mean() - mp["trend_seasonal"]["hit_rate"].mean()
print(f"Same model, mean vs median target: {gap:.2%} apart in the number,")
print(f"                                  {d * 100:.1f} points apart in the hit rate.")
print("Nobody in the meeting will notice the first number.")
'''))

cells.append(md("""
## 5. A hit rate is not a reproducible measurement

Re-running the same eleven years from a fresh draw moves the hit rate. Sort the
methods by how much:
"""))

cells.append(code('''
sd_tbl = pd.DataFrame([
    {
        "method": n,
        "mean": round(float(mp[n]["hit_rate"].mean()), 3),
        "sd": round(float(mp[n]["hit_rate"].std()), 3),
        "p05": round(float(np.quantile(mp[n]["hit_rate"], 0.05)), 3),
        "p95": round(float(np.quantile(mp[n]["hit_rate"], 0.95)), 3),
    }
    for n in names
]).sort_values("sd", ascending=False).reset_index(drop=True)
sd_tbl
'''))

cells.append(code('''
n = quarters_to_distinguish(0.50, 0.65)
print(f"Quarters to tell a 0.50 hitter from a 0.65 hitter: {n}  ({n / 4:.1f} years)")
print("An eleven-year-old company has 44 quarters.")
print()
pv = stats.binomtest(8, 12, 0.5, alternative="greater").pvalue
print(f"'We hit 8 of our last 12 targets' -> p = {pv:.3f} against a coin.")
print()
worst_two = list(sd_tbl["method"][:2])
print(f"The two LEAST reproducible methods are {worst_two[0]} and {worst_two[1]}")
print("-- the two best-specified forecasts in the list. Forecasting well puts the")
print("target in the middle of the distribution, which is exactly where the")
print("hit/miss verdict is most sensitive to noise. The most reproducible hit")
print("rates belong to the targets nobody would call forecasts.")
'''))

cells.append(md("""
## 6. The planning calendar is a parameter of the target

Same history, same method, different month asked in. `run_rate` annualises the
latest month, so it inherits that month's seasonality and carries it into a
quarter with a different one.
"""))

cells.append(code('''
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ors = origins(series)
truth = {o: truth_quarter(o)[0] for o in ors}

by_method = {}
for nm in ("run_rate", "last_quarter", "trend_ols", "trend_seasonal"):
    fn = METHODS[nm]
    by_method[nm] = [
        float(np.mean([fn(series[:o], o) / truth[o] for o in ors if o % 12 == m]))
        for m in range(12)
    ]
cal = pd.DataFrame(by_method, index=MONTHS).round(3)
for nm, col in by_method.items():
    print(f"{nm:16} swing across the calendar: {max(col) / min(col) - 1:.1%}")
cal
'''))

cells.append(md("""
## 7. Top-down and bottom-up do not reconcile, and the midpoint is achievable
by neither

The board wants 40% on last year. The headcount plan supports what it supports.
The meeting ends by splitting the difference.

Watch the two statistics disagree: the midpoint is the **closest of the three**
to the best available forecast, and the one that gets **missed**. Distance to a
forecast is two-sided; being met is one-sided.
"""))

cells.append(code('''
cap = np.array([m_capacity(series[:o], o) for o in ors])
top = np.array([m_top_down(series[:o], o) for o in ors])
spl = np.array([m_split_difference(series[:o], o) for o in ors])
ref = np.array([m_trend_seasonal(series[:o], o) for o in ors])

print(f"top-down above bottom-up at : {(top > cap).mean():.1%} of origins")
print(f"mean gap between them       : {(top / cap - 1).mean():.1%}")
print()
pd.DataFrame([
    {
        "method": nm,
        "|distance| from forecast": f"{np.abs(v / ref - 1).mean():.1%}",
        "signed distance": f"{(v / ref - 1).mean():+.1%}",
        "hit rate": round(float(mp[nm]["hit_rate"].mean()), 3),
    }
    for nm, v in (("capacity", cap), ("split_difference", spl), ("top_down", top))
])
'''))

cells.append(md("""
## 8. A hit-rate incentive pays for the choice of method

Two sentences, both said out loud in planning meetings, both defensible:

* *"same quarter last year, flat"* - `seasonal_naive`
* *"trend plus seasonality"* - `trend_seasonal`

Nothing about the work changes between them.
"""))

cells.append(code('''
BONUS = 100_000
gap = mp["trend_seasonal"]["ambition"].mean() / mp["seasonal_naive"]["ambition"].mean() - 1
months = np.log(1 + gap) / np.log(1 + G)
print(f"the softer target is {gap:.1%} lower  =  {months:.1f} months of real growth")
print(f"hit rate {mp['seasonal_naive']['hit_rate'].mean():.3f} vs "
      f"{mp['trend_seasonal']['hit_rate'].mean():.3f}\\n")

pay = pd.DataFrame([
    {"method": n,
     "hit rate": round(float(mp[n]["hit_rate"].mean()), 3),
     "E[bonus]": round(float(mp[n]["hit_rate"].mean()) * BONUS)}
    for n in names
]).sort_values("E[bonus]", ascending=False).reset_index(drop=True)

p_real = mp["trend_seasonal"]["hit_rate"].mean()
for nm in ("stretch_best_ever", "top_down"):
    print(f"{nm} needs {p_real / mp[nm]['hit_rate'].mean():.1f}x the payout "
          "to be worth the same attempt")
pay
'''))

cells.append(md("""
## 9. Most of the argument is inside the prediction interval

An 80% interval for the quarter, against the 66 pairwise gaps between the twelve
methods.
"""))

cells.append(code('''
lo_pi, hi_pi = prediction_interval(series, ORIGIN, 0.80)
width = hi_pi - lo_pi
gaps = np.array([abs(a - b) for a, b in itertools.combinations(tg.values(), 2)])
print(f"80% interval : {lo_pi:,.0f} to {hi_pi:,.0f}  (width {width:,.0f}, "
      f"{width / ((lo_pi + hi_pi) / 2):.1%} of the point forecast)")
print(f"pairs closer together than the interval: {(gaps < width).sum()} of {len(gaps)}")
print(f"targets that fall inside the interval  : "
      f"{sum(1 for v in tg.values() if lo_pi <= v <= hi_pi)} of {len(tg)}")
'''))

cells.append(md("""
## 10. The picture

Left: the twelve targets fanning out of one origin, against the 80% interval.
Right: ambition against hit rate, with the inverted pairs joined in orange.
"""))

cells.append(code('''
import matplotlib.pyplot as plt

INK, MUTED, ORANGE, GREEN, RED = "#141414", "#8a8a8a", "#d98324", "#4b7f52", "#c0392b"
fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.8))

lo_m = ORIGIN - 30
ax[0].plot(np.arange(lo_m, ORIGIN), series[lo_m:ORIGIN], color=INK, lw=1.1)
ax[0].plot(np.arange(ORIGIN, ORIGIN + HORIZON),
           series[ORIGIN:ORIGIN + HORIZON], color=INK, lw=1.1, ls=":")
x = ORIGIN + np.array([0.0, HORIZON - 1])
ax[0].fill_between(x, lo_pi / HORIZON, hi_pi / HORIZON, color=ORANGE, alpha=0.15)
for nm, v in tg.items():
    met = actual >= v
    ax[0].plot(x, [v / HORIZON] * 2, color=GREEN if met else RED, lw=2.2)
ax[0].axhline(actual / HORIZON, color=INK, lw=0.8, ls="--")
ax[0].set_title("twelve targets, one quarter\\ngreen = met, red = missed, band = 80% interval",
                loc="left", fontsize=9)
ax[0].set_ylabel("signups / month")

for i, j in itertools.combinations(range(len(names)), 2):
    if (amb[i] - amb[j]) * (hit[i] - hit[j]) > 0:
        ax[1].plot(amb[[i, j]], hit[[i, j]], color=ORANGE, lw=1.0, alpha=0.7)
ax[1].scatter(amb, hit, s=34, color="#4a7c8c", zorder=3)
for nm, a, h in zip(names, amb, hit):
    ax[1].annotate(nm, (a, h), fontsize=6.2, xytext=(5, 0),
                   textcoords="offset points", color=MUTED)
ax[1].axhline(0.5, color=MUTED, lw=0.7, ls=":")
ax[1].set_xlim(0.83, 1.30)
ax[1].set_xlabel("ambition (target / truth)")
ax[1].set_ylabel("hit rate")
ax[1].set_title("the hit rate is a property of the method\\norange = the harder target is met more often",
                loc="left", fontsize=9)
for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.show()
'''))

cells.append(md("""
## What a target is

A target is **a method plus a claim about the future**. On this history the
method moves the number by more than the business does, and the statistic used
to grade the result - the hit rate - is a property of the method that was
chosen, not of the work that was done.

Three things travel with a defensible target, and a single number carries none
of them:

1. **the method** - which of the twelve sentences produced it, written down
2. **the claim** - what it assumes about growth, seasonality and resourcing
3. **the interval** - because 52 of the 66 disagreements in that meeting are
   smaller than the uncertainty everybody is arguing inside

And one thing to stop doing: grading a team on a hit rate. It takes 134 quarters
to tell a 0.50 hitter from a 0.65 hitter, and no company has 33 years.
"""))

cells.append(md("""
## Try your own

Uncomment and edit. The engine cell above is the whole implementation, so any
of it can be changed in place.
"""))

cells.append(code('''
# --- 1. Your own target-setting method ------------------------------------
# def m_house_rule(hist, origin):
#     """Whatever your planning deck actually does."""
#     return float(hist[-12:-12 + HORIZON].sum() * 1.15)
#
# METHODS["house_rule"] = m_house_rule
# PROVENANCE["house_rule"] = "last year's quarter + 15%"
# mine = multipath(150, 999)["house_rule"]
# print(f"ambition {mine['ambition'].mean():.3f}  "
#       f"hit rate {mine['hit_rate'].mean():.3f} "
#       f"(sd {mine['hit_rate'].std():.3f})")

# --- 2. A noisier or flatter world ----------------------------------------
# SIGMA = 0.25      # a metric that moves a lot month to month
# G = 0.004         # a business that is barely growing
# s2 = make_history(seed=7)
# print({k: round(v) for k, v in targets_at(s2, 120).items()})

# --- 3. What would the board have to pay for the stretch target? ----------
# p_real = mp["trend_seasonal"]["hit_rate"].mean()
# for nm in METHODS:
#     p = mp[nm]["hit_rate"].mean()
#     if p > 0:
#         print(f"{nm:24} {p_real / p:6.2f}x payout to match trend_seasonal")
'''))

cells.append(md(f"""
---

**Day 159 of [phoebe-the-builder](https://github.com/{REPO})** - one small,
finished data tool a day.

The Streamlit version puts the world on sliders - noise, real growth, seasonal
amplitude, the board multiple, the hiring plan - and re-runs the hit rates live:

```bash
pip install -r requirements.txt
streamlit run app.py
```

`python evidence.py` prints the full ten-section audit.
`pytest test_targets.py` asserts every number in it.
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(nb, "demo.ipynb")
print(f"wrote demo.ipynb with {len(cells)} cells")
