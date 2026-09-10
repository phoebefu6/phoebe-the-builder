"""Generate demo.ipynb.

The cell-emitting plumbing below is copied verbatim from the novelty-decay build (which took
it from heterogeneous-effects, which took it from synthetic-control). It gets two things right
that fail silently until nbconvert runs: nbformat wants each source line to KEEP its trailing
newline, and every cell needs an id.

The notebook re-derives the engine inline rather than importing backtest.py, so a reader who
opens it in Colab from a bare link gets something that runs. Day 171's bug is the one to avoid
here: the inline re-derivation silently dropped a parameter and the surrounding prose then
asserted the opposite of what the code did. The re-derivation below is diffed against
backtest.py by test_notebook_matches_library in test_backtest.py.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

REPO = "phoebefu6/phoebe-the-builder"
PATH = "data-science-cookbook/forecast-backtest"

cells: List[Dict[str, Any]] = []


def _lines(src: str) -> List[str]:
    """nbformat wants each source line to KEEP its trailing newline."""
    return src.strip("\n").splitlines(keepends=True)


def _nid() -> str:
    global _N
    _N += 1
    return f"c{_N:02d}"


_N = 0


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "id": _nid(), "metadata": {}, "source": _lines(src)})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "id": _nid(),
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _lines(src),
        }
    )


# ---------------------------------------------------------------- 1. title
md(
    f"""
# The forecast was excellent in-sample

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{PATH}/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/{REPO}/main?labpath={PATH}/demo.ipynb)

A backtest does not measure a forecaster. It ESTIMATES a forecaster's future error, and like
any estimate it has a bias and a variance that belong to the PROTOCOL rather than to the model.
This notebook builds a process whose future can be redrawn on demand - so the quantity every
protocol claims to report is computable - and then scores the protocols against it.

**What is here**

1. The estimand: what a backtest is trying to estimate, computed by redrawing the future
2. A single split is one draw of an ORIGIN - and the origin moves the number more than the model does
3. Best-of-M on one split is a minimum operator, and the shared base model shrinks it by sqrt(1-rho)
4. Random k-fold on a lag table: the received warning names the wrong culprit
5. Folds that overlap are not folds - k_eff, and why 7 beats 20
6. The scoreboard: every protocol as an estimator
7. Try your own

This is **not** a forecasting model - [`ts-forecaster`](../ts-forecaster) already built one.
The object under test here is the evaluation protocol.
"""
)

# ---------------------------------------------------------------- 2. the engine
md(
    """
## 1. The estimand

Almost every backtest discussion skips the first question: *what number is the backtest an
estimate OF?* Here it is the **expected 12-step RMSE of this model, trained on this many
points, from this process** - and because the process is synthetic, it can be computed to
Monte Carlo precision by drawing 600 independent futures and scoring each one honestly.

Everything below compares a protocol's answer to that number.
"""
)

code(
    '''
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

PERIOD, H = 12, 12

def make_series(rng, n, level=100.0, trend=0.15, season_amp=8.0, period=PERIOD,
                sigma=4.0, rho=0.0, break_at=None, break_shift=0.0, break_trend=None):
    """Known signal + AR(1) noise, optionally with a regime break."""
    t = np.arange(n, dtype=float)
    signal = level + trend * t + season_amp * np.sin(2 * np.pi * t / period)
    if break_at is not None:
        after = t >= break_at
        signal = signal + break_shift * after
        if break_trend is not None:
            signal = signal + (break_trend - trend) * np.where(after, t - break_at, 0.0)
    if rho == 0.0:
        noise = rng.normal(0.0, sigma, n)
    else:
        e = rng.normal(0.0, sigma * np.sqrt(1 - rho ** 2), n)
        noise = np.empty(n); noise[0] = rng.normal(0.0, sigma)
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + e[i]
    return signal + noise

def _ols_forecast(y, h, period, poly):
    n = len(y); t_in = np.arange(n, dtype=float); t_out = np.arange(n, n + h, dtype=float)
    mu, sd = t_in.mean(), max(t_in.std(), 1e-9)
    def cols(t):
        ts = (t - mu) / sd
        c = [np.ones_like(ts)] + [ts ** p for p in range(1, poly + 1)]
        c += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
        return np.column_stack(c)
    beta, *_ = np.linalg.lstsq(cols(t_in), y, rcond=None)
    return cols(t_out) @ beta

f_naive        = lambda y, h, p=PERIOD: np.repeat(y[-1], h)
f_snaive       = lambda y, h, p=PERIOD: np.array([y[-p:][i % p] for i in range(h)])
f_drift        = lambda y, h, p=PERIOD: y[-1] + (y[-1] - y[0]) / max(len(y) - 1, 1) * np.arange(1, h + 1)
f_mean         = lambda y, h, p=PERIOD: np.repeat(y.mean(), h)
f_trend_season = lambda y, h, p=PERIOD: _ols_forecast(y, h, p, 1)
f_flexible     = lambda y, h, p=PERIOD: _ols_forecast(y, h, p, 6)

MODELS = {"naive": f_naive, "seasonal naive": f_snaive, "drift": f_drift,
          "mean": f_mean, "trend+season": f_trend_season, "flexible (deg 6)": f_flexible}

rmse = lambda e: float(np.sqrt(np.mean(np.asarray(e) ** 2)))

def forecast_at(y, model, origin, h, period=PERIOD, train_len=None):
    train = y[:origin] if train_len is None else y[max(0, origin - train_len):origin]
    return y[origin:origin + h], model(train, h, period)

def true_h_step_error(model, h, n_train, reps, rng, **kw):
    """The estimand. Only computable because the process can be redrawn."""
    errs = []
    for _ in range(reps):
        y = make_series(rng, n_train + h, **kw)
        a, p = forecast_at(y, model, n_train, h)
        errs.append(a - p)
    return rmse(np.concatenate(errs))

N = 180
TRUE = {name: true_h_step_error(m, H, N - H, 600, np.random.default_rng(101))
        for name, m in MODELS.items()}
print(f"{'model':<18}{'true 12-step RMSE':>20}")
for k, v in TRUE.items():
    print(f"{k:<18}{v:>20.3f}")
'''
)

# ---------------------------------------------------------------- 3. one origin
md(
    """
## 2. A single split is one draw of an ORIGIN

"Hold out the last 12 points" sounds like a description of the data. It is a description of
**one origin**, chosen because it happens to be the end of the file. Slide it and the number
moves - on the same series, with the same model, with nothing else changed.
"""
)

code(
    '''
def origins(n, h, k, min_train, step=1):
    return sorted(list(range(n - h, min_train - 1, -step))[:k])

def rolling_origin(y, model, h, k, min_train, period=PERIOD, step=1,
                   window="expanding", train_len=None):
    ors = origins(len(y), h, k, min_train, step)
    out = np.full((len(ors), h), np.nan)
    tl = None if window == "expanding" else (train_len if train_len is not None else min_train)
    for i, o in enumerate(ors):
        a, p = forecast_at(y, model, o, h, period, train_len=tl)
        out[i, :len(a)] = a - p
    return out

single_split = lambda y, model, h, period=PERIOD: forecast_at(y, model, len(y) - h, h, period)[0] \\
                                                  - forecast_at(y, model, len(y) - h, h, period)[1]

y = make_series(np.random.default_rng(3), 400)
E = rolling_origin(y, f_trend_season, H, k=60, min_train=96)
per = np.array([rmse(r[~np.isnan(r)]) for r in E])
ors = origins(len(y), H, 60, 96)
truth = TRUE["trend+season"]

print(f"the honest number for this model      {truth:.3f}")
print(f"best origin available to report       {per.min():.3f}   ({per.min()/truth*100:.0f}% of it)")
print(f"worst origin available to report      {per.max():.3f}   ({per.max()/truth*100:.0f}% of it)")
print(f"spread across {len(per)} origins on ONE series   sd {per.std(ddof=1):.3f}")
'''
)

code(
    '''
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ors, per, color="#2f6fdb", marker="o", ms=3, lw=1.4, label="RMSE if you split HERE")
ax.axhline(truth, color="#1f2733", ls="--", lw=1.8, label=f"the honest number ({truth:.2f})")
ax.scatter([ors[int(np.argmin(per))]], [per.min()], color="#2f8f5b", s=70, zorder=5)
ax.annotate(f"the number you could report: {per.min():.2f}",
            (ors[int(np.argmin(per))], per.min()), textcoords="offset points",
            xytext=(-40, 20), color="#2f8f5b",
            arrowprops=dict(arrowstyle="-", color="#2f8f5b"))
ax.set_xlabel("origin (last point of training)"); ax.set_ylabel("12-step RMSE")
ax.set_title("One series, one model, 60 choices of where to split")
ax.grid(True, color="#dfe4ea", lw=0.7); ax.set_axisbelow(True); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("notebook_figure.png", dpi=140); plt.show()
'''
)

md(
    """
Now the consequence for the decision the split is usually asked to make. Two models are
compared on one split; how often does the split rank them backwards? The answer is a function
of one ratio - the split's own spread divided by the gap between the models - and that ratio
is computable before any modelling happens.
"""
)

code(
    '''
rng = np.random.default_rng(7)
splits = {k: [] for k in MODELS}
for _ in range(300):
    yy = make_series(rng, N)
    for name, m in MODELS.items():
        splits[name].append(rmse(single_split(yy, m, H)))

names = list(MODELS)
pairs = sorted((abs(TRUE[a] - TRUE[b]), a, b) for i, a in enumerate(names) for b in names[i+1:])
print(f"{'pair':<38}{'true gap':>10}{'split sd':>10}{'sd/gap':>9}{'ranked backwards':>19}")
for gap, a, b in pairs[:3] + pairs[-1:]:
    lo, hi = (a, b) if TRUE[a] < TRUE[b] else (b, a)
    d = np.array(splits[hi]) - np.array(splits[lo])
    print(f"{lo + ' vs ' + hi:<38}{gap:>10.3f}{np.std(splits[lo], ddof=1):>10.3f}"
          f"{np.std(splits[lo], ddof=1)/gap:>9.1f}{np.mean(d < 0):>19.3f}")
'''
)

md(
    """
The two closest models are ranked backwards about a third of the time. That is not a small
sample or a bad model - it is one draw of an origin being asked to resolve a difference
smaller than its own noise.
"""
)

# ---------------------------------------------------------------- 4. best of M
md(
    """
## 3. Best-of-M on one split is a minimum operator

Build M candidate forecasters that are **equally good by construction**: the same base fit
plus an offset vector of equal norm, so their true errors differ by ~2%. Then do what every
project does - score them all on one split and report the winner's number.

The winner's number is `E[min of M]` standard deviations below the field. That is arithmetic,
available before any data exists. The one wrinkle worth deriving: candidates that share a base
fit are **correlated**, and an equicorrelated field is `sqrt(1-rho)` times an independent one
plus a common shock with mean zero - so the correlation shrinks the winner's curse by exactly
`sqrt(1-rho)`.
"""
)

code(
    '''
from math import erf, sqrt

def expected_min_of_m(m, rho=0.0, grid=20001):
    """E[min of m equicorrelated standard normals] = sqrt(1-rho) * E[min of m iid]."""
    z = np.linspace(-9, 9, grid)
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    dens = m * (1.0 - Phi) ** (m - 1) * phi
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(sqrt(1.0 - rho) * float(trap(z * dens, z)))

for m in (2, 5, 8, 20):
    print(f"E[min of {m:>2}] = {expected_min_of_m(m):+.4f}")
print(f"E[min of  8] at rho=0.68 = {expected_min_of_m(8, 0.68):+.4f}")
'''
)

code(
    '''
M, DELTA = 8, 2.0
u = np.random.default_rng(21).normal(size=(M, H))
u = u / np.linalg.norm(u, axis=1, keepdims=True) * np.sqrt(H)   # equal norm -> equal true error
cand = lambda j: (lambda yy, h, p=PERIOD: f_trend_season(yy, h, p) + DELTA * u[j, :h])

true_c = [true_h_step_error(cand(j), H, N - H, 400, np.random.default_rng(500 + j)) for j in range(M)]
print(f"true 12-step RMSE of the {M} candidates: {min(true_c):.3f} to {max(true_c):.3f} "
      f"({(max(true_c)/min(true_c)-1)*100:.1f}% apart)")

rng = np.random.default_rng(22)
rep, fut, field, mat = [], [], [], []
for _ in range(400):
    yy = make_series(rng, N + H)
    sp = np.array([rmse(single_split(yy[:N], cand(j), H)) for j in range(M)])
    mat.append(sp); j = int(np.argmin(sp))
    rep.append(sp[j]); field.append(sp.mean())
    a, p = forecast_at(yy, cand(j), N, H); fut.append(rmse(a - p))

mat = np.array(mat); cor = np.corrcoef(mat.T)
rho_bar = (cor.sum() - M) / (M * (M - 1))
sd_c = mat.std(axis=0, ddof=1).mean()
print()
print(f"winner's reported number        {np.mean(rep):.3f}")
print(f"the field's reported number     {np.mean(field):.3f}")
print(f"winner's ACTUAL next window     {np.mean(fut):.3f}   <- what shipping it delivers")
print(f"optimism of the published number {(np.mean(fut)/np.mean(rep)-1)*100:+.1f}%")
print()
print(f"candidate correlation rho       {rho_bar:.3f}   (they share the base fit)")
print(f"predicted winner, independent   {np.mean(field) + expected_min_of_m(M)*sd_c:.3f}")
print(f"predicted winner, equicorrelated {np.mean(field) + expected_min_of_m(M, rho_bar)*sd_c:.3f}")
print(f"measured winner                 {np.mean(rep):.3f}")
'''
)

md(
    """
The prediction lands within a few thousandths, and the independent-candidate version - the one
you would write down if you ignored that the candidates share a base model - is far too
pessimistic. Note also the second line of the scoreboard: **selecting the winner delivers
nothing**, because the candidates were equally good. All of the apparent improvement was the
minimum operator, and all of it evaporates on the next window.
"""
)

# ---------------------------------------------------------------- 5. k-fold
md(
    """
## 4. Random k-fold on a lag table names the wrong culprit

"Never shuffle a time series" is the most repeated rule in this area. It is worth measuring
rather than repeating. Build the supervised table everyone builds - rows of `(y[i-p..i-1] ->
y[i])` - and compare a shuffled k-fold against an honest future evaluation, sweeping the noise
autocorrelation.
"""
)

code(
    '''
def lag_table(y, p):
    n = len(y)
    return np.column_stack([y[i:n - p + i] for i in range(p)]), y[p:]

def _fit(X, z, lam=1e-6):
    Xb = np.column_stack([np.ones(len(X)), X])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ z)

_pred = lambda b, X: np.column_stack([np.ones(len(X)), X]) @ b

def kfold_rmse(y, p, k, rng, scheme="random", embargo=0):
    X, z = lag_table(y, p); n = len(z); idx = np.arange(n)
    if scheme == "random":
        idx = rng.permutation(n)
    errs = []
    for f in np.array_split(idx, k):
        test = np.asarray(f)
        if scheme == "random":
            train = np.setdiff1d(idx, test)
        else:
            lo, hi = test.min() - embargo, test.max() + embargo
            train = idx[(idx < lo) | (idx > hi)]
        if len(train) <= p + 2:
            continue
        errs.append(z[test] - _pred(_fit(X[train], z[train]), X[test]))
    return rmse(np.concatenate(errs))

def honest_next_rmse(y_full, n_train, p):
    X, z = lag_table(y_full[:n_train], p); beta = _fit(X, z)
    Xf, zf = lag_table(y_full[n_train - p:], p)
    return rmse(zf - _pred(beta, Xf))

POOL, FUT, P, K = 240, 120, 6, 5
print(f"{'AR(1) rho':>10}{'shuffled k-fold':>18}{'blocked+embargo':>18}{'honest future':>16}{'optimism':>11}")
for rho in (0.0, 0.3, 0.6, 0.9):
    rng = np.random.default_rng(int(rho * 100) + 5)
    kf, bl, hon = [], [], []
    for _ in range(150):
        yy = make_series(rng, POOL + FUT, trend=0.0, rho=rho, sigma=4.0)
        kf.append(kfold_rmse(yy[:POOL], P, K, rng, "random"))
        bl.append(kfold_rmse(yy[:POOL], P, K, rng, "blocked", embargo=P))
        hon.append(honest_next_rmse(yy, POOL, P))
    print(f"{rho:>10.2f}{np.mean(kf):>18.3f}{np.mean(bl):>18.3f}{np.mean(hon):>16.3f}"
          f"{(np.mean(kf)/np.mean(hon)-1)*100:>10.1f}%")
'''
)

md(
    """
Essentially no optimism at any autocorrelation. The reason is structural: **the row already
contains its neighbours.** A lag table's features ARE the values the shuffle is accused of
leaking, so a six-parameter model has nothing extra to do with them.

The leak is real, but it is a property of model CAPACITY rather than of the series. Hold rho
fixed at 0.6 and grow the lag count:
"""
)

code(
    '''
rng = np.random.default_rng(77)
print(f"{'lags p':>8}{'shuffled CV':>15}{'honest future':>16}{'optimism':>11}")
for p in (3, 12, 40, 80):
    kf, hon = [], []
    for _ in range(120):
        yy = make_series(rng, POOL + FUT, trend=0.0, rho=0.6, sigma=4.0)
        kf.append(kfold_rmse(yy[:POOL], p, K, rng, "random"))
        hon.append(honest_next_rmse(yy, POOL, p))
    print(f"{p:>8}{np.mean(kf):>15.3f}{np.mean(hon):>16.3f}{(np.mean(kf)/np.mean(hon)-1)*100:>10.1f}%")
'''
)

# ---------------------------------------------------------------- 6. k_eff
md(
    """
## 5. Folds that overlap are not folds

Rolling origin looks like the fix, and it partly is. But a 12-step window stepped by 1 shares
11 of its 12 test points with its neighbour, and every origin also shares the same training
history and the same realised series. So the standard error people publish - `sd across folds
/ sqrt(k)` - is not a standard error.

`k_eff` is the number of INDEPENDENT folds whose SE would equal the reported one. It is
measurable here because the series can be redrawn.
"""
)

code(
    '''
def effective_folds(sd_folds, true_sd):
    return float((sd_folds / true_sd) ** 2) if true_sd > 0 else float("nan")

MIN_TRAIN = 96
print(f"{'protocol':<22}{'reported SE':>13}{'true SE':>10}{'k_eff':>8}{'of k':>7}")
for k, step in [(20, 1), (20, 3), (20, 6), (20, 12), (8, 12)]:
    rng = np.random.default_rng(31)
    means, sdf = [], []
    for _ in range(200):
        yy = make_series(rng, N + 60)
        Ei = rolling_origin(yy, f_trend_season, H, k=k, min_train=MIN_TRAIN, step=step)
        pr = np.array([rmse(r[~np.isnan(r)]) for r in Ei])
        means.append(pr.mean()); sdf.append(pr.std(ddof=1))
    keff = effective_folds(float(np.mean(sdf)), float(np.std(means, ddof=1)))
    print(f"{f'k={k} step={step}':<22}{np.mean(sdf)/np.sqrt(k):>13.4f}"
          f"{np.std(means, ddof=1):>10.4f}{keff:>8.1f}{k:>7}")
'''
)

md(
    """
Twenty overlapping origins are worth **about one** independent fold. Non-overlapping ones do
better but still do not reach k, because they share training data and a single realisation of
the process.
"""
)

# ---------------------------------------------------------------- 7. scoreboard
md(
    """
## 6. The scoreboard: every protocol as an estimator

Each protocol claims to report the same quantity. Score them the way you would score any
estimator - bias, spread, and RMSE against the truth.
"""
)

code(
    '''
target = TRUE["trend+season"]
rng = np.random.default_rng(71)
prot = {"single split": (1, 1, "expanding"), "rolling k=5": (5, 1, "expanding"),
        "rolling k=20 (overlap 11/12)": (20, 1, "expanding"),
        "rolling step 12 (no overlap)": (20, 12, "expanding"),
        "sliding k=20 (train 72)": (20, 1, "sliding")}
out = {k: [] for k in prot}
for _ in range(300):
    yy = make_series(rng, N)
    for label, (k, step, win) in prot.items():
        if label == "single split":
            out[label].append(rmse(single_split(yy, f_trend_season, H)))
            continue
        Ei = rolling_origin(yy, f_trend_season, H, k=k, min_train=96, step=step,
                            window=win, train_len=72)
        out[label].append(rmse(Ei[~np.isnan(Ei)]))

print(f"estimand: true 12-step RMSE = {target:.3f}")
print()
print(f"{'protocol':<32}{'origins':>9}{'bias':>9}{'sd':>9}{'RMSE of the estimate':>23}")
for label, v in out.items():
    a = np.array(v)
    nf = 1 if label == "single split" else len(origins(N, H, prot[label][0], 96, prot[label][1]))
    print(f"{label:<32}{nf:>9}{a.mean()-target:>+9.3f}{a.std(ddof=1):>9.3f}"
          f"{np.sqrt(np.mean((a-target)**2)):>23.3f}")
'''
)

md(
    """
Every protocol here is nearly **unbiased**. What separates them is the noise of the number they
hand you - and that noise does not fall like `sqrt(k)`. Twenty overlapping origins buy about
1.4x against a single split where `sqrt(20) = 4.47`; the **seven** non-overlapping origins
available in the same series buy about 2.6x.

Fewer folds, a better estimate. The practical rule that falls out: **step your origins by the
horizon**, and report the spread, not just the mean.
"""
)

# ---------------------------------------------------------------- 8. try your own
md(
    """
## 7. Try your own

Everything above is one process. Change it - break the series, make the noise sticky, lengthen
the horizon - and rerun the comparison that matters to you.
"""
)

code(
    '''
# # A regime break 30 points before the deployment origin. Which window should you use,
# # and can the backtest tell you? (In the full evidence run it cannot: 0.075 accuracy.)
# rng = np.random.default_rng(0)
# for label, tl in [("expanding", None), ("sliding 36", 36), ("sliding 72", 72)]:
#     scores = []
#     for _ in range(200):
#         yy = make_series(rng, 220, break_at=178, break_shift=25.0)
#         a, p = forecast_at(yy, f_trend_season, 208, H, train_len=tl)
#         scores.append(rmse(a - p))
#     print(f"{label:<12}{np.mean(scores):>8.3f}")

# # Your own model: anything with the signature (y_train, h, period) -> h forecasts.
# def my_model(y, h, period=PERIOD):
#     return np.repeat(np.median(y[-24:]), h)
# print(true_h_step_error(my_model, H, N - H, 400, np.random.default_rng(1)))
'''
)

md(
    f"""
## What this build measured

| Finding | Number |
|---|---|
| Origin choice, one series, one model | 60 origins span 3.04-5.75 RMSE against a truth of 4.02 |
| Two closest models ranked backwards by one split | 0.317 |
| Split understates truth in proportion to a model's error spread | 0.5% to 10.8%, correlation 0.972 |
| `seasonal naive` vs `flexible deg 6` ranked backwards | 0.527 - worse than a coin flip |
| Best-of-8 equally-good candidates: published vs delivered | 3.663 reported, 4.379 actual (+19.6%) |
| Selection gain over picking at random | -1.4% |
| Winner predicted from `E[min of 8]` at rho=0.68 | 3.691 predicted, 3.663 measured |
| Shuffled k-fold optimism, p=6 lags, any rho | +0.2% to +0.7% |
| Shuffled k-fold optimism, p=80 lags, rho=0.6 | +30.3% |
| k_eff of 20 origins stepped by 1 (overlap 11/12) | 1.0 |
| Estimator RMSE: 20 overlapping origins vs 7 non-overlapping | 0.621 vs 0.341 |
| Backtest picks the right training window after a level break | 0.075 of the time |
| MAPE's optimum on a CV-30% series | 6 units BELOW the truth (15% of the level) |

**The full build:**
- `python evidence.py` - the eight-section measurement, ~6s
- `python -m pytest` - 25 assertions behind every number
- `streamlit run app.py` - pick a protocol, see the number it reports and the number it means
- `python make_chart.py` - the four-panel audit figure

**Reading, for the results this build re-derives rather than cites:** Tashman (2000) on
out-of-sample tests and rolling origins; Bergmeir & Benitez (2012) and Bergmeir, Hyndman &
Koo (2018) on cross-validation for time series; Hyndman & Koehler (2006) on scale-free error
measures and MAPE's asymmetry; Bailey et al. (2014) on backtest overfitting under selection;
Lopez de Prado (2018) on purging and embargoing.

[Back to the portfolio index](https://github.com/{REPO})
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("demo.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote demo.ipynb with {len(cells)} cells")
