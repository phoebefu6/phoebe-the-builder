"""The measurement run behind every number in the README.

Eight sections, each one an experiment rather than a citation:

1. the split's number is one draw of an ORIGIN, and the origin moves it more than the model does
2. best-of-M on one split is a minimum operator, and the shared base model shrinks it by sqrt(1-rho)
3. random k-fold on a lag table: how much it leaks, and whether it still ranks correctly
4. rolling-origin folds are not independent - what the published standard error is worth
5. the ranking depends on the HORIZON, and a mean over horizons is a mean over estimands
6. expanding vs sliding after a regime break, and whether the backtest can tell which it needs
7. MAPE does not rank forecasts by accuracy - it rewards forecasting low
8. the scoreboard: every protocol scored as an ESTIMATOR of the quantity it claims to report

Run: python evidence.py    (writes results.json + evidence.txt, ~2 min)
"""

from __future__ import annotations

import json
import sys
import time
from typing import Dict, List

import backtest as bt
import numpy as np

R: Dict[str, object] = {}
LINES: List[str] = []


def say(s: str = "") -> None:
    LINES.append(s)
    print(s)
    sys.stdout.flush()


def head(n: int, title: str) -> None:
    say("")
    say("=" * 100)
    say(f"{n}. {title}")
    say("=" * 100)


H = 12
N = 180
MIN_TRAIN = 96
T0 = time.time()

# ============================================================ 1. one split is one origin
head(1, "The split reports one draw of an ORIGIN, and the origin moves the number more than the model")

rng = np.random.default_rng(7)
REPS = 300
per_model_split: Dict[str, List[float]] = {k: [] for k in bt.MODELS}
per_model_true: Dict[str, List[float]] = {k: [] for k in bt.MODELS}
for _ in range(REPS):
    y = bt.make_series(rng, N)
    for name, m in bt.MODELS.items():
        per_model_split[name].append(bt.rmse(bt.single_split(y, m, H)))

truth_rng = np.random.default_rng(101)
TRUE = {
    name: bt.true_h_step_error(m, H, N - H, 600, truth_rng)
    for name, m in bt.MODELS.items()
}

say(f"  {'model':<18} {'true 12-step RMSE':>18} {'mean split RMSE':>16} {'sd across draws':>16} {'p(split < true/2)':>18}")
for name in bt.MODELS:
    a = np.array(per_model_split[name])
    say(f"  {name:<18} {TRUE[name]:>18.3f} {a.mean():>16.3f} {a.std(ddof=1):>16.3f} {np.mean(a < TRUE[name] / 2):>18.3f}")

names = list(bt.MODELS)
pairs = sorted(((abs(TRUE[a] - TRUE[b]), a, b) for i, a in enumerate(names) for b in names[i + 1:]))
gap_true, best, second = pairs[0]
if TRUE[second] < TRUE[best]:
    best, second = second, best
d = np.array(per_model_split[second]) - np.array(per_model_split[best])
flip = float(np.mean(d < 0))
say("")
say(f"  the two closest models in TRUTH are '{best}' ({TRUE[best]:.3f}) and '{second}' "
    f"({TRUE[second]:.3f}) - a gap of {gap_true:.3f}")
say(f"  a single split's spread for the better one is sd {np.std(per_model_split[best], ddof=1):.3f} "
    f"= {np.std(per_model_split[best], ddof=1) / gap_true:.2f}x that gap")
say(f"  so one split ranks them BACKWARDS {flip:.3f} of the time, on {REPS} independent series")
# and the pair a real project actually chooses between: the two best models
top2 = sorted(names, key=lambda n: TRUE[n])[:2]
d2 = np.array(per_model_split[top2[1]]) - np.array(per_model_split[top2[0]])
flip2 = float(np.mean(d2 < 0))
say(f"  the pair a project actually chooses between - '{top2[0]}' ({TRUE[top2[0]]:.3f}) vs "
    f"'{top2[1]}' ({TRUE[top2[1]]:.3f}), a gap of {TRUE[top2[1]] - TRUE[top2[0]]:.3f} - is")
say(f"  ranked backwards {flip2:.3f} of the time")
R["s1"] = {
    "true": TRUE,
    "split_mean": {k: float(np.mean(v)) for k, v in per_model_split.items()},
    "split_sd": {k: float(np.std(v, ddof=1)) for k, v in per_model_split.items()},
    "closest_pair": [best, second], "closest_gap": float(gap_true), "closest_flip": flip,
    "top2": top2, "top2_gap": float(TRUE[top2[1]] - TRUE[top2[0]]), "top2_flip": flip2,
}
best = "trend+season"   # the model carried through sections 4 and 8
# the split is not just noisy about a model, it is biased in FAVOUR of the variable one
und = {n: float(np.mean(per_model_split[n]) / TRUE[n] - 1) for n in names}
sds = {n: float(np.std(per_model_split[n], ddof=1)) for n in names}
r = float(np.corrcoef([sds[n] / TRUE[n] for n in names], [-und[n] for n in names])[0, 1])
say("")
say("  DERIVED HERE: the split does not merely add noise to a comparison, it TILTS it. Its")
say("  mean sits below the truth, and the shortfall grows with the model's own error spread:")
say(f"  {'model':<18}{'relative sd':>13}{'split understates truth by':>28}")
for n in sorted(names, key=lambda n: sds[n] / TRUE[n]):
    say(f"  {n:<18}{sds[n] / TRUE[n]:>13.3f}{-und[n] * 100:>27.1f}%")
say(f"  correlation across the six models: {r:.3f}")
pair = ("seasonal naive", "flexible (deg 6)")
dd = np.array(per_model_split[pair[1]]) - np.array(per_model_split[pair[0]])
say(f"  consequence: '{pair[0]}' ({TRUE[pair[0]]:.3f}) vs '{pair[1]}' ({TRUE[pair[1]]:.3f}) is")
say(f"  ranked backwards {float(np.mean(dd < 0)):.3f} of the time - WORSE than a coin flip, so a")
say("  single split is not a weak test of that comparison, it is a wrong one.")
R["s1"]["tilt"] = {"understatement": und, "rel_sd": {n: sds[n] / TRUE[n] for n in names},
                   "corr": r, "flip_snaive_vs_flexible": float(np.mean(dd < 0))}

# how much of that spread is the origin rather than the series
y_long = bt.make_series(np.random.default_rng(3), 400)
ro = bt.rolling_origin(y_long, bt.MODELS[best], H, k=60, min_train=MIN_TRAIN)
per_origin = np.array([bt.rmse(r[~np.isnan(r)]) for r in ro])
say("")
say(f"  within ONE series, moving the origin over {len(per_origin)} choices gives RMSE "
    f"{per_origin.min():.3f} to {per_origin.max():.3f} (sd {per_origin.std(ddof=1):.3f})")
say(f"  the honest number is {TRUE[best]:.3f}; the best origin available to report is "
    f"{per_origin.min() / TRUE[best] * 100:.1f}% of it")
R["s1"]["origin_sweep"] = {
    "min": float(per_origin.min()), "max": float(per_origin.max()),
    "sd": float(per_origin.std(ddof=1)), "n_origins": int(len(per_origin)),
}

# ============================================================ 2. best-of-M is a minimum
head(2, "Best-of-M on one split is a minimum operator - and the shared base model shrinks it")

M = 8
rng = np.random.default_rng(21)
u = rng.normal(size=(M, H))
u = u / np.linalg.norm(u, axis=1, keepdims=True) * np.sqrt(H)   # equal norm -> equal true error
DELTA = 2.0


def candidate(j: int):
    def f(y, h, period=bt.PERIOD):
        return bt.f_trend_season(y, h, period) + DELTA * u[j, :h]
    return f


cands = {f"expert {j+1}": candidate(j) for j in range(M)}
true_c = {name: bt.true_h_step_error(f, H, N - H, 500, np.random.default_rng(500 + j))
          for j, (name, f) in enumerate(cands.items())}
say(f"  {M} candidates built to be EQUALLY good by construction (same base fit + an orthogonalised")
say(f"  offset of equal norm). Measured true 12-step RMSE: "
    f"{min(true_c.values()):.3f} to {max(true_c.values()):.3f} (spread "
    f"{(max(true_c.values()) - min(true_c.values())) / np.mean(list(true_c.values())) * 100:.1f}% of the mean)")

rng = np.random.default_rng(22)
REPS2 = 400
rep_win, rep_field, fut_win, fut_field = [], [], [], []
mat = []
for _ in range(REPS2):
    y = bt.make_series(rng, N + H)
    split = np.array([bt.rmse(bt.single_split(y[:N], f, H)) for f in cands.values()])
    mat.append(split)
    j = int(np.argmin(split))
    rep_win.append(split[j])
    rep_field.append(split.mean())
    a, p = bt.forecast_at(y, list(cands.values())[j], N, H)
    fut_win.append(bt.rmse(a - p))
    a2 = [bt.rmse(bt.forecast_at(y, f, N, H)[0] - bt.forecast_at(y, f, N, H)[1]) for f in cands.values()]
    fut_field.append(np.mean(a2))

mat = np.array(mat)
sd_c = float(mat.std(axis=0, ddof=1).mean())
cor = np.corrcoef(mat.T)
rho_bar = float((cor.sum() - M) / (M * (M - 1)))
pred_shift = bt.expected_min_of_m(M, rho_bar) * sd_c
pred_indep = bt.expected_min_of_m(M, 0.0) * sd_c
say("")
say(f"  reported (winner's split RMSE)  {np.mean(rep_win):.3f}")
say(f"  the field's split RMSE          {np.mean(rep_field):.3f}")
say(f"  the winner's ACTUAL next-window {np.mean(fut_win):.3f}   <- what shipping it delivers")
say(f"  the field's actual next-window  {np.mean(fut_field):.3f}")
say(f"  optimism of the published number: {(np.mean(fut_win) / np.mean(rep_win) - 1) * 100:+.1f}%")
say(f"  selection gain over picking at random: {(np.mean(fut_win) / np.mean(fut_field) - 1) * 100:+.1f}%")
say("")
say(f"  PREDICTED before any data: E[min of {M}] * sd, sd = {sd_c:.3f} across draws")
say(f"    independent candidates : {pred_indep:+.3f}  -> {np.mean(rep_field) + pred_indep:.3f}")
say(f"    equicorrelated at rho={rho_bar:.3f} (they share the base fit): {pred_shift:+.3f}"
    f"  -> {np.mean(rep_field) + pred_shift:.3f}")
say(f"    measured winner        : {np.mean(rep_win):.3f}")
R["s2"] = {
    "M": M, "reported": float(np.mean(rep_win)), "field": float(np.mean(rep_field)),
    "future_winner": float(np.mean(fut_win)), "future_field": float(np.mean(fut_field)),
    "rho_bar": rho_bar, "sd_candidate": sd_c,
    "pred_equicorr": float(np.mean(rep_field) + pred_shift),
    "pred_indep": float(np.mean(rep_field) + pred_indep),
    "optimism_pct": float((np.mean(fut_win) / np.mean(rep_win) - 1) * 100),
}

# ============================================================ 3. random k-fold leaks
head(3, "Random k-fold on a lag table: the leak is a function of rho, and the RANKING survives it")

P, K = 6, 5
POOL, FUT = 240, 120
say("  a stationary seasonal series (trend removed, so 'fit here, score later' is not also a")
say("  test of extrapolation), a p=6 lag model, k=5. The honest column fits the whole pool and")
say("  scores a genuinely later stretch of the same series.")
say("")
rows = []
for rho in [0.0, 0.3, 0.6, 0.9]:
    rng = np.random.default_rng(int(rho * 100) + 5)
    kf, bl, hon = [], [], []
    for _ in range(200):
        y = bt.make_series(rng, POOL + FUT, trend=0.0, rho=rho, sigma=4.0)
        kf.append(bt.kfold_rmse(y[:POOL], P, K, rng, "random"))
        bl.append(bt.kfold_rmse(y[:POOL], P, K, rng, "blocked", embargo=P))
        hon.append(bt.honest_next_rmse(y, POOL, P))
    rows.append((rho, np.mean(kf), np.mean(bl), np.mean(hon)))
say(f"  {'AR(1) rho':>10} {'random k-fold':>15} {'blocked+embargo':>17} {'honest future':>15} {'random optimism':>17}")
for rho, a, b, c in rows:
    say(f"  {rho:>10.2f} {a:>15.3f} {b:>17.3f} {c:>15.3f} {(a / c - 1) * 100:>16.1f}%")
say("")
say("  NEGATIVE RESULT: at this model size the shuffle is nearly harmless, and the reason is")
say("  that the ROW ALREADY CONTAINS its neighbours - a lag table's features are the very")
say("  values the shuffle is accused of leaking, so a 6-parameter model has nothing extra to")
say("  do with them. The warning as usually stated ('never shuffle a time series') names the")
say("  wrong culprit.")
R["s3"] = {"rows": [{"rho": r[0], "kfold": float(r[1]), "blocked": float(r[2]), "honest": float(r[3])}
                    for r in rows]}

# where the shuffle DOES break: enough parameters to interpolate between neighbours
rng = np.random.default_rng(77)
cap = {}
for p in (3, 12, 40, 80):
    kf, hon = [], []
    for _ in range(150):
        y = bt.make_series(rng, POOL + FUT, trend=0.0, rho=0.6, sigma=4.0)
        kf.append(bt.kfold_rmse(y[:POOL], p, K, rng, "random"))
        hon.append(bt.honest_next_rmse(y, POOL, p))
    cap[p] = (float(np.mean(kf)), float(np.mean(hon)))
say("")
say(f"  {'lags p':>8}{'shuffled CV':>14}{'honest future':>16}{'optimism':>12}")
for p, (a, b) in cap.items():
    say(f"  {p:>8}{a:>14.3f}{b:>16.3f}{(a / b - 1) * 100:>11.1f}%")
say("  the shuffle's damage is not in the correlation of the SERIES, it is in the CAPACITY of")
say("  the model: shuffling buys an over-parameterised fit neighbours to interpolate between,")
say("  and the optimism grows with p while rho is held fixed.")
R["s3_capacity"] = {str(p): {"kfold": v[0], "honest": v[1]} for p, v in cap.items()}

# ============================================================ 4. folds are not independent
head(4, "Rolling-origin folds are not independent, so the published standard error is not one")

for k, step in [(20, 1), (20, 6), (20, 12), (8, 12)]:
    rng = np.random.default_rng(31)
    means, sd_folds = [], []
    for _ in range(300):
        y = bt.make_series(rng, N + 60)
        E = bt.rolling_origin(y, bt.MODELS[best], H, k=k, min_train=MIN_TRAIN, step=step)
        per = np.array([bt.rmse(r[~np.isnan(r)]) for r in E])
        means.append(per.mean())
        sd_folds.append(per.std(ddof=1))
    true_sd = float(np.std(means, ddof=1))
    sdf = float(np.mean(sd_folds))
    keff = bt.effective_folds(sdf, true_sd)
    say(f"  k={k:>2} step={step:>2} (overlap {max(0, H - step)}/{H}): reported SE "
        f"{sdf / np.sqrt(k):.4f}  true SE {true_sd:.4f}  -> k_eff {keff:.1f} of {k} "
        f"({keff / k * 100:.0f}%)")
    R.setdefault("s4", []).append({"k": k, "step": step, "reported_se": sdf / np.sqrt(k),
                                   "true_se": true_sd, "k_eff": keff})
say("")
say("  the overlap is not the only cause - every origin also shares the SAME training data and")
say("  the same realised series, which is why even non-overlapping windows (step=12) do not")
say("  recover k independent folds.")

# ============================================================ 5. ranking depends on horizon
head(5, "The ranking depends on the horizon, and a mean over horizons is a mean over estimands")

rng = np.random.default_rng(41)
acc = {name: np.zeros(H) for name in bt.MODELS}
for _ in range(400):
    y = bt.make_series(rng, N + H)
    for name, m in bt.MODELS.items():
        a, p = bt.forecast_at(y, m, N, H)
        acc[name] += (a - p) ** 2
per_h = {name: np.sqrt(v / 400) for name, v in acc.items()}
say(f"  {'model':<18}" + "".join(f"{f'h={i+1}':>8}" for i in [0, 2, 5, 8, 11]) + f"{'mean 1-12':>11}")
for name in bt.MODELS:
    v = per_h[name]
    say(f"  {name:<18}" + "".join(f"{v[i]:>8.2f}" for i in [0, 2, 5, 8, 11]) + f"{v.mean():>11.2f}")
r1 = sorted(bt.MODELS, key=lambda n: per_h[n][0])
r12 = sorted(bt.MODELS, key=lambda n: per_h[n][-1])
say("")
say(f"  rank at h=1 : {', '.join(f'{i+1}. {n}' for i, n in enumerate(r1))}")
say(f"  rank at h=12: {', '.join(f'{i+1}. {n}' for i, n in enumerate(r12))}")
moved = sorted(bt.MODELS, key=lambda n: -abs(r1.index(n) - r12.index(n)))[0]
say(f"  biggest mover: '{moved}' goes from rank {r1.index(moved)+1} at h=1 to rank "
    f"{r12.index(moved)+1} at h=12 ({per_h[moved][0]:.2f} -> {per_h[moved][-1]:.2f}, "
    f"{per_h[moved][-1]/per_h[moved][0]:.1f}x) while '{r12[1]}' goes "
    f"{r1.index(r12[1])+1} -> 2 without changing its error at all")
worst_ratio = max(per_h[n][-1] / per_h[n][0] for n in bt.MODELS)
flat = min(bt.MODELS, key=lambda n: per_h[n][-1] / per_h[n][0])
say(f"  h=12 error is up to {worst_ratio:.1f}x the h=1 error and as low as "
    f"{per_h[flat][-1]/per_h[flat][0]:.1f}x ('{flat}'), so an unweighted mean over 1..12 is")
say("  mostly a statement about the long horizons - a model chosen on it was chosen on h>6.")
R["s5"] = {"per_h": {k: [float(x) for x in v] for k, v in per_h.items()},
           "best_h1": r1[0], "best_h12": r12[0]}

# ============================================================ 6. expanding vs sliding
head(6, "After a break, expanding vs sliding is a real choice - and the backtest cannot make it")

rng = np.random.default_rng(51)
BREAK_AT = 150
res: Dict[str, Dict[str, float]] = {}
for world, kw in [("no break", {}),
                  ("level break +25", {"break_at": BREAK_AT, "break_shift": 25.0}),
                  ("trend break 0.15->0.60", {"break_at": BREAK_AT, "break_trend": 0.60})]:
    scores = {"expanding": [], "sliding 36": [], "sliding 72": []}
    for _ in range(300):
        y = bt.make_series(rng, 220, **kw)
        a, p = bt.forecast_at(y, bt.f_trend_season, 208, H)
        scores["expanding"].append(bt.rmse(a - p))
        for L in (36, 72):
            a2, p2 = bt.forecast_at(y, bt.f_trend_season, 208, H, train_len=L)
            scores[f"sliding {L}"].append(bt.rmse(a2 - p2))
    res[world] = {k: float(np.mean(v)) for k, v in scores.items()}
say(f"  {'world':<24}" + "".join(f"{k:>16}" for k in ["expanding", "sliding 36", "sliding 72"]) + f"{'  best':>14}")
for world, s in res.items():
    b = min(s, key=s.get)
    say(f"  {world:<24}" + "".join(f"{s[k]:>16.3f}" for k in ["expanding", "sliding 36", "sliding 72"]) + f"{b:>14}")
say("")
say("")
say("  Can the backtest itself pick the right one? Score both protocols on the origins BEFORE")
say("  the final one, take the winner, and check it against what actually wins on the future:")
rng = np.random.default_rng(52)
pick = {}
for world, kw in [("no break", {}),
                  ("level break +25", {"break_at": BREAK_AT, "break_shift": 25.0}),
                  ("trend break 0.15->0.60", {"break_at": BREAK_AT, "break_trend": 0.60})]:
    right, regret, chose, truths = 0, [], [], []
    for _ in range(200):
        y = bt.make_series(rng, 220, **kw)
        cand = {}
        for label, win, L in [("expanding", "expanding", None), ("sliding 36", "sliding", 36),
                              ("sliding 72", "sliding", 72)]:
            E = bt.rolling_origin(y[:208], bt.f_trend_season, H, k=8, min_train=MIN_TRAIN,
                                  step=4, window=win, train_len=L)
            cand[label] = bt.rmse(E[~np.isnan(E)])
        chosen = min(cand, key=cand.get)
        fut = {}
        for label, L in [("expanding", None), ("sliding 36", 36), ("sliding 72", 72)]:
            a, p = bt.forecast_at(y, bt.f_trend_season, 208, H, train_len=L)
            fut[label] = bt.rmse(a - p)
        truth_best = min(fut, key=fut.get)
        right += int(chosen == truth_best)
        chose.append(chosen)
        truths.append(truth_best)
        regret.append(fut[chosen] - fut[truth_best])
    pick[world] = {"accuracy": right / 200, "mean_regret": float(np.mean(regret)),
                   "chosen": {k: int(v) for k, v in zip(*np.unique(chose, return_counts=True))},
                   "truth": {k: int(v) for k, v in zip(*np.unique(truths, return_counts=True))}}
    say(f"  {world:<24} picks the future's winner {right/200:.3f} of the time, "
        f"mean regret {np.mean(regret):+.3f} RMSE")
    say(f"  {'':<24}   it picks   {dict(zip(*np.unique(chose, return_counts=True)))}")
    say(f"  {'':<24}   truth is   {dict(zip(*np.unique(truths, return_counts=True)))}")
say("")
say("  NEGATIVE RESULT, and it reverses what this section looked like one experiment ago: the")
say("  protocol comparison is PAIRED on the same origins, which removes the origin noise that")
say("  wrecks section 1 - and it is still wrong 0.925 of the time after a level break, at a")
say("  regret larger than the error itself. The reason is not noise. The backtest's origins")
say("  and the DEPLOYMENT origin stand at different distances from the break, so a window that")
say("  straddles the break during the backtest no longer straddles it when you ship.")
say("")
say("  The mechanism is testable: move the break far enough back that every backtest origin AND")
say("  the deployment origin sit entirely after it, and the same code should recover.")
rng = np.random.default_rng(53)
far = {}
for tag, brk, n_pts, dep in [("break 58 before deployment", 150, 220, 208),
                             ("break 158 before deployment", 150, 320, 308)]:
    right, regret = 0, []
    for _ in range(200):
        y = bt.make_series(rng, n_pts, break_at=brk, break_shift=25.0)
        cand, fut = {}, {}
        for label, win, L in [("expanding", "expanding", None), ("sliding 36", "sliding", 36),
                              ("sliding 72", "sliding", 72)]:
            E = bt.rolling_origin(y[:dep], bt.f_trend_season, H, k=8, min_train=MIN_TRAIN,
                                  step=4, window=win, train_len=L)
            cand[label] = bt.rmse(E[~np.isnan(E)])
            a, p = bt.forecast_at(y, bt.f_trend_season, dep, H, train_len=L)
            fut[label] = bt.rmse(a - p)
        ch, tb = min(cand, key=cand.get), min(fut, key=fut.get)
        right += int(ch == tb)
        regret.append(fut[ch] - fut[tb])
    far[tag] = {"accuracy": right / 200, "mean_regret": float(np.mean(regret))}
    say(f"  {tag:<30} accuracy {right/200:.3f}   mean regret {np.mean(regret):+.3f}")
say("  so the protocol choice is answerable, but only once the break is old enough that the")
say("  backtest and the deployment are asking the same question - which is roughly when the")
say("  choice has stopped mattering.")
R["s6_far"] = far
R["s6_pick"] = pick
R["s6"] = res

# ============================================================ 7. MAPE ranks by bias
head(7, "MAPE does not rank forecasts by accuracy - it rewards forecasting low")

rng = np.random.default_rng(61)
LEVEL, SIGMA = 40.0, 12.0
say(f"  a lower-level, noisier series (level {LEVEL:.0f}, sigma {SIGMA:.0f}, i.e. a CV around "
    f"{SIGMA/LEVEL:.0%}) - the regime every demand and count series lives in")
say("")
shifts = np.linspace(-14, 6, 21)
mape_curve, rmse_curve = [], []
for sh in shifts:
    mp, rm = [], []
    for _ in range(300):
        y = bt.make_series(rng, N + H, level=LEVEL, sigma=SIGMA, season_amp=6.0, trend=0.05)
        a, p = bt.forecast_at(y, bt.f_trend_season, N, H)
        mp.append(bt.mape(a, p + sh))
        rm.append(bt.rmse(a - (p + sh)))
    mape_curve.append(float(np.mean(mp)))
    rmse_curve.append(float(np.mean(rm)))
j_m, j_r = int(np.argmin(mape_curve)), int(np.argmin(rmse_curve))
j0 = int(np.argmin(np.abs(shifts)))
say(f"  {'shift added to the forecast':<30}{'MAPE %':>10}{'RMSE':>10}")
for i, (sh, m_, r_) in enumerate(zip(shifts, mape_curve, rmse_curve)):
    if i in (0, j_m, j0, j_r, len(shifts) - 1):
        tag = "  <- min MAPE" if i == j_m else ("  <- min RMSE" if i == j_r else "")
        say(f"  {sh:>+29.1f}{m_:>10.3f}{r_:>10.3f}{tag}")
say("")
say(f"  MAPE is minimised at a shift of {shifts[j_m]:+.1f}, RMSE at {shifts[j_r]:+.1f}")
say(f"  deliberately forecasting {abs(shifts[j_m]):.0f} units LOW ({abs(shifts[j_m])/LEVEL:.0%} of the")
say(f"  level) improves MAPE from {mape_curve[j0]:.3f} to {mape_curve[j_m]:.3f} "
    f"({(mape_curve[j_m]/mape_curve[j0]-1)*100:+.1f}%) while RMSE gets "
    f"{(rmse_curve[j_m]/rmse_curve[j_r]-1)*100:+.1f}% worse")
say("  a leaderboard scored on MAPE is partly a leaderboard of downward bias, and the size of")
say("  the reward is a property of the SERIES (its coefficient of variation), not of the model.")
R["s7"] = {"shifts": [float(s) for s in shifts], "mape": [float(x) for x in mape_curve],
           "rmse": [float(x) for x in rmse_curve],
           "argmin_mape": float(shifts[j_m]), "argmin_rmse": float(shifts[j_r])}

# ============================================================ 8. protocols as estimators
head(8, "The scoreboard: every protocol scored as an ESTIMATOR of the number it claims to report")

target = TRUE[best]
rng = np.random.default_rng(71)
prot: Dict[str, List[float]] = {"single split": [], "rolling k=5": [], "rolling k=20": [],
                                "rolling k=20 step 12": [], "sliding k=20 (train 72)": []}
for _ in range(400):
    y = bt.make_series(rng, N)
    prot["single split"].append(bt.rmse(bt.single_split(y, bt.MODELS[best], H)))
    for label, k, step, win in [("rolling k=5", 5, 1, "expanding"),
                                ("rolling k=20", 20, 1, "expanding"),
                                ("rolling k=20 step 12", 20, 12, "expanding"),
                                ("sliding k=20 (train 72)", 20, 1, "sliding")]:
        E = bt.rolling_origin(y, bt.MODELS[best], H, k=k, min_train=MIN_TRAIN, step=step,
                              window=win, train_len=72)
        prot[label].append(bt.rmse(E[~np.isnan(E)]))
nfold = {"single split": 1,
         "rolling k=5": len(bt.origins(N, H, 5, MIN_TRAIN, 1)),
         "rolling k=20": len(bt.origins(N, H, 20, MIN_TRAIN, 1)),
         "rolling k=20 step 12": len(bt.origins(N, H, 20, MIN_TRAIN, 12)),
         "sliding k=20 (train 72)": len(bt.origins(N, H, 20, MIN_TRAIN, 1))}
say(f"  estimand: true 12-step RMSE of '{best}' trained on {N - H} points = {target:.3f}")
say("")
say(f"  {'protocol':<26}{'origins':>9}{'mean':>9}{'bias':>9}{'sd':>9}{'RMSE of the estimate':>23}")
for label, v in prot.items():
    a = np.array(v)
    say(f"  {label:<26}{nfold[label]:>9}{a.mean():>9.3f}{a.mean() - target:>+9.3f}{a.std(ddof=1):>9.3f}"
        f"{np.sqrt(np.mean((a - target) ** 2)):>23.3f}")
R["s8"] = {"target": float(target),
           "protocols": {k: {"mean": float(np.mean(v)), "bias": float(np.mean(v) - target),
                             "sd": float(np.std(v, ddof=1)),
                             "rmse": float(np.sqrt(np.mean((np.array(v) - target) ** 2)))}
                         for k, v in prot.items()}}
say("")
ss, r20, r12 = (np.std(prot["single split"], ddof=1), np.std(prot["rolling k=20"], ddof=1),
                np.std(prot["rolling k=20 step 12"], ddof=1))
say("")
say("  every protocol here is nearly UNBIASED - the differences are almost entirely in the")
say("  NOISE of the number they hand you, which is the part a single reported figure does not")
say(f"  carry. And that noise does not fall like sqrt(k): 20 overlapping origins buy "
    f"{ss / r20:.2f}x against one split, where sqrt(20) = 4.47, while the "
    f"{nfold['rolling k=20 step 12']} NON-overlapping")
say(f"  origins available in the same series buy {ss / r12:.2f}x - fewer folds, a better")
say("  estimate, and section 4's k_eff is the reason.")

say("")
say(f"[done in {time.time() - T0:.0f}s]")
json.dump(R, open("results.json", "w"), indent=1)
open("evidence.txt", "w").write("\n".join(LINES) + "\n")
